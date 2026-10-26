"""Phase 5: Quality control and 3D consistency validation."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from core.dicom_loader import DicomVolume
from core.phantom_detector import PhantomDetectionResult, SliceStatus
from core.section_classifier import SectionClassificationResult
from core.roi_extractor import ExtractionResult, ROI
from analysis.texture_validator import validate_texture
from utils.logger import log
import config


# Minimum angular MODULATION DEPTH before a star patch counts as containing
# a star at all (see ``_dominant_rotational_order``): the HU swing of the
# arms at the winning rotational order, as a fraction of the patch's own
# 5-95 percentile contrast.
#
# This is the gate that catches a section window reaching past the physical
# star module into transition material — the ROI is still a well-formed
# circular patch, it just has no arms in it.
#
# Measured on the 336-series PCCT run: genuine star patches scored
# 0.378-0.567 (Star16 sits lower than Star8 because 16 narrower arms are
# blurred more by the reconstruction). The two patches that had run off the
# end of their module scored 0.015 and 0.023. 0.15 sits in that ~17x gap,
# ~6x above the worst defect and ~2.5x below the weakest genuine star.
#
# NOTE: this validator no longer imports ``classify_star``. That function
# remains the SECTION CLASSIFIER's discriminator and is deliberately left
# untouched — changing it would move section boundaries and invalidate
# already-exported ROI data, whereas validation is free to use the better
# measurement on the extracted patch.
STAR_MIN_MODULATION = 0.15

# Minimum pixels per arm a sampling ring must offer before its arm count is
# believed. Two pixels per cycle is the hard sampling limit; 3 leaves room
# for the reconstruction's own blur. Used to pick which rings of a star
# patch are worth reading — see ``_dominant_rotational_order``.
_MIN_PX_PER_ARM = 3.0


@dataclass
class ValidationReport:
    total_rois: int = 0
    accepted_rois: int = 0
    rejected_rois: int = 0
    section_reports: Dict[str, dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def validate_all(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    sections: SectionClassificationResult,
    extraction: ExtractionResult,
) -> ValidationReport:
    """Run all validation checks on extracted ROIs."""
    log.info("Phase 5: Validating ROIs...")

    report = ValidationReport()
    report.total_rois = len(extraction.rois)

    # 1. Validate star ROIs against the arm count their section claims
    _validate_star_sections(sections, extraction, report)

    # 2. Validate texture ROIs
    _validate_textures(extraction, sections, report)

    # 3. Validate insert completeness
    _validate_inserts(extraction, report)

    # 4. 3D consistency checks
    _validate_3d_consistency(dv, phantom, extraction, report)

    # Count accepted/rejected
    report.accepted_rois = sum(1 for r in extraction.rois if r.accepted)
    report.rejected_rois = sum(1 for r in extraction.rois if not r.accepted)

    log.info(
        f"Validation: {report.accepted_rois} accepted, "
        f"{report.rejected_rois} rejected out of {report.total_rois}"
    )

    if report.errors:
        for e in report.errors:
            log.error(f"  ERROR: {e}")
    if report.warnings:
        for w in report.warnings:
            log.warning(f"  WARNING: {w}")

    return report


def _dominant_rotational_order(
    patch: np.ndarray,
    fracs: Tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75),
    n_angles: int = 720,
    order_lo: int = 4,
    order_hi: int = 24,
) -> Tuple[int, float]:
    """Arm count of an already-centred star patch, read in POLAR space.

    Returns ``(order, dominance_ratio)`` — the median dominant rotational
    order across several sampling radii, and how far that order stands
    above the other candidates.

    Why not ``classify_star`` here
    ------------------------------
    That function answers a different question, on different pixels, and
    both differences hurt when the job is verification:

      * It re-cuts a patch from the VOLUME at the phantom WALL centre. The
        extractor deliberately does not trust that centre — it refines the
        star centre separately (``_refine_star_center``) because the wall
        track sits ~2-4 px off the feature array on soft-kernel/large-FOV
        recons. Sampling an N-fold pattern off-centre leaks power into
        N±1, so a true 16-arm star can read as 15-fold.
      * It works on the angular profile of the FFT MAGNITUDE, where an
        8-arm star's order-16 harmonic outruns its own order-8
        fundamental (measured p16/p8 ≈ 1.26 on STAR8 vs ≈ 2.70 on
        STAR16). Its rule 3 (``r16 >= 1.5 and p16 > p8``) therefore
        claims STAR16 for a genuine STAR8 — which is exactly why 193 of
        336 series in the PCCT run reported "STAR8: N/N slices failed"
        while every exported Star8 patch was a clean 8-arm star.

    The ROI patch handed in here is already centred on the refined centre,
    so the arms can be counted directly off the image with no FFT-of-FFT
    step. Measured on all 17.600 exported star patches of that run, this
    returns 16 for every Star16 and 8 for every Star8.
    """
    a = np.asarray(patch, dtype=np.float64)
    bad = np.isnan(a)
    if bad.all():
        return 0, 0.0
    if bad.any():
        a = np.where(bad, np.nanmean(a), a)

    h, w = a.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = min(h, w) / 2.0
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)

    # Only sample rings that can actually RESOLVE the finest order we are
    # willing to report. Arm width shrinks toward the hub, where neighbouring
    # arms merge into the star's solid centre, so an inner ring reports a
    # degraded count no matter how good the data is — on 500 mm-FOV recons
    # the innermost ring of a 31 px patch offers 2.1 px per arm at order 16,
    # below the 2 px/cycle sampling limit. A plain median over fixed radii
    # therefore lets the unresolvable hub outvote the resolvable rim: it
    # returned 15 instead of 16 on 46 of 17.600 exported patches that a human
    # can count 16 arms on. Requiring MIN_PX_PER_ARM keeps only rings whose
    # answer means something.
    usable = [radius * f for f in np.arange(0.35, 0.86, 0.05)
              if 2.0 * np.pi * radius * f / order_hi >= _MIN_PX_PER_ARM]
    if not usable:
        # Patch too small for the full order range — fall back to the
        # outermost rings, which are still the most resolvable ones.
        usable = [radius * f for f in (0.75, 0.80, 0.85)]

    # Average the per-ring angular profiles BEFORE the FFT. The arms are
    # radial wedges, so every ring sees the same angular phase and the true
    # order adds coherently while per-ring noise averages down. (Averaging
    # the per-ring VERDICTS instead — the old median — throws that coherence
    # away and lets one bad ring change the answer.)
    profiles = []
    for r in usable:
        ys = np.clip(np.rint(cy + r * np.sin(angles)).astype(int), 0, h - 1)
        xs = np.clip(np.rint(cx + r * np.cos(angles)).astype(int), 0, w - 1)
        p = a[ys, xs]
        profiles.append(p - p.mean())
    if not profiles:
        return 0, 0.0

    mean_profile = np.mean(profiles, axis=0)
    spectrum = np.abs(np.fft.rfft(mean_profile))
    if len(spectrum) <= order_hi:
        return 0, 0.0
    band = spectrum[order_lo:order_hi + 1]
    k = int(np.argmax(band))

    # Report MODULATION DEPTH, not the winner's lead over the other orders.
    #
    # A relative lead is not evidence that a star is present: on a patch with
    # no angular structure at all the spectrum is noise everywhere, and noise
    # still beats noise — the two slices of the 336-series run that sampled
    # module-transition material instead of the star scored 7.1 and 5.9 on
    # such a ratio, indistinguishable from a weak-but-real star. Their
    # modulation depth was 0.015 and 0.023 against 0.38-0.57 for every
    # genuine star patch measured, because the physical quantity that
    # actually separates them is how deeply the arms swing the HU, relative
    # to the patch's own contrast.
    amplitude = 2.0 * float(band[k]) / len(mean_profile)
    valid = patch[~np.isnan(patch)] if np.isnan(patch).any() else patch
    contrast = float(np.percentile(valid, 95) - np.percentile(valid, 5))
    modulation = amplitude / (contrast + 1e-9)
    return order_lo + k, float(modulation)


def _validate_star_sections(
    sections: SectionClassificationResult,
    extraction: ExtractionResult,
    report: ValidationReport,
):
    """Stage 1: the star ROI actually written to disk must have the arm
    count its section name claims.

    Validates the PRODUCT (the extracted patch) rather than re-deriving a
    second opinion from the volume — if the exported patch has 8 arms,
    "STAR8" is the right label no matter what an independent re-detection
    on differently-centred pixels would have said.
    """
    star_rois: Dict[str, List[ROI]] = {}
    for roi in extraction.rois:
        if roi.label in ("Star", "Star8", "Star16"):
            star_rois.setdefault(roi.section, []).append(roi)

    for sec in sections.sections:
        if sec.name not in ("STAR8", "STAR16"):
            continue

        expected_arms = config.SECTIONS[sec.name]["star_arms"]
        mismatches = 0
        checked = 0
        seen_orders: Dict[int, int] = {}

        for roi in star_rois.get(sec.name, []):
            order, modulation = _dominant_rotational_order(roi.pixels)
            checked += 1
            seen_orders[order] = seen_orders.get(order, 0) + 1
            if order != expected_arms or modulation < STAR_MIN_MODULATION:
                mismatches += 1

        if checked > 0 and mismatches / checked > 0.3:
            report.warnings.append(
                f"{sec.name}: {mismatches}/{checked} star ROIs are not "
                f"{expected_arms}-fold (measured orders: "
                f"{dict(sorted(seen_orders.items()))})"
            )


def _validate_textures(
    extraction: ExtractionResult,
    sections: SectionClassificationResult,
    report: ValidationReport,
):
    """Stage 2: Validate each texture ROI against its expected pattern.

    Skips texture-score warnings for the first and last
    :data:`config.TEXTURE_VALIDATION_EDGE_TRIM` slices of each star
    section. Those slices sit at the physical texture insert's z-edge,
    where slice-averaged HU still partially includes background and the
    pattern metrics are expected to degrade. Reporting them as warnings
    added 10-20 per scan without indicating any real placement issue.
    """
    texture_names = set()
    for sec_cfg in config.SECTIONS.values():
        if "textures" in sec_cfg:
            texture_names.update(sec_cfg["textures"].values())

    # Build a set of "core" (non-edge) slice indices per star section.
    edge_trim = getattr(config, "TEXTURE_VALIDATION_EDGE_TRIM", 3)
    core_slices_by_section: Dict[str, set] = {}
    for sec in sections.sections:
        if sec.name not in ("STAR8", "STAR16"):
            continue
        start = sec.start_idx + edge_trim
        end = sec.end_idx - edge_trim
        core_slices_by_section[sec.name] = (
            set(range(start, end)) if start < end else set()
        )

    for roi in extraction.rois:
        if roi.label not in texture_names:
            continue

        core = core_slices_by_section.get(roi.section)
        if core is not None and roi.slice_idx not in core:
            # Edge slice of a star section — pattern is physically partial
            # on this slice; skip the noisy warning.
            continue

        is_valid, method, score = validate_texture(roi.pixels, roi.label)
        if not is_valid:
            # Log as warning but don't reject — positions are geometry-based
            report.warnings.append(
                f"{roi.section} slice {roi.slice_idx}: {roi.label} texture score low ({method}, {score:.3f})"
            )

    # Check 4/4 texture requirement per section per slice
    for sec_name in ("STAR8", "STAR16"):
        sec_cfg = config.SECTIONS.get(sec_name)
        if not sec_cfg or "textures" not in sec_cfg:
            continue

        expected_textures = set(sec_cfg["textures"].values())

        # Group by slice
        for slice_idx, rois in extraction.per_slice.items():
            section_rois = [r for r in rois if r.section == sec_name]
            tex_rois = [r for r in section_rois if r.label in expected_textures]
            accepted_tex = [r for r in tex_rois if r.accepted]

            if tex_rois and len(accepted_tex) < len(expected_textures):
                missing = expected_textures - {r.label for r in accepted_tex}
                if missing:
                    report.warnings.append(
                        f"{sec_name} slice {slice_idx}: missing textures {missing}"
                    )


def _validate_inserts(extraction: ExtractionResult, report: ValidationReport):
    """Validate 20/20 insert detection per section."""
    for sec_name in ("INSERT1", "INSERT2"):
        sec_cfg = config.SECTIONS.get(sec_name)
        if not sec_cfg:
            continue

        expected = set(sec_cfg["outer_inserts"] + sec_cfg["inner_inserts"])

        # Check per slice
        for slice_idx, rois in extraction.per_slice.items():
            insert_rois = [
                r for r in rois
                if r.section == sec_name and r.label.startswith("Insert_")
            ]
            found_numbers = set()
            for r in insert_rois:
                try:
                    num = int(r.label.split("_")[1])
                    found_numbers.add(num)
                except (IndexError, ValueError):
                    pass

            if insert_rois and len(found_numbers) < len(expected):
                missing = expected - found_numbers
                report.warnings.append(
                    f"{sec_name} slice {slice_idx}: {len(found_numbers)}/{len(expected)} "
                    f"inserts detected, missing {sorted(missing)}"
                )


def _validate_3d_consistency(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    extraction: ExtractionResult,
    report: ValidationReport,
):
    """3D consistency checks across slices."""
    # Check phantom center drift
    valid_circles = [
        c for c in phantom.circles if c.status == SliceStatus.VALID
    ]
    if len(valid_circles) > 1:
        cx_vals = [c.center_x for c in valid_circles]
        cy_vals = [c.center_y for c in valid_circles]
        max_drift = max(
            max(cx_vals) - min(cx_vals),
            max(cy_vals) - min(cy_vals),
        )
        if max_drift > config.MAX_CENTER_DRIFT_PX:
            report.warnings.append(
                f"Phantom center drift {max_drift:.1f}px exceeds threshold "
                f"({config.MAX_CENTER_DRIFT_PX}px)"
            )

    # Check insert HU consistency across slices within a section
    insert_hu_by_label: Dict[str, List[float]] = {}
    for roi in extraction.rois:
        if roi.label.startswith("Insert_") and roi.accepted:
            key = f"{roi.section}_{roi.label}"
            valid_pixels = roi.pixels[~np.isnan(roi.pixels)]
            if valid_pixels.size > 0:
                insert_hu_by_label.setdefault(key, []).append(float(np.mean(valid_pixels)))

    for key, hu_values in insert_hu_by_label.items():
        if len(hu_values) > 3:
            std = np.std(hu_values)
            if std > 50:
                report.warnings.append(
                    f"{key}: HU std across slices = {std:.1f} (high variability)"
                )

    # Check texture statistics consistency
    _all_texture_names = set()
    for sec_cfg in config.SECTIONS.values():
        if "textures" in sec_cfg:
            _all_texture_names.update(sec_cfg["textures"].values())

    texture_stats: Dict[str, List[float]] = {}
    for roi in extraction.rois:
        if roi.label in _all_texture_names:
            if roi.accepted:
                valid = roi.pixels[~np.isnan(roi.pixels)]
                if valid.size > 0:
                    key = f"{roi.section}_{roi.label}"
                    texture_stats.setdefault(key, []).append(float(np.mean(valid)))

    for key, means in texture_stats.items():
        if len(means) > 3:
            std = np.std(means)
            if std > 100:
                report.warnings.append(
                    f"{key}: mean HU std across slices = {std:.1f}"
                )
