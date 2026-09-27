"""Trim each detected section to an inner physical-z window for cross-
protocol comparability.

Why this matters
----------------
``section_classifier`` finds each section's full extent (including the
PVE-affected boundary slices, which it labels ``TRANSITION``). The
TRANSITION trim's per-slice sensitivity differs across protocols:
thin-slice (1 mm) scans tend to flag fewer slices as TRANSITION than
thick-slice (3-5 mm) scans, so the same physical insert module ends up
contributing a different fraction of its true physical extent to ROI
extraction depending on slice thickness. That makes cross-protocol HU /
noise / texture comparisons unreliable.

What this module does
---------------------
For each section, measure the section's detected physical z-extent from
the non-TRANSITION slices, then retain only the inner fraction (default
``config.SECTION_INNER_WINDOW_FRACTION = 0.7``) centred on the section.
Any non-TRANSITION slice outside that inner window gets re-labelled
TRANSITION, so every downstream loop that already filters
``slice_labels[i] != "TRANSITION"`` automatically respects the new
boundary. The intersection-with-TRANSITION semantics are explicit: a
slice survives only if it was already non-TRANSITION AND its
slice_location falls inside the inner window.

The fraction-based approach was chosen because phantom module thicknesses
are not known a priori. With 0.7, the kept volume per section is
≈ 0.7 × (detected extent in mm), centred — protocol-independent because
the underlying physical module is the same physical thickness regardless
of how many slices sample it.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from core.dicom_loader import DicomVolume
from core.phantom_detector import PhantomDetectionResult
from core.section_classifier import SectionClassificationResult, SectionRange
from utils.logger import log
import config


def _compute_section_z_centroid(
    dv: DicomVolume,
    sec: SectionRange,
    phantom: Optional[PhantomDetectionResult],
) -> Optional[float]:
    """Return the data-driven physical-z centroid of a section in mm.

    Why this is protocol-independent
    --------------------------------
    The geometric midpoint of ``(detected_start_idx, detected_end_idx)``
    is NOT protocol-independent: section_classifier's TRANSITION pass is
    more aggressive on thick-slice acquisitions, so the same physical
    module is reported as a narrower z-range on thick scans than on
    thin ones, and its geometric midpoint shifts accordingly.

    The signal-weighted centroid IS protocol-independent. The signal
    used here is the per-slice Sobel gradient magnitude summed inside
    the phantom interior:

      * Slices inside the physical module are full of sharp features
        (insert disks, star spokes, texture squares) → high gradient.
      * PVE-fading boundary slices have blurred features → mid gradient.
      * Inter-module gap slices have only smooth uniform plastic →
        low gradient.

    ``z_centroid = Σ z_i × g_i / Σ g_i`` therefore lands on the slice
    where the section's content is densest, which IS the same physical
    z for the same physical module regardless of how it was sampled.

    Returns ``None`` when the centroid can't be computed (missing
    detected bounds, no phantom info, or zero total signal).
    """
    det_start = sec.detected_start_idx
    det_end = sec.detected_end_idx
    if det_start is None or det_end is None or det_end <= det_start:
        return None
    if det_start < 0 or det_end > dv.num_slices:
        return None
    if dv.slice_locations is None or len(dv.slice_locations) < det_end:
        return None

    z = np.asarray(dv.slice_locations, dtype=np.float64)
    h, w = dv.volume.shape[1], dv.volume.shape[2]

    # Build the interior mask once (used for every slice in the section).
    if (phantom is not None
            and getattr(phantom, "global_radius", None) is not None
            and phantom.global_radius > 0):
        cx_px = float(phantom.global_center_x)
        cy_px = float(phantom.global_center_y)
        # 0.9 × radius excludes the phantom wall (which would otherwise
        # contribute a large constant gradient peak on every slice and
        # bias toward the wall's brightness pattern).
        r_px = 0.9 * float(phantom.global_radius)
    else:
        cx_px, cy_px = w / 2.0, h / 2.0
        r_px = 0.45 * min(w, h)

    yy, xx = np.ogrid[:h, :w]
    interior = ((xx - cx_px) ** 2 + (yy - cy_px) ** 2) <= r_px * r_px

    weights: List[float] = []
    zs: List[float] = []
    for i in range(det_start, det_end):
        slc = dv.volume[i].astype(np.float32)
        gx = cv2.Sobel(slc, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(slc, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        signal = float(grad[interior].sum())
        weights.append(signal)
        zs.append(float(z[i]))

    w_arr = np.asarray(weights, dtype=np.float64)
    z_arr = np.asarray(zs, dtype=np.float64)
    total = float(w_arr.sum())
    if total <= 0:
        return None
    return float((z_arr * w_arr).sum() / total)


def trim_sections_to_inner_window(
    dv: DicomVolume,
    sections: SectionClassificationResult,
    phantom: Optional[PhantomDetectionResult] = None,
    half_mm: Optional[float] = None,
    fraction: Optional[float] = None,
) -> None:
    """Mark slices outside each section's inner z-window as TRANSITION.

    Two driving modes (mutually exclusive — fixed-mm takes precedence):

      * ``half_mm`` (default: ``config.SECTION_INNER_WINDOW_HALF_MM``) —
        when set to a positive number, every section keeps a ±N mm slab
        centred on its detected z-midpoint. Slice count scales inversely
        with slice thickness, giving a PROTOCOL-INDEPENDENT physical
        kept volume.

      * ``fraction`` (default: ``config.SECTION_INNER_WINDOW_FRACTION``)
        — fallback used only when ``half_mm`` is ``None``. Keeps the
        inner fraction of each section's detected non-TRANSITION extent.
        Not protocol-independent (detected extent itself varies with
        TRANSITION sensitivity).

    Modifies ``sections.slice_labels`` and refreshes ``sec.slice_count``
    in place.
    """
    if half_mm is None:
        half_mm = config.SECTION_INNER_WINDOW_HALF_MM
    if fraction is None:
        fraction = config.SECTION_INNER_WINDOW_FRACTION
    use_fixed_mm = half_mm is not None and half_mm > 0

    if not use_fixed_mm and (fraction >= 1.0 or fraction <= 0.0):
        log.info(
            f"Phase 2c: Section inner-window trim disabled "
            f"(half_mm={half_mm}, fraction={fraction:.2f})."
        )
        return

    if dv.slice_locations is None or len(dv.slice_locations) == 0:
        log.warning(
            "Phase 2c: No slice_locations available — cannot apply inner-"
            "window trim. Skipping."
        )
        return

    if use_fixed_mm:
        log.info(
            f"Phase 2c: Trimming each section to fixed ±{half_mm:.1f} mm "
            f"window (= {2.0 * half_mm:.1f} mm total) centred on section "
            f"z-midpoint. Slice count scales 1:1/dz across protocols."
        )
    else:
        log.info(
            f"Phase 2c: Trimming each section to inner {fraction * 100:.0f}% "
            f"of its detected physical z-extent (legacy fraction mode; not "
            f"protocol-independent)."
        )

    z = np.asarray(dv.slice_locations, dtype=np.float64)
    half_fraction = fraction / 2.0
    n_slices = len(z)

    # Names of all OTHER named sections — slices labelled with these are
    # "protected" from being reclaimed by the current section in fixed-mm
    # mode (so adjacent sections can't steal slices from each other).
    named_section_names = {
        s.name for s in sections.sections
        if s.name not in ("UNKNOWN", "INVALID", "TRANSITION")
    }

    for sec in sections.sections:
        # Only trim named functional sections; UNKNOWN / INVALID stay as-is.
        if sec.name in ("UNKNOWN", "INVALID", "TRANSITION"):
            continue

        # Snapshot the ORIGINAL section_classifier bounds before we touch
        # them. Coronal-overlay and any other downstream code that needs
        # the "as-detected" coronal slab length reads ``detected_*``.
        if sec.detected_start_idx is None:
            sec.detected_start_idx = sec.start_idx
        if sec.detected_end_idx is None:
            sec.detected_end_idx = sec.end_idx

        all_indices: List[int] = list(range(sec.start_idx, sec.end_idx))
        if not all_indices:
            continue
        all_z: List[float] = [float(z[i]) for i in all_indices]

        non_trans_indices: List[int] = [
            i for i in all_indices
            if sections.slice_labels[i] != "TRANSITION"
        ]
        non_trans_z: List[float] = [float(z[i]) for i in non_trans_indices]

        if not non_trans_indices and not use_fixed_mm:
            log.warning(
                f"  {sec.name}: 0 non-TRANSITION slices — trim skipped."
            )
            continue
        if not use_fixed_mm and len(non_trans_indices) < 2:
            log.warning(
                f"  {sec.name}: only 1 non-TRANSITION slice — too few to "
                f"estimate z-extent for fraction-based trim; skipped."
            )
            continue

        # Detected (post-classifier) z-extent for diagnostics.
        if non_trans_z:
            z_min = min(non_trans_z)
            z_max = max(non_trans_z)
        else:
            z_min = min(all_z)
            z_max = max(all_z)
        z_extent = z_max - z_min
# === MERKEZ NOKTASI HESAPLAMA (DÜZELTİLDİ) ===
        is_insert = sec.name.startswith("INSERT")

        if use_fixed_mm:
            # INSERT modülleri içi dolu silindirler olduğu için sinyal gradyanı gürültüye kayar.
            # Bu yüzden INSERT'lerde sinyal merkezini (z_centroid) aramayıp doğrudan geometrik ortayı alıyoruz.
            if is_insert:
                z_centroid = None 
            else:
                z_centroid = _compute_section_z_centroid(dv, sec, phantom)

            if z_centroid is not None:
                z_center = z_centroid
                centroid_source = "signal centroid"
            else:
                # Kesik çizgilerin tam geometrik ortası
                z_center = 0.5 * (z[sec.start_idx] + z[sec.end_idx - 1])
                centroid_source = "geometric midpoint (forced)"
            
            sec.midpoint_z_mm = z_center
            kept_half = float(half_mm)
            mode = f"fixed ±{kept_half:.1f} mm, {centroid_source}"
        else:
            # Yüzdelik (fraction) modunda da asimetrik kaymaları önlemek için:
            if is_insert:
                z_center = 0.5 * (z[sec.start_idx] + z[sec.end_idx - 1])
            else:
                z_center = 0.5 * (z_min + z_max)
                
            sec.midpoint_z_mm = z_center
            kept_half = z_extent * half_fraction
            mode = f"{fraction * 100:.0f}% of {z_extent:.1f} mm extent"
        # ===============================================

        z_lo = z_center - kept_half
        z_hi = z_center + kept_half

        # Build the set of OTHER sections' names — used for protected-claim
        # in fixed-mm mode.
        other_section_names = named_section_names - {sec.name}

        # === Fixed-mm mode: ROUND-COUNT selection ===
        # Pick the N slices closest to ``z_center``, where ``N`` =
        # round(2·half_mm / dz). This is deterministic — for a given
        # dz and half_mm, every protocol gets exactly the same count
        # (modulo "protected" slices that belong to another section).
        # The previous "z in window" rule was off-by-one in some grid
        # alignments because the window endpoints didn't always fall on
        # a slice position; round-count makes the physical kept volume
        # equal across protocols within ±dz/2.
        if use_fixed_mm:
            # Use the actual reconstruction interval, NOT dv.slice_thickness
            # (which is beam thickness — different from the physical
            # spacing between reconstructed slices when overlapping or
            # multi-detector reconstruction is used).
            #
            # MULTI-SLICE-PER-Z handling: some Siemens PCCT acquisitions
            # produce TWO slices at every unique z position (e.g. two
            # detector layers or two energy bins reconstructed separately).
            # Both slices represent the SAME physical location. If we count
            # them as independent positions we'd label only half the
            # physical extent the user expects.
            # Solution: deduplicate by unique z, compute spacing from
            # unique positions, and pick the right number of UNIQUE z
            # values. Then include EVERY slice that lives at one of the
            # selected unique z values.
            z_all = np.asarray(dv.slice_locations, dtype=np.float64)
            # Round to 4 decimals (0.1 µm) so float noise doesn't make
            # genuine duplicates appear distinct.
            unique_z = np.unique(np.round(z_all, 4))
            actual_dz = None
            if unique_z.size > 1:
                diffs_unique = np.abs(np.diff(np.sort(unique_z)))
                pos_diffs_unique = diffs_unique[diffs_unique > 1e-6]
                if pos_diffs_unique.size > 0:
                    actual_dz = float(np.median(pos_diffs_unique))
            if actual_dz is None or actual_dz <= 0:
                actual_dz = float(dv.slice_thickness) if dv.slice_thickness else 1.0
            slice_thickness = actual_dz
            # Number of UNIQUE z positions targeted. n_target slices to
            # include is computed below from the actual multiplicity.
            n_target_unique = max(1, int(round(2.0 * kept_half / slice_thickness)))
            slices_per_z = float(z_all.size) / float(unique_z.size) if unique_z.size else 1.0
            n_target = int(round(n_target_unique * slices_per_z))

            # FRAGMENT GUARD: a real module's detected extent is close to
            # the target window (a full CalibraMics module is ~40 mm vs
            # the 30 mm window). If the detected physical span is under
            # HALF the target window, this "section" is almost certainly
            # a residual classification fragment (e.g. a few transition
            # slices whose FFT briefly matched a star pattern) — expanding
            # it via reclaim would swallow genuine TRANSITION material and
            # neighbouring-module territory (seen on 20251126 Q1/80kVp:
            # a 4 mm bogus STAR16 inflated to 30 mm, `reclaimed 18`).
            # ``z_extent`` is a max-min z range, so add one slice_thickness
            # to convert to physical coverage — that keeps a legitimate
            # MIN_SECTION_SLICES=5 section on a 3 mm protocol (12 mm range
            # = 15 mm physical) exactly at the allowed boundary.
            if (z_extent + slice_thickness) < kept_half:
                log.warning(
                    f"  {sec.name}: detected span {z_extent:.1f} mm "
                    f"(~{z_extent + slice_thickness:.1f} mm physical) is under "
                    f"half the {2.0 * kept_half:.1f} mm target window — likely "
                    f"a residual classification fragment; keeping detected "
                    f"slices only (no expansion/reclaim)."
                )
                continue

            # OVER-EXTENSION CLAMP: the fixed-mm window must never reach
            # PAST the module the classifier actually found.
            #
            # The guard above only catches sections narrower than HALF the
            # window; a section between half and full window width was still
            # expanded to the full 2*kept_half and therefore sampled material
            # beyond the module's own boundary. Measured on the 336-series
            # PCCT run: 3 of 1344 sections had a detected span under the
            # 30 mm window (STAR8 at 19, 21 and 23 mm; every other section
            # sat at 33-91 mm). Their trailing slices ran off the end of the
            # star module — the exported patch was a well-formed circular ROI
            # containing no star at all (angular modulation 0.02 against a
            # 0.56 section plateau), and nothing downstream could tell.
            #
            # Clamping only ever NARROWS the window, and only for sections
            # the detector already reports as short, so the protocol
            # independence of the fixed-mm window is untouched everywhere it
            # actually applies: a section wider than the window keeps exactly
            # the same slices as before.
            detected_physical = z_extent + slice_thickness
            if detected_physical < 2.0 * kept_half:
                clamped_half = detected_physical / 2.0
                log.warning(
                    f"  {sec.name}: detected span {z_extent:.1f} mm "
                    f"(~{detected_physical:.1f} mm physical) is NARROWER than "
                    f"the {2.0 * kept_half:.1f} mm target window — clamping the "
                    f"kept window to the detected module so it cannot extend "
                    f"past the module boundary into transition material."
                )
                kept_half = clamped_half
                n_target_unique = max(1, int(round(2.0 * kept_half / slice_thickness)))
                n_target = int(round(n_target_unique * slices_per_z))

            # SLIDING-WINDOW MAX-SIGNAL selection (changed 2026-05-18):
            #
            # Previous "pick N closest to z_centroid" approach put the
            # window symmetrically around the centroid. On asymmetric
            # signal profiles (e.g. STAR sections where the trailing
            # boundary signal drops slightly while the leading boundary
            # is still clean) this puts one marginal slice at the
            # trailing edge of the kept window. The user observed this
            # as the last slice (e.g. slice 105) being visually inside
            # the boundary transition.
            #
            # New approach: among all consecutive N-unique-z windows
            # within a ±EXT_MARGIN extension of the centroid, pick the
            # window whose TOTAL GRADIENT SIGNAL is maximum. This
            # naturally avoids marginal slices on one end by swapping
            # them for cleaner slices on the other end (e.g. trade
            # marginal slice 105 with signal 0.991 for clean
            # ex-TRANSITION slice 75 with signal 1.002, giving 30
            # clean slices = 30 mm physical).
            #
            # If the slid window includes ex-TRANSITION slices they're
            # still cleaner than the alternative; we reclaim them
            # exactly as the previous two-pass code did.
            unique_z_sorted = np.sort(unique_z)
            # Find the z index closest to the centroid; the window will
            # be centred there ± half_n ± EXT_MARGIN.
            center_idx = int(np.argmin(np.abs(unique_z_sorted - z_center)))
            half_n = n_target_unique // 2
            
            # INSERT'lerde kaymayı engellemek için payı 0 yapıyoruz (merkeze kilitliyoruz).
            # STAR'lar desen aramak için ±6 dilim kaymaya devam edebilir.
            if sec.name.startswith("INSERT"):
                ext_margin = 0 
            else:
                ext_margin = 6  
                
            ext_lo = max(0, center_idx - half_n - ext_margin)
            ext_hi = min(unique_z_sorted.size,
                          center_idx + (n_target_unique - half_n) + ext_margin)
            candidates = unique_z_sorted[ext_lo:ext_hi]

            def _matching_slices(zv: float) -> List[int]:
                return [
                    i for i, zi in enumerate(z_all)
                    if abs(zi - zv) < 1e-4
                ]

            # Compute per-z gradient signal weight (reuse the same
            # interior mask as _compute_section_z_centroid). Skip any
            # z whose slices are entirely protected by another named
            # section — they can never be selected anyway.
            if phantom is not None and getattr(phantom, "global_radius", None):
                _cx = float(phantom.global_center_x)
                _cy = float(phantom.global_center_y)
                _r = 0.9 * float(phantom.global_radius)
            else:
                _h, _w = dv.volume.shape[1], dv.volume.shape[2]
                _cx, _cy = _w / 2.0, _h / 2.0
                _r = 0.45 * min(_w, _h)
            _yy, _xx = np.ogrid[:dv.volume.shape[1], :dv.volume.shape[2]]
            _interior = ((_xx - _cx) ** 2 + (_yy - _cy) ** 2) <= _r * _r

            def _z_signal(zv: float) -> float:
                idxs = _matching_slices(zv)
                if not idxs:
                    return 0.0
                # Reject z whose slices are all in another named section.
                if all(sections.slice_labels[i] in other_section_names
                       for i in idxs):
                    return -1.0  # ineligible (won't win max)
                slc = dv.volume[idxs[0]].astype(np.float32)
                gx_ = cv2.Sobel(slc, cv2.CV_32F, 1, 0, ksize=3)
                gy_ = cv2.Sobel(slc, cv2.CV_32F, 0, 1, ksize=3)
                return float(np.sqrt(gx_ * gx_ + gy_ * gy_)[_interior].sum())

            cand_signals = np.array([_z_signal(zv) for zv in candidates])

            # EXPAND-THEN-TRIM (added 2026-05-18): for STAR sections only.
            # Find the best n_target_unique + 2 consecutive entries (so
            # the cleanest 30 mm window plus a 1-z buffer on each side),
            # then trim the 1 outermost z on each end of the chosen
            # window. This guarantees:
            #   * The final labelled window is 30 z = 30 mm (cross-section
            #     consistent with inserts).
            #   * The 2 outermost z slots that the sliding window
            #     considered for its 32-z window are EXCLUDED, providing
            #     a 1-slice safety buffer between the labelled volume
            #     and the section_classifier's TRANSITION boundary.
            # On clean sections where the plateau is wider than 32 mm
            # both ends, this just trims clean buffer; on sections where
            # the plateau is exactly 30 mm and one end is marginal, the
            # buffer trims the marginal end and shifts the window to
            # the cleaner side. INSERT sections use n_target_unique
            # directly (no expansion) — they don't have the visible
            # boundary-mount issue STAR sections have.
            is_star = sec.name in ("STAR8", "STAR16")
            search_n = n_target_unique + (2 if is_star else 0)

            best_start = -1
            best_sum = -np.inf
            if len(candidates) >= search_n:
                for s_idx in range(0, len(candidates) - search_n + 1):
                    window = cand_signals[s_idx:s_idx + search_n]
                    if np.any(window < 0):
                        continue
                    total = float(window.sum())
                    if total > best_sum:
                        best_sum = total
                        best_start = s_idx

            chosen_z = set()
            reclaimed_count = 0
            kept_indices: List[int] = []

            if best_start < 0:
                # Fallback: take the closest n_target_unique even if
                # some are ineligible (extremely unlikely on real data).
                unique_z_ranked = sorted(unique_z, key=lambda zv: abs(zv - z_center))
                for zv in unique_z_ranked:
                    if len(chosen_z) >= n_target_unique:
                        break
                    matching = _matching_slices(zv)
                    if all(sections.slice_labels[i] in other_section_names
                           for i in matching):
                        continue
                    chosen_z.add(zv)
                    for i in matching:
                        if sections.slice_labels[i] in other_section_names:
                            continue
                        if sections.slice_labels[i] != sec.name:
                            sections.slice_labels[i] = sec.name
                            reclaimed_count += 1
                        kept_indices.append(i)
            else:
                # Adopt the max-signal window. For STAR sections, trim
                # 1 z from each end of the search_n window to leave a
                # 1-slice safety buffer; INSERT sections use the full
                # search_n which equals n_target_unique.
                if is_star:
                    window_z = candidates[best_start + 1
                                          : best_start + 1 + n_target_unique]
                else:
                    window_z = candidates[best_start
                                          : best_start + n_target_unique]
                for zv in window_z:
                    chosen_z.add(float(zv))
                    matching = _matching_slices(float(zv))
                    for i in matching:
                        if sections.slice_labels[i] in other_section_names:
                            continue
                        if sections.slice_labels[i] != sec.name:
                            sections.slice_labels[i] = sec.name
                            reclaimed_count += 1
                        kept_indices.append(i)

            kept_indices.sort()

            # Drop any slices that used to be this section but didn't
            # make the round-count cut — they go back to TRANSITION.
            dropped_count = 0
            kept_set = set(kept_indices)
            for i in all_indices:
                if i not in kept_set and sections.slice_labels[i] == sec.name:
                    sections.slice_labels[i] = "TRANSITION"
                    dropped_count += 1

            kept_count = len(kept_indices)
            if kept_indices:
                # Effective window from the slices actually kept.
                z_kept_min = float(min(z[i] for i in kept_indices))
                z_kept_max = float(max(z[i] for i in kept_indices))
                eff_z_lo = min(z_kept_min, z_kept_max)
                eff_z_hi = max(z_kept_min, z_kept_max)
                # Expand sec range to encompass everything we kept.
                sec.start_idx = min(kept_indices)
                sec.end_idx = max(kept_indices) + 1
            else:
                eff_z_lo = eff_z_hi = z_center
            sec.slice_count = kept_count

            extra = ""
            if reclaimed_count > 0:
                extra += f", reclaimed {reclaimed_count}"
            short = ""
            if kept_count < n_target:
                short = (
                    f" (TARGET WAS {n_target} but {n_target - kept_count} "
                    f"slot(s) blocked by neighbouring sections)"
                )
            n_unique_kept = len(chosen_z)
            mult_str = (
                f" ({slices_per_z:.1f} slices per unique z)"
                if slices_per_z > 1.05 else ""
            )
            phys_mm = n_unique_kept * slice_thickness
            log.info(
                f"  {sec.name}: detected {z_min:.1f}..{z_max:.1f} mm "
                f"(span {z_extent:.1f} mm), midpoint z={z_center:.1f} mm, "
                f"target {n_target_unique} unique-z = {n_target} slices "
                f"(= {2.0 * kept_half:.1f} mm at dz={slice_thickness:.2f} mm"
                f"{mult_str}, {mode}); "
                f"kept {kept_count} slices ({n_unique_kept} unique z = "
                f"{phys_mm:.1f} mm physical) spanning "
                f"{min(eff_z_lo, eff_z_hi):.1f}..{max(eff_z_lo, eff_z_hi):.1f} mm"
                f", trimmed {dropped_count}{extra}{short}"
            )
        else:
            # === Fraction mode: bounded reclaim (only inside [start,end)) ===
            kept_count = 0
            dropped_count = 0
            for i in all_indices:
                in_window = z_lo <= z[i] <= z_hi
                if in_window:
                    kept_count += 1
                else:
                    if sections.slice_labels[i] != "TRANSITION":
                        sections.slice_labels[i] = "TRANSITION"
                        dropped_count += 1
            sec.slice_count = kept_count
            log.info(
                f"  {sec.name}: detected {z_min:.1f}..{z_max:.1f} mm "
                f"(span {z_extent:.1f} mm), kept window "
                f"{z_lo:.1f}..{z_hi:.1f} mm ({2.0 * kept_half:.1f} mm, "
                f"{mode}), slices kept {kept_count}, trimmed {dropped_count}"
            )
