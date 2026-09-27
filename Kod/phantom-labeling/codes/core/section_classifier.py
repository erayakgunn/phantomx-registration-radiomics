"""Phase 2: Classify slices into STAR16, STAR8, INSERT1, INSERT2 sections."""

import math
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.dicom_loader import DicomVolume
from core.phantom_detector import PhantomDetectionResult, SliceStatus
from analysis.insert_detector import classify_slice_type
from analysis.star_classifier import classify_star
from analysis.texture_validator import texture_family
from utils.logger import log
import config


# Ham HU görüntüsündeki gerçek materyal dokusunu doğrulamak için minimum varyans eşiği.
# Keskin kernel gürültüsü ham HU varyansını 350'nin üzerine çıkaramazken, 
# gerçek doku kartları (honeycomb, line gratings) >1200 varyans üretir.
_RAW_HU_VAR_MIN = 800.0
_TEXTURE_REQUIRED_CARDINALS = 3


def _cardinal_texture_patches(
    hu_slice: np.ndarray,
    cx: float,
    cy: float,
    pixel_spacing: Tuple[float, float],
) -> List[np.ndarray]:
    """Raw HU patches at the 4 cardinal texture-card positions.

    Patches clipped to less than half their nominal side by the image
    border are dropped.
    """
    px_per_cm_y = 10.0 / pixel_spacing[0]
    px_per_cm_x = 10.0 / pixel_spacing[1]
    dist_px_x = config.TEXTURE_CENTER_DISTANCE_CM * px_per_cm_x
    dist_px_y = config.TEXTURE_CENTER_DISTANCE_CM * px_per_cm_y
    side_px_x = max(4, int(round(config.TEXTURE_SIDE_CM * px_per_cm_x * config.ROI_SCALE)))
    side_px_y = max(4, int(round(config.TEXTURE_SIDE_CM * px_per_cm_y * config.ROI_SCALE)))

    h, w = hu_slice.shape
    patches = []

    for ang_deg in (270, 0, 90, 180):
        rad = math.radians(ang_deg)
        tx = cx + dist_px_x * math.cos(rad)
        ty = cy + dist_px_y * math.sin(rad)
        x0 = max(0, int(round(tx - side_px_x / 2)))
        y0 = max(0, int(round(ty - side_px_y / 2)))
        x1 = min(w, x0 + side_px_x)
        y1 = min(h, y0 + side_px_y)

        if x1 - x0 < side_px_x // 2 or y1 - y0 < side_px_y // 2:
            continue

        patches.append(hu_slice[y0:y1, x0:x1].astype(np.float32))

    return patches


def _is_texture_card(patch: np.ndarray) -> bool:
    """True if a cardinal patch holds real texture-card material.

    1. Kriter: Yoğunluk Kapısı (Hasta kemiği, yatak veya dış plastik kalıpları eler)
    Sadece pozitif (yoğun) sapmayı eleriz: kemik/metal/yatak gibi
    kontaminasyon HU'yu YUKARI çeker. Negatif tarafı elemiyoruz çünkü
    bazı fantom tasarımlarının (örn. SEP/SOMATOM Edge Plus CalibraMics
    PhantomX, ince-cidarlı/hava-kovuğu doku desenleri) gerçek doku
    kareleri çoğunlukla hava olup ortalama HU -300..-400 aralığına
    düşer; bu fiziksel olarak geçerli bir dokudur, kontaminasyon değil.

    2. Kriter: Ham HU Varyans Kapısı
    Gürültü manipülasyonuna açık uint8 yerine doğrudan fiziksel HU varyansına bakıyoruz.
    Uniform PhantomX plastik/su kesitleri düşük varyans (<300) verir ve burada elenir.
    """
    if np.mean(patch) > 300.0:
        return False
    return np.var(patch) >= _RAW_HU_VAR_MIN


def _has_texture_squares(
    hu_slice: np.ndarray,
    cx: float,
    cy: float,
    pixel_spacing: Tuple[float, float],
) -> bool:
    """True if a STAR section's 4 cardinal texture squares are visible.

    Sweeps the 4 orthogonal angles around the phantom center and checks for
    true material variance using raw HU profiles to suppress reconstruction noise.
    """
    n_textured = sum(
        1
        for patch in _cardinal_texture_patches(hu_slice, cx, cy, pixel_spacing)
        if _is_texture_card(patch)
    )
    return n_textured >= _TEXTURE_REQUIRED_CARDINALS


def _interior_air_cavity_ok(
    hu_slice: np.ndarray, cx: float, cy: float, r_px: float
) -> bool:
    """True if the slice interior looks like the air-cavity calibration
    phantom, False if it's a solid soft-tissue (anthropomorphic body)
    section that happens to share the scan.

    See ``config.PHANTOM_MIN_LOW_HU_FRACTION`` for the physics. We measure
    the fraction of the inner ``PHANTOM_INTERIOR_SAMPLE_FRACTION`` · R disc
    that sits below ``PHANTOM_INTERIOR_LOW_HU_MAX``: the calibration phantom
    is air-cavity (fraction ≥ ~0.78), a body section is solid (≤ ~0.06).
    Robust to centring error because the body is dense everywhere.
    """
    h, w = hu_slice.shape
    sample_r = r_px * config.PHANTOM_INTERIOR_SAMPLE_FRACTION
    yy, xx = np.ogrid[:h, :w]
    interior = (xx - cx) ** 2 + (yy - cy) ** 2 < sample_r ** 2
    if not interior.any():
        return False
    vals = hu_slice[interior]
    low_frac = float(np.mean(vals < config.PHANTOM_INTERIOR_LOW_HU_MAX))
    return low_frac >= config.PHANTOM_MIN_LOW_HU_FRACTION


@dataclass
class SectionRange:
    name: str
    start_idx: int
    end_idx: int  # exclusive
    slice_count: int
    detected_start_idx: Optional[int] = None
    detected_end_idx: Optional[int] = None
    midpoint_z_mm: Optional[float] = None


@dataclass
class SectionClassificationResult:
    slice_labels: List[str]
    sections: List[SectionRange]
    inserts_first: bool = False


def _detect_partial_volume(
    hu_slice: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius_px: float,
) -> float:
    """Detect partial volume effect indicating transition zone."""
    h, w = hu_slice.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - phantom_cx) ** 2 + (yy - phantom_cy) ** 2)

    ring_mask = (dist > phantom_radius_px * 0.3) & (dist < phantom_radius_px * 0.85)
    values = hu_slice[ring_mask]

    if values.size == 0:
        return 0.0

    intermediate = np.sum((values > -700) & (values < -300))
    total = values.size
    pve_ratio = intermediate / total

    return float(pve_ratio)


def classify_sections(
    dv: DicomVolume, phantom: PhantomDetectionResult
) -> SectionClassificationResult:
    """Classify each slice into a phantom section securely."""
    log.info("Phase 2: Classifying sections with locked validation gates...")

    n = dv.num_slices
    labels = ["UNKNOWN"] * n
    star_confidences = [0.0] * n

    star_radius_px = dv.cm_to_pixels(config.STAR_RADIUS_CM)

    # Pass 1: Classify each valid slice
    for i in range(n):
        circ = phantom.circles[i]
        if circ.status == SliceStatus.INVALID:
            labels[i] = "INVALID"
            continue

        # Sample at the PER-SLICE boundary track (Phase 1 now emits a smooth,
        # outlier-rejected, drift-following track — one circle per slice, see
        # core/phantom_detector.py). This is safe against the STAR16/STAR8
        # interleaving that forced the old global-only anchor: the historical
        # culprit was RAW per-slice jitter (several px frame-to-frame); the
        # track's adjacent-slice variation is sub-pixel (it's a robust line),
        # so the texture/blob verdicts no longer flip between consecutive
        # slices of the same module. Using the track lets classification track
        # real couch drift and — critically — localizes soft-kernel / large-
        # FOV series at their true per-region centre instead of one fixed
        # volume-centre that sampled the body phantom.
        cx = circ.center_x
        cy = circ.center_y
        r_px = circ.radius

        # Hava-kovuğu kapısı (MÜHÜR): dolu yumuşak-doku (antropomorfik gövde)
        # kesiti ASLA kalibrasyon modülü olamaz. Bu taramalar yüksek-z ucunda
        # bir gövde fantomu içeriyor; yumuşak kernel / büyük FOV tespiti
        # bozup dead-center'a düşürdüğünde gövde STAR16/STAR8 olarak
        # sızıyordu. Doku-karesi/blob testlerinden ÖNCE peşinen elenir,
        # böylece aşağıdaki "back door" STAR kurtarmasından da geçemez.
        if not _interior_air_cavity_ok(dv.volume[i], cx, cy, r_px):
            labels[i] = "TRANSITION"
            continue

        # 4 cardinal karesinin varlığını PEŞİNEN kontrol et (MÜHÜR)
        has_textures = _has_texture_squares(dv.volume[i], cx, cy, dv.pixel_spacing)

        # Klasik blob sayıcı
        slice_type = classify_slice_type(dv.volume[i], cx, cy, dv.pixel_spacing, r_px)

        if slice_type == "STAR":
            # Eğer doku kareleri fiziksel olarak yoksa ASLA STAR olamaz (PhantomX engelleme)
            if not has_textures:
                labels[i] = "TRANSITION"
                continue
                
            star_class, n_peaks, conf = classify_star(dv.volume[i], cx, cy, star_radius_px)
            star_confidences[i] = conf
            if star_class in ("STAR8", "STAR16"):
                labels[i] = star_class
            else:
                labels[i] = "STAR"
            continue
            
        elif slice_type == "INSERT":
            labels[i] = "INSERT"
            continue

        # === ARKA KAPILAR KAPATILDI ===
        # Eğer bir kesit üstteki birincil testlerden geçemediyse, sırf gürültü simetrisi yüzünden
        # buralarda STAR olarak kurtarılmasına izin vermiyoruz. Yalnızca doku testi pozitifse 
        # ve hacmin güvenli iç bölgesindeyse 'STAR' olarak işaretlenir.
        if has_textures and (int(n * 0.15) < i < int(n * 0.85)):
            labels[i] = "STAR"
            continue

        pve_score = _detect_partial_volume(dv.volume[i], cx, cy, r_px)
        if pve_score > 0.05:
            labels[i] = "TRANSITION"
            continue

    # Pass 1.5: Smooth isolated STAR8/STAR16 mislabels
    _smooth_isolated_star_labels(labels, window=2)

    # Pass 2: Fill small UNKNOWN/TRANSITION gaps
    _fill_gaps(labels, "INSERT", max_gap=3)
    _fill_gaps_star(labels, max_gap=2)

    # Pass 2.5: Re-enforce the texture-square gate authoritatively
    for i in range(n):
        if labels[i] not in ("STAR", "STAR8", "STAR16"):
            continue
        circ = phantom.circles[i]
        if circ.status == SliceStatus.INVALID:
            continue
        # Same rationale as Pass 1: use the per-slice boundary track.
        cx2 = circ.center_x
        cy2 = circ.center_y

        if not _has_texture_squares(dv.volume[i], cx2, cy2, dv.pixel_spacing):
            if int(n * 0.22) < i < int(n * 0.78):
                labels[i] = "UNKNOWN"
            else:
                labels[i] = "TRANSITION"

    # Pass 3: Cluster contiguous slices
    sections = _cluster_sections(labels)

    # Pass 4: Assign section types by z-order
    sections_pre_pass4 = _cluster_sections(labels, check_min_slices=False)
    insert_indices_all = []
    for sec in sections_pre_pass4:
        if sec.name == "INSERT" and sec.slice_count >= config.MIN_SECTION_SLICES:
            for i in range(sec.start_idx, sec.end_idx):
                insert_indices_all.append(i)
    star_indices_all = [i for i, l in enumerate(labels) if l in ("STAR", "STAR8", "STAR16")]

    inserts_first = False
    if insert_indices_all and star_indices_all:
        median_insert = insert_indices_all[len(insert_indices_all) // 2]
        median_star = star_indices_all[len(star_indices_all) // 2]
        inserts_first = median_insert < median_star

    if inserts_first:
        log.info("  Detected INSERTS-FIRST scan orientation (Siemens PCCT Style)")
        last_insert_idx = insert_indices_all[-1]
        for i in range(0, last_insert_idx + 1):
            if labels[i] in ("STAR", "STAR8", "STAR16"):
                labels[i] = "TRANSITION"
    elif insert_indices_all:
        log.info("  Detected STARS-FIRST scan orientation (Canonical Style)")
        first_insert_idx = insert_indices_all[0]
        for i in range(first_insert_idx, len(labels)):
            if labels[i] in ("STAR", "STAR8", "STAR16"):
                labels[i] = "TRANSITION"

    sections = _cluster_sections(labels)
    star_groups = _find_star_groups(sections, labels, config.MIN_SECTION_SLICES, star_confidences)
    single_module_slices = max(1, round(42.0 / dv.slice_thickness))
    insert_groups = _find_insert_groups(
        labels, config.MIN_SECTION_SLICES, single_module_slices
    )

    if len(insert_groups) == 1:
        big = insert_groups[0]
        if big.slice_count >= int(1.5 * single_module_slices):
            mid = (big.start_idx + big.end_idx) // 2
            labels[mid] = "TRANSITION"
            insert_groups = _find_insert_groups(
                labels, config.MIN_SECTION_SLICES, single_module_slices
            )

    star_type_names = ["STAR8", "STAR16"] if inserts_first else ["STAR16", "STAR8"]
    insert_type_names = ["INSERT2", "INSERT1"] if inserts_first else ["INSERT1", "INSERT2"]

    star_groups.sort(key=lambda s: s.start_idx)

    # A LONE star group would inherit star_type_names[0] purely from
    # z-order, but a partial scan can contain EITHER physical module:
    # the 20251126 140kVp 20-slice series holds only the STAR8 module,
    # which z-order naming exported as "STAR16" with the wrong texture
    # names. The FFT star classifier cannot arbitrate on this protocol
    # (it reads the physical STAR8 spokes as STAR16), so ask the texture
    # cards instead — honeycombs mean STAR8, bar/checker gratings mean
    # STAR16. Only a lone group is second-guessed; with two groups the
    # z-order assignment is already anchored by relative position.
    if len(star_groups) == 1:
        resolved = _resolve_lone_star_identity(dv, phantom, star_groups[0])
        if resolved is None:
            log.info(
                f"  Lone star group: texture identity ambiguous, keeping "
                f"z-order default {star_type_names[0]}"
            )
        elif resolved != star_type_names[0]:
            log.info(
                f"  Lone star group: texture cards identify {resolved} "
                f"(overriding z-order default {star_type_names[0]})"
            )
            star_type_names[0] = resolved
        else:
            log.info(
                f"  Lone star group: texture cards confirm z-order "
                f"default {resolved}"
            )

    for idx, sec in enumerate(star_groups[:2]):
        new_name = star_type_names[idx]
        sec.name = new_name
        for i in range(sec.start_idx, sec.end_idx):
            if labels[i] in ("STAR", "STAR8", "STAR16"):
                labels[i] = new_name

    for idx, sec in enumerate(insert_groups[:2]):
        new_name = insert_type_names[idx]
        sec.name = new_name
        for i in range(sec.start_idx, sec.end_idx):
            if labels[i] == "INSERT":
                labels[i] = new_name

    # Pass 4.5: Canonicalize leftovers. The phantom physically has exactly
    # one STAR16, one STAR8 and two INSERT modules, and those are exactly
    # the groups chosen above. Any star/insert-labelled slice OUTSIDE the
    # chosen group spans is a residual fragment (e.g. a 5-slice FFT hit in
    # the STAR16→STAR8 transition zone) that kept its raw per-slice label
    # because the renaming loops only touch the chosen spans. Left alone,
    # a fragment of >= MIN_SECTION_SLICES survives clustering as a bogus
    # standalone section (seen on 20251126 Q1/80kVp: a 4 mm "STAR16"
    # fragment that section_trim then inflated to a full 30 mm window,
    # extracting transition slices as STAR16). Demote them to TRANSITION.
    covered = [False] * n
    for sec in list(star_groups[:2]) + list(insert_groups[:2]):
        for i in range(sec.start_idx, sec.end_idx):
            covered[i] = True
    stray_labels = ("STAR", "STAR8", "STAR16", "INSERT")
    n_stray = 0
    for i in range(n):
        if not covered[i] and labels[i] in stray_labels:
            labels[i] = "TRANSITION"
            n_stray += 1
    if n_stray:
        log.info(
            f"  Canonicalized {n_stray} stray star/insert slice label(s) "
            f"outside the chosen module groups to TRANSITION"
        )

    sections = _cluster_sections(labels)
    _infer_missing_star(labels, sections, config.MIN_SECTION_SLICES)

    sections = _cluster_sections(labels)

    trim = max(1, round(config.SECTION_BOUNDARY_TRIM / dv.slice_thickness))
    hard_trim_cap = max(1, round(3.0 / dv.slice_thickness))
    star_extra_trim = max(0, round(1.0 / dv.slice_thickness))
    
    for sec in sections:
        if sec.name in ("INSERT1", "INSERT2", "STAR8", "STAR16"):
            sec_trim = min(trim, hard_trim_cap, max(1, sec.slice_count // 4))
            if sec.name in ("STAR8", "STAR16"):
                sec_trim = sec_trim + star_extra_trim
            max_keep_per_side = max(0, (sec.slice_count - config.MIN_SECTION_SLICES) // 2)
            sec_trim = min(sec_trim, max_keep_per_side)
            for t in range(sec_trim):
                labels[sec.start_idx + t] = "TRANSITION"
                labels[sec.end_idx - 1 - t] = "TRANSITION"

    _finalize_small_fragments(labels)
    sections = _cluster_sections(labels, check_min_slices=True)

    return SectionClassificationResult(slice_labels=labels, sections=sections, inserts_first=inserts_first)


# En fazla bu kadar kesit örneklenir — yalnız kalan yıldız grubunun kimliği
# için grup boyunca eşit aralıklı örnekleme yeterli, her kesit gerekmez.
_LONE_STAR_MAX_SAMPLE_SLICES = 9


def _resolve_lone_star_identity(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    group: SectionRange,
) -> Optional[str]:
    """Decide from texture-card evidence whether a lone star group is
    physically STAR16 or STAR8. Returns None when the vote is not
    decisive (caller keeps the z-order default).

    The two modules carry unmistakably different texture cards: STAR8's
    are honeycombs (6-fold angular symmetry), STAR16's are bar/checker
    gratings (2-/4-fold) — see ``texture_family``. Per slice, the 4
    cardinal patches vote by majority: a physical STAR8 slice reads 3-4
    hex cards (the Sinusoidal card may read grid), a physical STAR16
    slice reads ~3 grid cards (the Square card's diagonal harmonics can
    read hex, so unanimity must not be required).
    """
    indices = list(range(group.start_idx, group.end_idx))
    if len(indices) > _LONE_STAR_MAX_SAMPLE_SLICES:
        picks = np.linspace(0, len(indices) - 1, _LONE_STAR_MAX_SAMPLE_SLICES)
        indices = sorted({indices[int(round(p))] for p in picks})

    votes = {"STAR8": 0, "STAR16": 0}
    for i in indices:
        circ = phantom.circles[i]
        if circ.status == SliceStatus.INVALID:
            continue
        n_hex = 0
        n_grid = 0
        patches = _cardinal_texture_patches(
            dv.volume[i], circ.center_x, circ.center_y, dv.pixel_spacing
        )
        for patch in patches:
            if not _is_texture_card(patch):
                continue
            if texture_family(patch) == "hex":
                n_hex += 1
            else:
                n_grid += 1
        if n_hex > n_grid:
            votes["STAR8"] += 1
        elif n_grid > n_hex:
            votes["STAR16"] += 1

    winner = max(votes, key=votes.get)
    loser = "STAR8" if winner == "STAR16" else "STAR16"
    if votes[winner] >= 2 and votes[winner] >= 2 * votes[loser]:
        return winner
    return None


def _find_star_groups(
    sections: List[SectionRange], labels: List[str], min_slices: int,
    star_confidences: Optional[List[float]] = None
) -> List[SectionRange]:
    star_indices = [i for i, l in enumerate(labels) if l in ("STAR", "STAR8", "STAR16")]
    if not star_indices:
        return []

    runs = []
    current_run = [star_indices[0]]
    for idx in star_indices[1:]:
        if idx - current_run[-1] <= config.SECTION_RUN_MERGE_GAP:
            current_run.append(idx)
        else:
            runs.append(current_run)
            current_run = [idx]
    runs.append(current_run)

    if len(runs) >= 2:
        runs.sort(key=len, reverse=True)
        if len(runs[1]) < min_slices and len(runs[0]) >= min_slices * 2:
            run = runs[0]
            split_idx = _find_star_split(run, labels, star_confidences)
            if split_idx is not None:
                g1, g2 = run[:split_idx], run[split_idx:]
                if min(len(g1), len(g2)) >= max(len(g1), len(g2)) * 0.3:
                    groups = [g1, g2]
                else:
                    mid = len(run) // 2
                    groups = [run[:mid], run[mid:]]
            else:
                mid = len(run) // 2
                groups = [run[:mid], run[mid:]]
        else:
            if len(runs[1]) < len(runs[0]) * 0.3:
                merged = sorted(runs[0] + runs[1])
                mid = len(merged) // 2
                groups = [merged[:mid], merged[mid:]]
            else:
                groups = runs[:2]
    elif len(runs) == 1 and len(runs[0]) >= min_slices * 2:
        run = runs[0]
        split_idx = _find_star_split(run, labels, star_confidences)
        if split_idx is not None:
            g1, g2 = run[:split_idx], run[split_idx:]
            if min(len(g1), len(g2)) >= max(len(g1), len(g2)) * 0.3:
                groups = [g1, g2]
            else:
                mid = len(run) // 2
                groups = [run[:mid], run[mid:]]
        else:
            mid = len(run) // 2
            groups = [run[:mid], run[mid:]]
    else:
        groups = runs

    result = []
    for group in groups:
        start = group[0]
        end = group[-1] + 1
        count = end - start
        if count >= min_slices:
            # Fill the WHOLE bridged span, including "INVALID" and stray
            # "INSERT" — not just UNKNOWN/TRANSITION. This run is, by
            # construction, bounded by confirmed star-type slices on both
            # sides within SECTION_RUN_MERGE_GAP of each other, so an
            # isolated INVALID in between is Phase 1 per-slice noise, and
            # an isolated INSERT is blob-count noise (a real insert module
            # is 40+ slices and can never sit inside a star run) — not
            # "no phantom here". Leaving either unfilled would make
            # `_cluster_sections` split this one physical module back into
            # 2+ same-named SectionRanges, which downstream code
            # (`roi_extractor`'s per-name dicts, the per-section boundary
            # trim) does not expect: on 20251126 Q1/80kVp a single stray
            # INSERT at slice 57 split STAR16 into a real section plus a
            # 5-slice fragment that was then extracted over transition
            # material.
            for i in range(start, end):
                if labels[i] in ("UNKNOWN", "TRANSITION", "INVALID", "INSERT"):
                    labels[i] = "STAR"
            result.append(SectionRange("STAR", start, end, count))

    result.sort(key=lambda s: s.slice_count, reverse=True)
    return result[:2]


def _find_insert_groups(
    labels: List[str], min_slices: int, single_module_slices: int
) -> List[SectionRange]:
    """Merge raw per-slice "INSERT" labels into (at most) the 2 physical
    insert modules, bridging SOME Phase 1 gaps the same way
    ``_find_star_groups`` does for star modules — but, unlike stars,
    insert slices carry no INSERT1-vs-INSERT2 sub-label, so there is no
    safe way to re-split an over-merged run afterwards the way
    ``_find_star_split`` does. A pure distance threshold can't stand in
    for that: on real data the INSERT1-to-INSERT2 boundary gap (6 slices
    on one archived Siemens PCCT series) was narrower than the intra-module
    Phase 1 noise gaps this needed to bridge on SEP data (up to ~16
    slices) — no single cutoff gets both right.
    Instead, merge two adjacent runs only when the gap is small AND the
    combined size still plausibly fits in ONE physical module
    (<= 1.5x ``single_module_slices``, the same bound the caller already
    uses to decide whether a single merged run needs force-splitting).
    A merge that would blow past that bound is almost certainly bridging
    two distinct modules, not patching a hole in one — so it's rejected
    and the two runs are kept separate instead.
    """
    insert_indices = [i for i, l in enumerate(labels) if l == "INSERT"]
    if not insert_indices:
        return []

    runs = []
    current_run = [insert_indices[0]]
    for idx in insert_indices[1:]:
        if idx - current_run[-1] <= 1:
            current_run.append(idx)
        else:
            runs.append(current_run)
            current_run = [idx]
    runs.append(current_run)

    max_plausible = 1.5 * single_module_slices
    merged = True
    while merged and len(runs) > 1:
        merged = False
        for k in range(len(runs) - 1):
            gap = runs[k + 1][0] - runs[k][-1]
            combined_size = runs[k + 1][-1] + 1 - runs[k][0]
            if gap <= config.SECTION_RUN_MERGE_GAP and combined_size <= max_plausible:
                runs[k:k + 2] = [runs[k] + runs[k + 1]]
                merged = True
                break

    # Keep the (up to) 2 largest runs — any further runs are noise
    # (isolated stray "INSERT" blob-count hits far from either real module).
    runs.sort(key=len, reverse=True)
    chosen_runs = runs[:2]

    result = []
    for run in chosen_runs:
        start, end = run[0], run[-1] + 1
        count = end - start
        if count < min_slices:
            continue
        # Fill the WHOLE bridged span, including "INVALID" and stray
        # star-family labels — see the matching comment in
        # _find_star_groups for why this is safe here (bounded on both
        # sides by confirmed same-type slices; an isolated star hit inside
        # an insert run is FFT/blob noise, a real star module is 40+
        # slices) and why it matters (one SectionRange per name — a single
        # stray mid-span label splits the module into 2+ sections that
        # each get boundary-trimmed separately).
        for i in range(start, end):
            if labels[i] in ("UNKNOWN", "TRANSITION", "INVALID",
                             "STAR", "STAR8", "STAR16"):
                labels[i] = "INSERT"
        result.append(SectionRange("INSERT", start, end, count))

    result.sort(key=lambda s: s.start_idx)
    return result


def _smooth_isolated_star_labels(labels: List[str], window: int = 2) -> None:
    n = len(labels)
    flips = []
    for i in range(n):
        current = labels[i]
        if current not in ("STAR8", "STAR16"):
            continue
        opposite = "STAR16" if current == "STAR8" else "STAR8"
        same, opp = 0, 0
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j == i:
                continue
            if labels[j] == current:
                same += 1
            elif labels[j] == opposite:
                opp += 1
        if same == 0 and opp >= window + 1:
            flips.append((i, opposite))
    for i, new_label in flips:
        labels[i] = new_label


def _find_star_split(star_indices: List[int], labels: List[str],
                     star_confidences: Optional[List[float]] = None) -> Optional[int]:
    if len(star_indices) < 4:
        return None

    n = len(star_indices)
    star8_positions = [k for k, idx in enumerate(star_indices) if labels[idx] == "STAR8"]
    star16_positions = [k for k, idx in enumerate(star_indices) if labels[idx] == "STAR16"]
    if (len(star8_positions) >= max(4, n // 5) and len(star16_positions) >= max(4, n // 5)):
        c8 = sum(star8_positions) / len(star8_positions)
        c16 = sum(star16_positions) / len(star16_positions)
        if abs(c8 - c16) >= 0.25 * n:
            split = int(round((c8 + c16) / 2.0))
            if n // 4 <= split <= n - n // 4:
                return split

    if star_confidences is not None:
        n = len(star_indices)
        high_conf_star16_count = 0
        drop_start = None
        drop_len = 0
        for i in range(n):
            si = star_indices[i]
            is_high_conf_star16 = (labels[si] == "STAR16" and star_confidences[si] >= 0.75)
            is_star8 = (labels[si] == "STAR8")
            if is_high_conf_star16:
                high_conf_star16_count += 1
                drop_start = None
                drop_len = 0
            elif is_star8:
                if high_conf_star16_count >= 3 and i >= max(1, n // 5):
                    if drop_start is None:
                        drop_start = i
                    drop_len += 1
                    if drop_len >= 3:
                        return drop_start

    margin = max(1, len(star_indices) // 10)
    best_cluster_start = None
    best_cluster_len = 0
    i = margin
    while i < len(star_indices) - margin:
        if labels[star_indices[i]] == "STAR8":
            cluster_start = i
            j = i
            while j < len(star_indices) and labels[star_indices[j]] == "STAR8":
                j += 1
            cluster_len = j - cluster_start
            if cluster_len > best_cluster_len:
                best_cluster_len = cluster_len
                best_cluster_start = cluster_start
            i = j
        else:
            i += 1

    if best_cluster_start is not None and best_cluster_len >= 2:
        return best_cluster_start

    n = len(star_indices)
    best_gap = 0
    best_gap_pos = None
    for i in range(1, n):
        gap = star_indices[i] - star_indices[i - 1]
        if gap >= 2:
            center_dist = abs(i / n - 0.5)
            if 0.2 <= i / n <= 0.8:
                if gap > best_gap or (gap == best_gap and best_gap_pos is not None and center_dist < abs(best_gap_pos / n - 0.5)):
                    best_gap = gap
                    best_gap_pos = i

    if best_gap_pos is not None:
        return best_gap_pos

    split_pos = int(len(star_indices) * 0.45)
    if split_pos >= 2 and len(star_indices) - split_pos >= 2:
        return split_pos

    return None


def _fill_gaps_star(labels: List[str], max_gap: int = 5):
    star_types = {"STAR", "STAR8", "STAR16"}
    fillable = {"UNKNOWN", "TRANSITION", "INVALID"}
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] in star_types:
            j = i + 1
            while j < n and labels[j] in star_types:
                j += 1
            gap_start = j
            while gap_start < n and labels[gap_start] in fillable:
                gap_start += 1
            gap_len = gap_start - j
            if gap_len > 0 and gap_len <= max_gap and gap_start < n and labels[gap_start] in star_types:
                for k in range(j, gap_start):
                    labels[k] = "STAR"
                continue
            i = j
        else:
            i += 1


def _fill_gaps(labels: List[str], target_type: str, max_gap: int = 5):
    n = len(labels)
    i = 0
    fillable = {"UNKNOWN", "TRANSITION", "INVALID"}
    while i < n:
        if labels[i] == target_type:
            j = i + 1
            while j < n and labels[j] == target_type:
                j += 1
            gap_start = j
            while gap_start < n and labels[gap_start] in fillable:
                gap_start += 1
            gap_len = gap_start - j
            if gap_len > 0 and gap_len <= max_gap and gap_start < n and labels[gap_start] == target_type:
                for k in range(j, gap_start):
                    labels[k] = target_type
                continue
            i = j
        else:
            i += 1


def _cluster_sections(labels: List[str], check_min_slices: bool = True) -> List[SectionRange]:
    if not labels:
        return []
    sections = []
    current_label = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            sections.append(SectionRange(current_label, start, i, i - start))
            current_label = labels[i]
            start = i
    sections.append(SectionRange(current_label, start, len(labels), len(labels) - start))

    if check_min_slices:
        for sec in sections:
            if sec.name in ("INSERT", "INSERT1", "INSERT2", "STAR8", "STAR16"):
                if sec.slice_count < config.MIN_SECTION_SLICES:
                    sec.name = "TRANSITION"
    return sections


def _finalize_small_fragments(labels: List[str]) -> None:
    if not labels:
        return
    small_named = ("INSERT", "INSERT1", "INSERT2", "STAR", "STAR8", "STAR16")
    n = len(labels)
    i = 0
    while i < n:
        cur = labels[i]
        if cur not in small_named:
            i += 1
            continue
        j = i
        while j < n and labels[j] == cur:
            j += 1
        if (j - i) < config.MIN_SECTION_SLICES:
            for k in range(i, j):
                labels[k] = "TRANSITION"
        i = j


def _infer_missing_star(labels: List[str], sections: List[SectionRange], min_slices: int):
    star16 = next((s for s in sections if s.name == "STAR16"), None)
    star8 = next((s for s in sections if s.name == "STAR8"), None)
    insert1 = next((s for s in sections if s.name == "INSERT1"), None)

    if star16 is None:
        return
    if insert1 is not None and star16.start_idx < insert1.start_idx:
        if star8 is not None:
            gap_start, gap_end = star8.end_idx, insert1.start_idx
        else:
            gap_start, gap_end = star16.end_idx, insert1.start_idx
    else:
        return

    if gap_end <= gap_start:
        return
    fillable = {"UNKNOWN", "TRANSITION", "STAR", "INVALID"}
    if sum(1 for i in range(gap_start, gap_end) if labels[i] in fillable) < min_slices:
        return

    for i in range(gap_start, gap_end):
        if labels[i] in fillable:
            labels[i] = "STAR8"