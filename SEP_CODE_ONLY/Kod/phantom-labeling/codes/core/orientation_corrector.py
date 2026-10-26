"""Detects upside-down phantom acquisitions and canonicalises the volume.

A phantom placed upside-down in the scanner produces:
  * Reversed Z-order of sections (INSERT2 → INSERT1 → STAR8 → STAR16
    instead of STAR16 → STAR8 → INSERT1 → INSERT2). ``section_classifier``
    detects this and exposes it as ``SectionClassificationResult.inserts_first``;
    the section *names* it assigns are already correct (so INSERT1 always
    refers to the same physical module regardless of orientation).
  * Mirrored X-axis within every slice — textures and inserts at angle θ
    appear at (180° − θ). This breaks the angle-based texture matching
    in ``rotation_detector`` (e.g. "Vertical" expected at 0° gets found
    at 180° where "Horizontal" should be) and makes overlays look weird
    to anyone used to the canonical orientation.

When an upside-down acquisition is detected, this module flips the volume
in BOTH axes (Z and X) in place and updates Phase 1 (phantom centres) and
Phase 2 (section indices + per-slice labels) so the rest of the pipeline
runs on a canonical "stars-first, non-mirrored" volume. After the flip:

  * Section z-order matches the canonical convention.
  * Texture angles match ``config.SECTIONS[...]["textures"]``.
  * Painted overlays show inserts in their canonical positions.
  * The ``inserts_first`` flag is cleared so downstream code doesn't
    re-apply any compensation.

Only the volume is mutated — DICOM headers (slice_locations) are also
reversed for consistency. The original on-disk DICOM files are not modified.
"""

from __future__ import annotations

import numpy as np

from core.dicom_loader import DicomVolume
from core.phantom_detector import PhantomDetectionResult
from core.section_classifier import SectionClassificationResult
from utils.logger import log


def canonicalize_orientation(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    sections: SectionClassificationResult,
) -> bool:
    """If the scan is upside-down (sections.inserts_first), flip the volume
    in Z and X and update Phase 1 / Phase 2 outputs so subsequent phases
    see a canonical orientation. Modifies ``dv``, ``phantom``, ``sections``
    in place. Returns True if a flip was applied; False if the scan was
    already canonical.
    """
    if not sections.inserts_first:
        return False

    n_slices = dv.num_slices
    width = dv.width

    log.info(
        "Phase 2b: Upside-down phantom detected (inserts-first acquisition). "
        f"Flipping volume in Z (reversing {n_slices} slices) and "
        f"X (mirroring width={width}) to canonicalise."
    )

    # --- Volume + slice locations ---
    # np.ascontiguousarray so downstream slicing (e.g. cv2 calls that require
    # C-contiguous memory) doesn't trip on the non-contiguous flipped view.
    dv.volume = np.ascontiguousarray(dv.volume[::-1, :, ::-1])
    if dv.slice_locations is not None and len(dv.slice_locations) > 0:
        dv.slice_locations = np.ascontiguousarray(dv.slice_locations[::-1])

    # --- Phantom Phase 1 outputs ---
    # Z-flip: reverse the per-slice circle list so circles[i] still refers
    # to the i-th slice in the NEW orientation.
    phantom.circles = list(reversed(phantom.circles))
    # X-flip: mirror the per-slice and global centre x-coordinate. y is
    # untouched (no y-axis flip). Radius is invariant under either flip.
    x_max = float(width - 1)
    for circ in phantom.circles:
        circ.center_x = x_max - circ.center_x
    phantom.global_center_x = x_max - phantom.global_center_x

    # --- Section Phase 2 outputs ---
    # Z-flip: a section that occupied [start, end) in the old z-order now
    # occupies [n - end, n - start) in the new (reversed) z-order.
    # slice_count is invariant. Section name stays the same — section_classifier
    # already assigned names so they correspond to physical modules (e.g. the
    # INSERT2 section, which was at low z in inserts_first, is now at high z
    # = canonical INSERT2 position).
    for sec in sections.sections:
        new_start = n_slices - sec.end_idx
        new_end = n_slices - sec.start_idx
        sec.start_idx = new_start
        sec.end_idx = new_end
    sections.sections.sort(key=lambda s: s.start_idx)
    # Reverse the per-slice label list to match the new z-order.
    sections.slice_labels = list(reversed(sections.slice_labels))
    # The volume is now canonical; clear the flag so downstream code doesn't
    # think it still needs compensation.
    sections.inserts_first = False

    log.info(
        f"  Canonical layout after flip: "
        + " → ".join(f"{s.name}[{s.start_idx}:{s.end_idx}]"
                     for s in sections.sections
                     if s.name in ("STAR16", "STAR8", "INSERT1", "INSERT2"))
    )
    return True
