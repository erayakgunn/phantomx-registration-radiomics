"""Painted-ROI overlay generation for visual centering verification.

The standard overlay (``output.visualizer.generate_overlays``) draws a
1-pixel circle around each insert centre — useful for checking that the
detected centre falls on the insert, but it leaves the actual ROI area
empty so you can't see whether the ROI's footprint matches the insert's
footprint pixel-for-pixel.

This module fills each insert ROI with a semi-transparent colour overlay
using the SAME boolean mask that ``_extract_nearest_neighbor_circle``
uses when it builds the .npy export — i.e. what you see painted on these
PNGs is exactly the set of pixels that gets written to disk for that ROI.
Any mismatch between the painted area and the underlying bright insert
disk is a real detection error, not a rendering artefact.

Output: ``<output_dir>/<series_name>/painted/slice_NNNN.png``, one PNG
per slice that carries any insert ROIs.
"""

import os
from typing import List

import cv2
import numpy as np

from core.dicom_loader import DicomVolume
from core.phantom_detector import PhantomDetectionResult, SliceStatus
from core.section_classifier import SectionClassificationResult
from core.roi_extractor import ExtractionResult, ROI
from output.visualizer import _hu_to_rgb_adaptive, _insert_subroi_size_mm
from utils.file_utils import ensure_dir
from utils.logger import log
import config


# BGR fill colour and opacity for the painted ROI. Magenta matches the
# 15 mm sub-ROI colour used by the standard overlay so the two views are
# easy to correlate.
_PAINT_COLOR_BGR = (255, 0, 255)
_PAINT_ALPHA = 0.45
# Outline drawn on top of the fill in solid colour at the same radius —
# emphasises the boundary so under-coverage at the rim is visible.
_OUTLINE_COLOR_BGR = (255, 0, 255)
# White centre marker — 1 px dot at the refined (sub-pixel) centre. Use
# a different colour from the fill so the centre is unambiguous even
# when the fill saturates.
_CENTER_DOT_COLOR_BGR = (255, 255, 255)


def paint_insert_rois(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    sections: SectionClassificationResult,
    extraction: ExtractionResult,
    output_dir: str,
    series_name: str,
) -> None:
    """Render every insert slice with the largest sub-ROI painted as a
    filled semi-transparent disk on top of the CT grayscale.

    Only the largest sub-ROI per insert (typically 15 mm) is painted —
    that diameter equals the physical insert size, so its painted disk
    should exactly cover the bright insert disk in the slice. Mismatch
    = centring error, off radius, or wrong slot mapping.
    """
    log.info("Phase 6b: Painting detected insert ROIs...")

    paint_dir = ensure_dir(os.path.join(output_dir, series_name, "painted"))
    avg_spacing_mm = (dv.pixel_spacing[0] + dv.pixel_spacing[1]) / 2.0

    if not config.INSERT_SUBROI_DIAMETERS_MM:
        log.warning("  No INSERT_SUBROI_DIAMETERS_MM configured; skipping paint.")
        return
    largest_diam_mm = max(config.INSERT_SUBROI_DIAMETERS_MM)
    radius_px = (largest_diam_mm / 2.0) / avg_spacing_mm
    radius_px_sq = radius_px * radius_px
    outline_radius = int(np.ceil(radius_px))

    # Stable per-insert nametag anchor (per-section median of the insert's
    # centre across its slices). The circle + white centre dot are still drawn
    # at each slice's own centre — that's the whole point of this verification
    # view — but the TEXT is pinned so the "Insert_NN" nametag doesn't appear
    # to slide between slices. See the matching note in visualizer.py.
    from collections import defaultdict as _dd
    _ins_centers = _dd(list)
    for _roi in extraction.rois:
        if _insert_subroi_size_mm(_roi.label) == largest_diam_mm:
            _ins_centers[_roi.label.rsplit("_", 1)[0]].append(
                (_roi.center_x, _roi.center_y)
            )
    insert_text_anchor = {
        lbl: (int(round(np.median([c[0] for c in pts]))),
              int(round(np.median([c[1] for c in pts]))))
        for lbl, pts in _ins_centers.items()
    }

    count = 0
    for i in range(dv.num_slices):
        slice_label = sections.slice_labels[i]
        if slice_label in ("UNKNOWN", "INVALID", "TRANSITION"):
            continue

        rois: List[ROI] = extraction.per_slice.get(i, [])
        insert_rois = [
            r for r in rois
            if _insert_subroi_size_mm(r.label) == largest_diam_mm
        ]
        if not insert_rois:
            continue

        circ = phantom.circles[i]
        img = _hu_to_rgb_adaptive(dv.volume[i], circ)
        h, w = dv.volume[i].shape

        # Build the union mask of all insert ROIs on this slice using
        # the same `dx² + dy² ≤ r²` test that _extract_nearest_neighbor_circle
        # uses to gate which pixels go into the .npy export. The painted
        # area is therefore exactly the saved-ROI footprint.
        yy, xx = np.ogrid[:h, :w]
        paint_mask = np.zeros((h, w), dtype=bool)
        for roi in insert_rois:
            dx = xx - roi.center_x
            dy = yy - roi.center_y
            paint_mask |= (dx * dx + dy * dy) <= radius_px_sq

        # Alpha-blend the fill colour onto the painted pixels.
        if paint_mask.any():
            fill_layer = img.copy()
            fill_layer[paint_mask] = _PAINT_COLOR_BGR
            cv2.addWeighted(fill_layer, _PAINT_ALPHA, img, 1.0 - _PAINT_ALPHA, 0, dst=img)

        # Draw the boundary outline (1 px, anti-aliased + sub-pixel shift)
        # so it lands at the exact float (cx, cy) of the refined centre, not
        # the int-rounded pixel. Combined with cv2.LINE_AA this produces a
        # smooth circle without the Bresenham staircase artefact.
        _SHIFT = 4
        _F = 1 << _SHIFT
        radius_shifted = int(round(radius_px * _F))
        for roi in insert_rois:
            cx_sub = int(round(roi.center_x * _F))
            cy_sub = int(round(roi.center_y * _F))
            cv2.circle(
                img, (cx_sub, cy_sub), radius_shifted,
                _OUTLINE_COLOR_BGR, 1, cv2.LINE_AA, _SHIFT,
            )
            # 1 px white dot at the sub-pixel centre (rounded to nearest pixel).
            cx_i = int(round(roi.center_x))
            cy_i = int(round(roi.center_y))
            if 0 <= cx_i < w and 0 <= cy_i < h:
                img[cy_i, cx_i] = _CENTER_DOT_COLOR_BGR

        # Insert label next to each circle — anchored to the STABLE per-insert
        # median centre (not this slice's centre) so the nametag stays put.
        for roi in insert_rois:
            base_label = roi.label.rsplit("_", 1)[0]  # strip "_15mm"
            ax, ay = insert_text_anchor.get(
                base_label, (int(round(roi.center_x)), int(round(roi.center_y)))
            )
            cv2.putText(
                img,
                base_label,
                (ax + outline_radius + 2, ay - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        out_path = os.path.join(paint_dir, f"slice_{i:04d}.png")
        cv2.imwrite(out_path, img)
        count += 1

    log.info(f"  Painted {count} slices to {paint_dir}")
