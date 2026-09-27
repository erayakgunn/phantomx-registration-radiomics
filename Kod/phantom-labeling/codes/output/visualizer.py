"""Overlay PNG generation for visual inspection."""

import cv2
import numpy as np
import re
from typing import Dict, List, Optional

from core.dicom_loader import DicomVolume
from core.phantom_detector import PhantomDetectionResult, SliceStatus
from core.section_classifier import SectionClassificationResult
from core.roi_extractor import ExtractionResult, ROI
from utils.logger import log
from utils.file_utils import ensure_dir
import config
import os


# Colors (BGR for OpenCV)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)


# Matches the size-suffix on insert sub-ROI labels (e.g. "Insert_01_5mm").
_INSERT_SIZE_RE = re.compile(r"^Insert_\d+_(\d+)mm$")


def _insert_subroi_size_mm(label: str) -> Optional[int]:
    """Return the diameter in mm if ``label`` is an insert sub-ROI, else None."""
    m = _INSERT_SIZE_RE.match(label)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def generate_overlays(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    sections: SectionClassificationResult,
    extraction: ExtractionResult,
    output_dir: str,
    series_name: str,
):
    """Generate overlay PNGs for all processed slices."""
    log.info("Phase 6: Generating overlay PNGs...")

    overlay_dir = ensure_dir(os.path.join(output_dir, series_name, "overlays"))
    count = 0

    # Stable per-insert nametag anchor. Each slice draws its insert circles at
    # that slice's OWN (Theil-Sen-smoothed) centre — correct, exactly centred —
    # but those centres still vary sub-pixel from slice to slice, and the
    # integer-rounded text origin then jitters ±1 px between consecutive
    # slices, so the "Insert_NN" nametag looks like it slides around even
    # though the circle is fine. Pin the TEXT to the per-section MEDIAN of that
    # insert's centre across all its slices so the nametag stays put while the
    # circle tracks its slice. (Star/texture labels already use a per-section
    # constant centre, so only inserts need this.)
    from collections import defaultdict as _dd
    _ins_centers = _dd(list)
    for _roi in extraction.rois:
        if _insert_subroi_size_mm(_roi.label) is not None:
            _num_label = _roi.label.rsplit("_", 1)[0]  # "Insert_NN"
            _ins_centers[_num_label].append((_roi.center_x, _roi.center_y))
    insert_text_anchor = {
        lbl: (float(np.median([c[0] for c in pts])),
              float(np.median([c[1] for c in pts])))
        for lbl, pts in _ins_centers.items()
    }

    # Render every slice that is at least transitional phantom (skip only the
    # empty scan ends labelled UNKNOWN/INVALID). The body / anthropomorphic
    # ("phantomX") slices and the inter-module TRANSITION slices ARE rendered
    # so the operator can see them — but they carry no ROIs, so no labels get
    # drawn on them (labeling happens only on the STAR/INSERT calibration
    # sections, which is where extraction.per_slice has entries).
    for i in range(dv.num_slices):
        label = sections.slice_labels[i]
        if label in ("UNKNOWN", "INVALID"):
            continue

        circ = phantom.circles[i]
        rois = extraction.per_slice.get(i, [])

        img = _hu_to_rgb_adaptive(dv.volume[i], circ)

        # Draw phantom boundary (anti-aliased + sub-pixel via shift=4 so the
        # green circle is centred at the actual float (cx, cy), not the int-
        # rounded pixel — eliminates the Bresenham staircase that otherwise
        # looks like spikes/bumps around the perimeter on small radii).
        if circ.status == SliceStatus.VALID:
            _SHIFT = 4
            _F = 1 << _SHIFT
            cv2.circle(
                img,
                (int(round(circ.center_x * _F)), int(round(circ.center_y * _F))),
                int(round(circ.radius * _F)),
                GREEN,
                2,
                cv2.LINE_AA,
                _SHIFT,
            )

        # Draw all sub-ROI circles per insert (5 mm yellow, 10 mm cyan,
        # 15 mm magenta — colours from INSERT_SUBROI_COLORS_BGR). The text
        # ``Insert_NN`` label is anchored to the outermost (largest)
        # circle only, so the overlay shows 60 nested circles but 20
        # labels rather than 60 stacked labels.
        largest_subroi_diam_mm = (
            max(config.INSERT_SUBROI_DIAMETERS_MM)
            if config.INSERT_SUBROI_DIAMETERS_MM else None
        )
        avg_spacing_mm = (dv.pixel_spacing[0] + dv.pixel_spacing[1]) / 2.0

        # Sub-pixel circle rendering parameters. shift=4 → 1/16-pixel
        # resolution, plenty for visually smooth circles at any radius.
        _SHIFT = 4
        _F = 1 << _SHIFT

        # Draw ROIs
        for roi in rois:
            cx_i = int(round(roi.center_x))
            cy_i = int(round(roi.center_y))

            size_mm = _insert_subroi_size_mm(roi.label)
            if size_mm is not None:
                color = config.INSERT_SUBROI_COLORS_BGR.get(size_mm, CYAN)
                # Only attach the text label to the outermost ring per
                # insert; the smaller sub-ROIs are drawn unlabelled to
                # keep the image readable.
                if size_mm == largest_subroi_diam_mm:
                    text_label = roi.label.rsplit("_", 1)[0]  # "Insert_NN"
                else:
                    text_label = None
            else:
                # STAR8 / STAR16 sections (Star + texture ROIs) use
                # magenta to visually separate them from the inserts
                # (which use the yellow/cyan/magenta sub-ROI palette).
                if roi.section in ("STAR8", "STAR16"):
                    color = MAGENTA if roi.accepted else RED
                else:
                    color = CYAN if roi.accepted else RED
                text_label = roi.label

            if roi.shape_type == "circle":
                # Sub-pixel radius — matches the actual ROI mask radius
                # (size_cm cm / 2 / mm_per_px), not the integer-ceiled
                # pixel array half-size. Avoids the half-pixel offset
                # between the drawn outline and the underlying ROI.
                radius_sub_px = (roi.size_cm * 10.0 / 2.0) / avg_spacing_mm
                cx_sub = int(round(roi.center_x * _F))
                cy_sub = int(round(roi.center_y * _F))
                r_sub = int(round(radius_sub_px * _F))
                if roi.accepted or size_mm is not None:
                    cv2.circle(
                        img, (cx_sub, cy_sub), r_sub, color, 1,
                        cv2.LINE_AA, _SHIFT,
                    )
                else:
                    _draw_dashed_circle(
                        img, roi.center_x, roi.center_y, radius_sub_px, color,
                    )
                # Outline radius rounded to integer for label anchoring.
                _outline_r_int = int(np.ceil(radius_sub_px))
            else:
                hh = roi.pixels.shape[0] // 2
                hw = roi.pixels.shape[1] // 2
                rot_deg = getattr(roi, "shape_rotation_deg", 0.0)
                if abs(rot_deg) > 0.3:
                    # Draw a ROTATED square so the outline visually tracks
                    # the texture frame (which is tilted with the phantom).
                    # cv2.boxPoints handles the four-corner geometry.
                    rect = (
                        (float(roi.center_x), float(roi.center_y)),
                        (float(2 * hw), float(2 * hh)),
                        float(rot_deg),
                    )
                    pts = cv2.boxPoints(rect)
                    pts_int = np.round(pts).astype(np.int32)
                    if roi.accepted:
                        cv2.polylines(
                            img, [pts_int], isClosed=True,
                            color=color, thickness=1, lineType=cv2.LINE_AA,
                        )
                    else:
                        # Dashed rotated outline: draw each side as 4 dashes
                        for k in range(4):
                            p0 = pts[k]
                            p1 = pts[(k + 1) % 4]
                            for t in (0.0, 0.5):
                                seg_a = p0 + (p1 - p0) * t
                                seg_b = p0 + (p1 - p0) * (t + 0.25)
                                cv2.line(
                                    img,
                                    (int(round(seg_a[0])), int(round(seg_a[1]))),
                                    (int(round(seg_b[0])), int(round(seg_b[1]))),
                                    color, 1, cv2.LINE_AA,
                                )
                    # For label anchoring use the bounding box of the rotated
                    # rectangle (~max corner distance from centre).
                    _outline_r_int = int(np.ceil(
                        max(abs(p[0] - roi.center_x) for p in pts)
                    ))
                else:
                    pt1 = (cx_i - hw, cy_i - hh)
                    pt2 = (cx_i + hw, cy_i + hh)
                    if roi.accepted:
                        cv2.rectangle(img, pt1, pt2, color, 1, cv2.LINE_AA)
                    else:
                        _draw_dashed_rect(img, pt1, pt2, color)
                    _outline_r_int = max(hh, hw)

            if text_label is not None:
                if size_mm is not None and roi.shape_type == "circle":
                    # Insert sub-ROI: keep the label just outside the outline
                    # so it never covers the small nested 5/10/15 mm circles.
                    # Anchor the TEXT to the stable per-insert median centre
                    # (not this slice's centre) so the nametag doesn't jitter
                    # between slices while the circle tracks its slice.
                    _ax, _ay = insert_text_anchor.get(
                        text_label, (roi.center_x, roi.center_y)
                    )
                    _axi, _ayi = int(round(_ax)), int(round(_ay))
                    text_anchor = (_axi + _outline_r_int + 2, _ayi - 2)
                else:
                    # Texture / star ROI: centre the label ON the ROI so it
                    # reads as centred on the square instead of trailing off
                    # to one side (cv2 text origin is the glyph bottom-left,
                    # so offset back by half the text box).
                    (_tw, _th), _ = cv2.getTextSize(
                        text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
                    )
                    text_anchor = (cx_i - _tw // 2, cy_i + _th // 2)
                cv2.putText(
                    img, text_label, text_anchor,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
                )

        # Info band at top
        accepted_count = sum(1 for r in rois if r.accepted)
        total_count = len(rois)
        info = f"Slice {i} | {label} | {accepted_count}/{total_count} ROIs"
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(
            img, info, (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA,
        )

        # Resize to 512x512
        img = cv2.resize(img, config.OVERLAY_SIZE, interpolation=cv2.INTER_AREA)

        out_path = os.path.join(overlay_dir, f"slice_{i:04d}.png")
        if not cv2.imwrite(out_path, img):
            log.warning(f"Failed to write overlay: {out_path}")
        else:
            count += 1

    log.info(f"Generated {count} overlay PNGs in {overlay_dir}")


def _hu_to_rgb_adaptive(hu_slice: np.ndarray, circ) -> np.ndarray:
    """Convert HU slice to RGB using adaptive window/level from phantom interior."""
    h, w = hu_slice.shape

    if circ.status == SliceStatus.VALID:
        # Sample pixels inside phantom to compute adaptive WW/WL
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - circ.center_x) ** 2 + (yy - circ.center_y) ** 2)
        mask = dist < circ.radius * 0.95
        interior = hu_slice[mask]

        if interior.size > 0:
            p2, p98 = np.percentile(interior, [2, 98])
            margin = (p98 - p2) * 0.1
            low = p2 - margin
            high = p98 + margin
        else:
            low, high = -200, 200
    else:
        low, high = -200, 200

    if high <= low:
        high = low + 1

    gray = np.clip((hu_slice - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _draw_dashed_circle(img, cx, cy, radius, color, dash_len=8):
    """Draw an anti-aliased dashed circle with sub-pixel endpoint precision.

    ``cx``, ``cy``, ``radius`` may be floats; the dash endpoints are computed
    at full precision and rendered with cv2.LINE_AA so each dash segment is
    smooth instead of staircase-stepped.
    """
    _SHIFT = 4
    _F = 1 << _SHIFT
    circumference = max(1, int(2 * np.pi * radius))
    for i in range(0, circumference, dash_len * 2):
        a1 = 2 * np.pi * i / circumference
        a2 = 2 * np.pi * min(i + dash_len, circumference) / circumference
        pt1 = (
            int(round((cx + radius * np.cos(a1)) * _F)),
            int(round((cy + radius * np.sin(a1)) * _F)),
        )
        pt2 = (
            int(round((cx + radius * np.cos(a2)) * _F)),
            int(round((cy + radius * np.sin(a2)) * _F)),
        )
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA, _SHIFT)


def _draw_dashed_rect(img, pt1, pt2, color, dash_len=6):
    """Draw a dashed rectangle (anti-aliased segments)."""
    x1, y1 = pt1
    x2, y2 = pt2
    for start, end in [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]:
        _draw_dashed_line(img, start, end, color, dash_len)


def _draw_dashed_line(img, pt1, pt2, color, dash_len=6):
    """Draw an anti-aliased dashed line with sub-pixel endpoints."""
    _SHIFT = 4
    _F = 1 << _SHIFT
    dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    if dist < 1:
        return
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    d = 0.0
    drawing = True
    while d < dist:
        d_end = min(d + dash_len, dist)
        if drawing:
            p1 = (
                int(round((pt1[0] + dx * d) * _F)),
                int(round((pt1[1] + dy * d) * _F)),
            )
            p2 = (
                int(round((pt1[0] + dx * d_end) * _F)),
                int(round((pt1[1] + dy * d_end) * _F)),
            )
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA, _SHIFT)
        d = d_end
        drawing = not drawing
