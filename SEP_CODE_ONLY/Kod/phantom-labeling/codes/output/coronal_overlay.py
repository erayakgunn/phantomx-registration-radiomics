"""Coronal-view overlay showing per-section z-extents and the inner-window trim.

The pipeline runs detection and ROI extraction per axial slice but the
section boundaries are a property of the z-axis. ``section_trim`` decides
which slices a section keeps based on physical z (mm), so to sanity-check
that decision visually we want to render the SAME information in coronal
view: the (z, x) plane sliced through the phantom centre.

Output: ``<output_dir>/<series_name>/coronal/coronal.png``

Layout (top-to-bottom mirrors slice index 0..N-1):
  * grayscale background = ``volume[:, y_center, :]`` rescaled to a CT-
    soft-tissue window, then resampled so 1 px on screen = 1 mm in both
    axes (i.e. anisotropic resampling along z by ``slice_thickness /
    pixel_spacing[1]``).
  * For each detected functional section:
      - thin dashed line at the section's detected start/end indices
        (i.e. what ``section_classifier`` produced before the trim)
      - solid filled band over the kept (non-TRANSITION) slices = what
        the rest of the pipeline actually operates on
      - section name printed on the left margin

If the kept band covers a tight centred slab inside the dashed extent,
the inner-window trim is doing the right thing.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from core.dicom_loader import DicomVolume
from core.phantom_detector import PhantomDetectionResult
from core.section_classifier import SectionClassificationResult, SectionRange
from utils.file_utils import ensure_dir
from utils.logger import log
import config


# BGR colours per section, chosen for high contrast against grayscale.
_SECTION_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "STAR16":  (0, 255, 255),    # yellow
    "STAR8":   (0, 255, 0),      # green
    "INSERT1": (255, 0, 255),    # magenta
    "INSERT2": (255, 255, 0),    # cyan
}


def generate_coronal_overlay(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    sections: SectionClassificationResult,
    output_dir: str,
    series_name: str,
) -> None:
    """Render the coronal view PNG showing section z-extents + kept windows."""
    if dv.num_slices < 2:
        log.warning("Coronal overlay: too few slices, skipping.")
        return

    out_dir = ensure_dir(os.path.join(output_dir, series_name, "coronal"))
    out_path = os.path.join(out_dir, "coronal.png")

    # Slice the volume coronally through the phantom centre y. Falls back
    # to image height/2 if the global centre is unavailable.
    h, w = dv.volume.shape[1], dv.volume.shape[2]
    if (phantom is not None
            and getattr(phantom, "global_center_y", None) is not None):
        y_center = int(round(float(phantom.global_center_y)))
    else:
        y_center = h // 2
    y_center = max(0, min(h - 1, y_center))
    coronal = dv.volume[:, y_center, :].astype(np.float32)  # (N_slices, W)

    # Soft-tissue style WW/WL — keeps inserts and stars all visible.
    low_hu, high_hu = -400.0, 400.0
    gray = np.clip(
        (coronal - low_hu) / (high_hu - low_hu) * 255.0, 0, 255
    ).astype(np.uint8)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Resample so on-screen pixels are isotropic (1 px ≈ 1 mm in both
    # axes). x spacing = ``pixel_spacing[1]`` mm/col. For the z spacing
    # we use the actual ``slice_locations`` interval (= reconstruction
    # spacing), NOT ``slice_thickness`` (which is beam thickness and
    # may not match the actual sample spacing for overlapping recon).
    x_mm_per_px = float(dv.pixel_spacing[1])
    # True physical-z extent = N_unique_z × reconstruction spacing.
    # Multi-slice-per-z acquisitions (Siemens PCCT dual-detector /
    # dual-energy) have raw num_slices = 2 × N_unique_z, so the naive
    # `num_slices × dz` overstates the vertical extent by 2× and gives
    # a distorted aspect-ratio coronal view.
    z_per_raw_slice = None
    if dv.slice_locations is not None and len(dv.slice_locations) > 1:
        z_arr_full = np.asarray(dv.slice_locations, dtype=np.float64)
        unique_z = np.unique(np.round(z_arr_full, 4))
        if unique_z.size > 1:
            diffs = np.abs(np.diff(np.sort(unique_z)))
            pos_diffs = diffs[diffs > 1e-6]
            if pos_diffs.size > 0:
                dz_unique = float(np.median(pos_diffs))
                total_phys_mm = float(unique_z.size) * dz_unique
                z_per_raw_slice = total_phys_mm / float(dv.num_slices)
    if z_per_raw_slice is None or z_per_raw_slice <= 0:
        z_per_raw_slice = float(dv.slice_thickness) if dv.slice_thickness else 1.0
    z_mm_per_px = z_per_raw_slice  # kept for the y_of math below
    if x_mm_per_px > 0 and z_mm_per_px > 0:
        target_h = max(2, int(round(dv.num_slices * z_mm_per_px / x_mm_per_px)))
        if target_h != dv.num_slices:
            img = cv2.resize(img, (w, target_h), interpolation=cv2.INTER_LINEAR)
    rendered_h = img.shape[0]

    def y_of(slice_idx: float) -> int:
        """Map slice index (possibly fractional) to row in the rendered image."""
        scaled = slice_idx / max(1, dv.num_slices - 1) * (rendered_h - 1)
        return int(round(scaled))

    # Phantom horizontal extent in the coronal slab. Painted bands clamp
    # to this so the coloured area on-screen equals the area used in the
    # ``A = z × W`` calculation (no air on either side).
    if (phantom is not None
            and getattr(phantom, "global_radius", None) is not None
            and phantom.global_radius > 0):
        cx_px = float(phantom.global_center_x)
        r_px = float(phantom.global_radius)
        phantom_x_lo_px = int(round(max(0.0, cx_px - r_px)))
        phantom_x_hi_px = int(round(min(w - 1, cx_px + r_px)))
        phantom_width_mm = 2.0 * r_px * x_mm_per_px
    else:
        phantom_x_lo_px = 0
        phantom_x_hi_px = w - 1
        phantom_width_mm = w * x_mm_per_px

    z_array = np.asarray(dv.slice_locations, dtype=np.float64)

    # Draw a colour band per section.
    for sec in sections.sections:
        if sec.name in ("UNKNOWN", "INVALID", "TRANSITION"):
            continue
        color = _SECTION_COLORS_BGR.get(sec.name)
        if color is None:
            continue

        # Detected extent — what ``section_classifier`` originally found,
        # BEFORE ``section_trim`` may have expanded/shrunk the live
        # indices. ``detected_*`` falls back to live indices for sections
        # that were never trimmed.
        det_start = sec.detected_start_idx if sec.detected_start_idx is not None else sec.start_idx
        det_end = sec.detected_end_idx if sec.detected_end_idx is not None else sec.end_idx
        y_det_lo = y_of(det_start)
        y_det_hi = y_of(max(det_start, det_end - 1))
        _draw_dashed_hline(
            img, y_det_lo, color,
            x_start=phantom_x_lo_px, x_end=phantom_x_hi_px,
        )
        _draw_dashed_hline(
            img, y_det_hi, color,
            x_start=phantom_x_lo_px, x_end=phantom_x_hi_px,
        )

        # Kept window — actual non-TRANSITION slices after trim. Drawn
        # as a translucent solid band over those rows.
        kept_idx = [
            i for i in range(sec.start_idx, sec.end_idx)
            if sections.slice_labels[i] != "TRANSITION"
        ]
        if kept_idx:
            y_kept_lo = y_of(min(kept_idx))
            y_kept_hi = y_of(max(kept_idx))
            overlay = img.copy()
            # Clamp horizontal extent to the phantom — no air on either side.
            cv2.rectangle(
                overlay,
                (phantom_x_lo_px, y_kept_lo),
                (phantom_x_hi_px, y_kept_hi),
                color, -1,
            )
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)

            # Total physical extent of the kept slices: count UNIQUE
            # z positions × reconstruction spacing. On multi-slice-per-z
            # acquisitions (some Siemens PCCT scans produce two slices
            # at every z — different detector layers or energy bins),
            # raw len(kept_idx) double-counts the physical extent.
            # Using N_unique × dz_unique gives the true mm covered.
            kept_z_unique = np.unique(np.round(z_array[kept_idx], 4))
            actual_dz_disp = None
            if len(z_array) > 1:
                # Spacing of UNIQUE z positions across the whole volume
                # (not just kept_idx — gives a more stable estimate).
                unique_z_all = np.unique(np.round(z_array, 4))
                if unique_z_all.size > 1:
                    diffs_disp = np.abs(np.diff(np.sort(unique_z_all)))
                    pos_diffs_disp = diffs_disp[diffs_disp > 1e-6]
                    if pos_diffs_disp.size > 0:
                        actual_dz_disp = float(np.median(pos_diffs_disp))
            if actual_dz_disp is None or actual_dz_disp <= 0:
                actual_dz_disp = float(dv.slice_thickness) if dv.slice_thickness else 1.0
            z_thickness_mm = float(len(kept_z_unique) * actual_dz_disp)

            # Vertical centre of the kept band — only used as the anchor
            # for the label box (no midpoint marker line drawn).
            mid = (y_kept_lo + y_kept_hi) // 2
        else:
            mid = (y_det_lo + y_det_hi) // 2
            z_thickness_mm = 0.0

        # Total detected length of the section — the "coronal slab"
        # length the user needs to know before picking ±N. Computed from
        # the ORIGINAL section_classifier bounds (``detected_*``), not
        # the post-trim live bounds, so the figure tells you the true
        # physical extent of the section regardless of how aggressively
        # the trim shrank/expanded the kept window. ``abs(...)`` because
        # scans acquired feet-first have decreasing slice_locations and
        # would otherwise display a negative length.
        if (0 <= det_start < len(z_array)
                and 0 < det_end <= len(z_array)):
            total_section_len_mm = float(
                abs(z_array[det_end - 1] - z_array[det_start])
            )
        else:
            total_section_len_mm = 0.0

        # Three-line label inside the painted band:
        #   line 1: section name
        #   line 2: total detected coronal slab length (L)
        #   line 3: kept slice count and physical span in mm
        n_kept_slices = len(kept_idx) if kept_idx else 0
        name_text = sec.name
        length_text = f"L = {total_section_len_mm:.1f} mm"
        slice_text = (
            f"N = {n_kept_slices} slices "
            f"({z_thickness_mm:.1f} mm)"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        name_scale = 0.5
        line_scale = 0.42
        (n_w, n_h), _ = cv2.getTextSize(name_text, font, name_scale, 1)
        (l_w, l_h), _ = cv2.getTextSize(length_text, font, line_scale, 1)
        (s_w, s_h), _ = cv2.getTextSize(slice_text, font, line_scale, 1)
        box_w = max(n_w, l_w, s_w) + 8
        box_h = n_h + l_h + s_h + 14  # 3 lines + padding

        band_cx = (phantom_x_lo_px + phantom_x_hi_px) // 2
        box_left = band_cx - box_w // 2
        if box_w <= (phantom_x_hi_px - phantom_x_lo_px):
            box_left = max(phantom_x_lo_px,
                           min(box_left, phantom_x_hi_px - box_w))
        box_left = max(0, min(box_left, w - box_w))
        box_top = mid - box_h // 2
        box_top = max(22, min(box_top, rendered_h - box_h))
        box_right = box_left + box_w
        box_bot = box_top + box_h

        cv2.rectangle(img, (box_left, box_top), (box_right, box_bot),
                      (0, 0, 0), -1)
        cv2.putText(img, name_text,
                    (box_left + 4, box_top + n_h + 1),
                    font, name_scale, color, 1, cv2.LINE_AA)
        cv2.putText(img, length_text,
                    (box_left + 4, box_top + n_h + l_h + 5),
                    font, line_scale, color, 1, cv2.LINE_AA)
        cv2.putText(img, slice_text,
                    (box_left + 4, box_top + n_h + l_h + s_h + 10),
                    font, line_scale, color, 1, cv2.LINE_AA)

    # Header: brief legend + scan summary.
    legend = (
        f"Coronal | y={y_center} | dashed=detected extent  "
        f"filled=kept window  |  z-scale {z_mm_per_px:.2f} mm/slice"
    )
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, legend, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA)

    if not cv2.imwrite(out_path, img):
        log.warning(f"Failed to write coronal overlay: {out_path}")
        return
    log.info(
        f"Coronal overlay saved to {out_path} "
        f"(rendered {img.shape[1]}x{img.shape[0]}, "
        f"coronal slab at y={y_center})"
    )


def _draw_solid_hline(img: np.ndarray, y: int, color: Tuple[int, int, int],
                      thickness: int = 1,
                      x_start: Optional[int] = None,
                      x_end: Optional[int] = None) -> None:
    """Draw an anti-aliased solid horizontal line at row ``y``, clamped
    to ``[x_start, x_end]`` (defaults to the full image width). Used to
    mark the kept-window midpoint inside each coronal-overlay band.
    """
    h, w, _ = img.shape
    if not (0 <= y < h):
        return
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = w - 1
    x_start = max(0, min(w - 1, int(x_start)))
    x_end = max(0, min(w - 1, int(x_end)))
    if x_end <= x_start:
        return
    _SHIFT = 4
    _F = 1 << _SHIFT
    y_sub = int(round(y * _F))
    cv2.line(
        img,
        (int(round(x_start * _F)), y_sub),
        (int(round(x_end * _F)), y_sub),
        color, thickness, cv2.LINE_AA, _SHIFT,
    )


def _draw_dashed_hline(img: np.ndarray, y: int, color: Tuple[int, int, int],
                       dash_len: int = 8,
                       x_start: Optional[int] = None,
                       x_end: Optional[int] = None) -> None:
    """Draw an anti-aliased dashed horizontal line at row ``y``.

    ``x_start`` / ``x_end`` (inclusive) clip the dashed line to a column
    range — used to keep the section boundary dashes inside the phantom
    width on coronal overlays.
    """
    h, w, _ = img.shape
    if not (0 <= y < h):
        return
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = w - 1
    x_start = max(0, min(w - 1, int(x_start)))
    x_end = max(0, min(w - 1, int(x_end)))
    if x_end <= x_start:
        return
    _SHIFT = 4
    _F = 1 << _SHIFT
    y_sub = int(round(y * _F))
    for seg_start in range(x_start, x_end, dash_len * 2):
        seg_end = min(seg_start + dash_len, x_end)
        cv2.line(
            img,
            (int(round(seg_start * _F)), y_sub),
            (int(round(seg_end * _F)), y_sub),
            color, 1, cv2.LINE_AA, _SHIFT,
        )
