"""Generate before/after registration overlays for the SEP phantom UI.

The "before" state is a centroid-translation baseline (stable bone mask).
The "after" state is the production rigid transform used for the workbook.
Only the fixed CT is used as background; fixed and moving organ contours are
drawn on top.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
LAB = Path(os.environ.get("SEP_LAB", str(ROOT / "PhantomX" / "LABELED")))
BATCH = Path(os.environ.get(
    "SEP_BATCH",
    str(ROOT / "PhantomX" / "REGISTRATION" / "batch_to_20260805_v1.0.0"),
))
OUT = Path(__file__).resolve().parent
ASSETS = OUT / "assets"
CODE_DIR = ROOT / "PCCT" / "Kod" / "registration-radiomics"
if not (CODE_DIR / "registration.py").exists():
    CODE_DIR = ROOT / "Kod" / "registration-radiomics"
if not (CODE_DIR / "registration.py").exists():
    raise ModuleNotFoundError(f"registration.py bulunamadı: {CODE_DIR}")
sys.path.insert(0, str(CODE_DIR))

from registration import translation_centroid_output_transform  # noqa: E402

FIXED_DAY = os.environ.get("SEP_FIXED_DAY", "20260805")
FIXED_BASE = os.environ.get(
    "SEP_FIXED_BASE",
    "USZ_SIEMENS_SOMATOM_Edge_Plus_20260805_S3_Br40f_120_100_300_1_"
    "PolyE_take1_recon1mm",
)
FIXED_LABEL = os.environ.get("SEP_FIXED_LABEL", f"{FIXED_DAY}_t1")
SESSION_SUFFIX = os.environ.get("SEP_SESSION_SUFFIX", "SEP_Aorta_Spleen")
LABELS = [*range(27, 44), *range(92, 116), 116]
ORGANS = {52: "aorta", 1: "spleen"}
ORGAN_LABELS = {52: "Aorta", 1: "Spleen"}

# Display window in CT Hounsfield units.
WINDOW_LEVEL = 50.0
WINDOW_WIDTH = 500.0

CASES = [
    {
        "id": "high",
        "day": "20260730",
        "suffix": "_take1_recon1mm",
        "title": "Çok iyi iyileşen",
        "series": "20260730_t1",
        "tone": "mint",
        "tag": "En yüksek kazanç",
        "note": (
            "Aorta +0.072, Spleen +0.149. Belirgin rigid düzeltmesi; "
            "sonuç iki organda da yüksek."
        ),
    },
    {
        "id": "medium",
        "day": "20260128",
        "suffix": "_take1_recon1mm",
        "title": "İyi iyileşen",
        "series": "20260128_t1",
        "tone": "sky",
        "tag": "Dengeli kazanç",
        "note": (
            "Aorta +0.020, Spleen +0.036. 0128 force rigid sonrası "
            "Aorta 0.923, Spleen 0.965."
        ),
    },
    {
        "id": "good",
        "day": "20260327",
        "suffix": "_take1_recon1mm",
        "title": "Yüksek final",
        "series": "20260327_t1",
        "tone": "violet",
        "tag": "Final 0.932",
        "note": (
            "Aorta +0.054, Spleen +0.041. Final Dice yüksek; "
            "rigid katkısı net görülüyor."
        ),
    },
    {
        "id": "low",
        "day": "20260731",
        "suffix": "_s40",
        "title": "Az iyileşen / düşük",
        "series": "20260731_t10",
        "tone": "amber",
        "tag": "Sınırlı kazanç",
        "note": (
            "Aorta +0.012, Spleen +0.020. 0731 boundary/shape outlier; "
            "centroid zaten <1 mm, rigid tavanına yakın."
        ),
    },
]


def locate_batch(day: str, suffix: str) -> Path:
    for path in BATCH.iterdir():
        if path.is_dir() and path.name.startswith(day) and path.name.endswith(suffix):
            return path
    raise FileNotFoundError(f"batch dir not found: {day} {suffix}")


def native_paths(day: str, suffix: str) -> tuple[Path, Path, str]:
    batch = locate_batch(day, suffix)
    base = batch.name.replace(f"{day}_{SESSION_SUFFIX}__", "")
    day_dir = LAB / f"{day}_{SESSION_SUFFIX}"
    return (
        day_dir / "nifti" / base / "ct.nii.gz",
        day_dir / "segmentation" / base / "seg.nii.gz",
        base,
    )


def stable_union(seg: sitk.Image) -> sitk.Image:
    array = sitk.GetArrayFromImage(seg)
    mask = np.isin(array, LABELS)
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(seg)
    return image


def dice(reference: np.ndarray, candidate: np.ndarray, label: int) -> float:
    a = reference == label
    b = candidate == label
    denom = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 0.0


def crop_bounds(
    fixed: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    organ: int,
    margin: int,
) -> tuple[int, int, int, int]:
    """Return y0,y1,x0,x1 for an axial crop around all three masks."""
    z, y, x = np.where(
        (fixed == organ) | (before == organ) | (after == organ)
    )
    if not len(y):
        raise ValueError(f"empty mask for organ {organ}")
    y0, y1 = max(0, int(y.min()) - margin), min(fixed.shape[1], int(y.max()) + margin + 1)
    x0, x1 = max(0, int(x.min()) - margin), min(fixed.shape[2], int(x.max()) + margin + 1)
    return int(y0), int(y1), int(x0), int(x1)


def coronal_bounds(
    fixed: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    organ: int,
    margin: int,
) -> tuple[int, int, int, int, int]:
    """Return z0,z1,x0,x1 and the coronal y slice through the organ."""
    z, y, x = np.where(
        (fixed == organ) | (before == organ) | (after == organ)
    )
    if not len(z):
        raise ValueError(f"empty mask for organ {organ}")
    z0, z1 = max(0, int(z.min()) - margin), min(fixed.shape[0], int(z.max()) + margin + 1)
    x0, x1 = max(0, int(x.min()) - margin), min(fixed.shape[2], int(x.max()) + margin + 1)
    fixed_z, fixed_y, _ = np.where(fixed == organ)
    y_slice = int(round(float(np.median(fixed_y)))) if len(fixed_y) else int(round(float(np.median(y))))
    y_slice = max(0, min(fixed.shape[1] - 1, y_slice))
    return int(z0), int(z1), int(x0), int(x1), y_slice


def contour_points(mask: np.ndarray, x0: int, y0: int, scale_x: float, scale_y: float):
    contours = []
    if not mask.any():
        return contours
    for contour in measure.find_contours(mask.astype(float), 0.5):
        # find_contours returns (row, col) = (axis-0, axis-1)
        pts = [
            ((col - y0) * scale_x, (row - x0) * scale_y)
            for row, col in contour
        ]
        contours.append(pts)
    return contours


def render(
    ct: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    out_path: Path,
    *,
    scale: float = 2.0,
    size: int = 720,
    quality: int = 90,
) -> None:
    low = WINDOW_LEVEL - WINDOW_WIDTH / 2
    high = WINDOW_LEVEL + WINDOW_WIDTH / 2
    gray = np.clip((ct.astype(np.float32) - low) / (high - low), 0, 1)
    base = np.repeat(gray[:, :, None], 3, axis=2)

    f = fixed_mask.astype(bool)
    m = moving_mask.astype(bool)
    out = base.copy()
    # Filled translucent masks: green fixed, coral moving.
    fg = np.array([0.12, 0.94, 0.58], dtype=np.float32)
    mg = np.array([1.0, 0.28, 0.28], dtype=np.float32)
    out = out * (1 - 0.13 * f[:, :, None]) + fg * (0.13 * f[:, :, None])
    out = out * (1 - 0.16 * m[:, :, None]) + mg * (0.16 * m[:, :, None])
    rgb = np.clip(out * 255, 0, 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    image = Image.fromarray(rgb).resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image)

    # Fixed contour (green), then moving contour (coral).
    for pts in contour_points(fixed_mask, 0, 0, scale, scale):
        if len(pts) > 1:
            draw.line(pts, fill=(50, 255, 150), width=max(2, int(2.4 * scale)), joint="curve")
    for pts in contour_points(moving_mask, 0, 0, scale, scale):
        if len(pts) > 1:
            draw.line(pts, fill=(255, 82, 82), width=max(2, int(2.4 * scale)), joint="curve")

    image = image.resize((size, size), Image.Resampling.LANCZOS)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(
            out_path, format="JPEG", quality=quality, optimize=True, progressive=True
        )
    else:
        image.save(out_path, optimize=True)


def write_case(case: dict) -> dict:
    fixed_ct_path = LAB / f"{FIXED_DAY}_{SESSION_SUFFIX}" / "nifti" / FIXED_BASE / "ct.nii.gz"
    fixed_seg_path = LAB / f"{FIXED_DAY}_{SESSION_SUFFIX}" / "segmentation" / FIXED_BASE / "seg.nii.gz"
    fixed_ct = sitk.ReadImage(str(fixed_ct_path))
    fixed_seg = sitk.ReadImage(str(fixed_seg_path))
    fixed_array = sitk.GetArrayFromImage(fixed_seg)
    fixed_ct_array = sitk.GetArrayFromImage(fixed_ct)

    moving_ct_path, moving_seg_path, _ = native_paths(case["day"], case["suffix"])
    moving_seg = sitk.ReadImage(str(moving_seg_path))
    _ = moving_ct_path  # CT not needed for these overlays

    before_transform = translation_centroid_output_transform(
        stable_union(fixed_seg), stable_union(moving_seg)
    )
    after_transform = sitk.ReadTransform(
        str(locate_batch(case["day"], case["suffix"]) / "moving_to_fixed.tfm")
    )

    before_seg = sitk.Resample(
        moving_seg, fixed_seg, before_transform,
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
    )
    after_seg = sitk.Resample(
        moving_seg, fixed_seg, after_transform,
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
    )
    before_array = sitk.GetArrayFromImage(before_seg)
    after_array = sitk.GetArrayFromImage(after_seg)

    case_out = ASSETS / case["id"]
    case_out.mkdir(parents=True, exist_ok=True)

    payload = {
        "id": case["id"],
        "title": case["title"],
        "series": case["series"],
        "fixed": FIXED_LABEL,
        "tone": case["tone"],
        "tag": case["tag"],
        "note": case["note"],
        "summary": f"{case['series']} → {FIXED_LABEL}",
        "method": "Centroid translation → Rigid (bone/landmark)",
        "scores": {},
        "views": {},
    }

    for organ, organ_name in ORGANS.items():
        y0, y1, x0, x1 = crop_bounds(
            fixed_array, before_array, after_array, organ,
            margin=95 if organ == 52 else 55,
        )
        z0, z1, cx0, cx1, y_slice = coronal_bounds(
            fixed_array, before_array, after_array, organ,
            margin=55 if organ == 52 else 38,
        )

        z_indices = np.where((fixed_array == organ).any(axis=(1, 2)))[0]
        picks = np.percentile(z_indices, [20, 50, 80])
        picks = [int(round(v)) for v in picks]
        picks = sorted(set(max(int(z_indices.min()), min(int(z_indices.max()), p)) for p in picks))
        if len(picks) < 3:
            picks = sorted(set(int(v) for v in np.linspace(z_indices.min(), z_indices.max(), 3)))

        axial = []
        for index, z in enumerate(picks):
            before_path = case_out / f"{organ_name}_axial_{index}_before.png"
            after_path = case_out / f"{organ_name}_axial_{index}_after.png"
            ct = fixed_ct_array[z, y0:y1, x0:x1]
            render(ct, fixed_array[z, y0:y1, x0:x1] == organ,
                   before_array[z, y0:y1, x0:x1] == organ, before_path)
            render(ct, fixed_array[z, y0:y1, x0:x1] == organ,
                   after_array[z, y0:y1, x0:x1] == organ, after_path)
            axial.append({
                "label": f"z {z}",
                "before": f"assets/{case['id']}/{before_path.name}",
                "after": f"assets/{case['id']}/{after_path.name}",
            })

        all_slices = []
        for z in z_indices:
            z = int(z)
            all_before_path = case_out / f"{organ_name}_all_{z}_before.jpg"
            all_after_path = case_out / f"{organ_name}_all_{z}_after.jpg"
            ct = fixed_ct_array[z, y0:y1, x0:x1]
            fixed_slice = fixed_array[z, y0:y1, x0:x1] == organ
            render(ct, fixed_slice, before_array[z, y0:y1, x0:x1] == organ,
                   all_before_path, size=560, quality=84)
            render(ct, fixed_slice, after_array[z, y0:y1, x0:x1] == organ,
                   all_after_path, size=560, quality=84)
            all_slices.append({
                "label": f"z {z}",
                "before": f"assets/{case['id']}/{all_before_path.name}",
                "after": f"assets/{case['id']}/{all_after_path.name}",
            })

        coronal_before = case_out / f"{organ_name}_coronal_before.png"
        coronal_after = case_out / f"{organ_name}_coronal_after.png"
        ct_c = fixed_ct_array[z0:z1, y_slice, cx0:cx1]
        render(ct_c, fixed_array[z0:z1, y_slice, cx0:cx1] == organ,
               before_array[z0:z1, y_slice, cx0:cx1] == organ, coronal_before)
        render(ct_c, fixed_array[z0:z1, y_slice, cx0:cx1] == organ,
               after_array[z0:z1, y_slice, cx0:cx1] == organ, coronal_after)

        payload["views"][organ_name] = {
            "axial": axial,
            "all": all_slices,
            "coronal": [{
                "label": f"y {y_slice}",
                "before": f"assets/{case['id']}/{coronal_before.name}",
                "after": f"assets/{case['id']}/{coronal_after.name}",
            }],
        }
        payload["scores"][organ_name] = {
            "before": round(dice(fixed_array, before_array, organ), 5),
            "after": round(dice(fixed_array, after_array, organ), 5),
        }

    print(
        f"{case['id']}: Aorta "
        f"{payload['scores']['aorta']['before']:.3f}→{payload['scores']['aorta']['after']:.3f}, "
        f"Spleen {payload['scores']['spleen']['before']:.3f}→{payload['scores']['spleen']['after']:.3f}"
    )
    return payload


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    payloads = [write_case(case) for case in CASES]
    data = {
        "fixed": FIXED_LABEL,
        "beforeLabel": "Centroid translation",
        "afterLabel": "Rigid (bone/landmark)",
        "cases": payloads,
    }
    (OUT / "data.js").write_text(
        "window.CASES = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(payloads)} cases and {OUT / 'data.js'}")


if __name__ == "__main__":
    main()
