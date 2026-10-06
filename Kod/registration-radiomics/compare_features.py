"""Compare baseline and registration-based MIRP feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META_COLUMNS = {
    "registration_pair", "analysis_method", "radiomics_mode", "sample_name",
    "image_file_name", "image_directory", "image_mask_file_name",
    "image_mask_directory", "image_modality", "image_settings_id",
}


def numeric_features(frame: pd.DataFrame) -> list[str]:
    excluded = META_COLUMNS | {c for c in frame.columns if c.startswith("dcm_")}
    result = []
    for column in frame.columns:
        if (column in excluded or column == "image_mask_label"
                or column.startswith(("image_", "sample_", "dcm_"))):
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            result.append(column)
    return result


def icc_31(values: np.ndarray) -> float:
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 2:
        return float("nan")
    n, k = values.shape
    row_means = values.mean(axis=1)
    grand_mean = values.mean()
    ms_between = k * float(((row_means - grand_mean) ** 2).sum()) / (n - 1)
    ms_error = float(((values - row_means[:, None]) ** 2).sum()) / (n * (k - 1))
    denominator = ms_between + (k - 1) * ms_error
    return float("nan") if denominator == 0 else (ms_between - ms_error) / denominator


def main(args: argparse.Namespace) -> None:
    registered = {
        mode: pd.read_csv(args.registration_root / f"all_features_registered_{mode}.csv")
        for mode in ("3d", "2d")
    }
    rows_by_mode: dict[str, list[dict]] = {"3d": [], "2d": []}
    for pair_dir in sorted(args.registration_root.iterdir()):
        if not pair_dir.is_dir() or "__" not in pair_dir.name:
            continue
        session, series = pair_dir.name.split("__", 1)
        baseline_dir = args.dataset_root / session / "features" / series
        for mode, frame in registered.items():
            baseline_path = baseline_dir / ("features.csv" if mode == "3d" else "features_2d.csv")
            if not baseline_path.exists():
                continue
            base = pd.read_csv(baseline_path)
            reg = frame[frame["registration_pair"] == pair_dir.name]
            for organ in ("spleen", "aorta"):
                base_row = base[base["image_mask_label"] == organ]
                reg_row = reg[reg["image_mask_label"] == organ]
                if base_row.empty or reg_row.empty:
                    continue
                base_values = base_row.iloc[0]
                reg_values = reg_row.iloc[0]
                for feature in numeric_features(reg):
                    if feature not in base.columns:
                        continue
                    b = pd.to_numeric(pd.Series([base_values[feature]]), errors="coerce").iloc[0]
                    r = pd.to_numeric(pd.Series([reg_values[feature]]), errors="coerce").iloc[0]
                    rows_by_mode[mode].append({
                        "pair": pair_dir.name,
                        "session": session,
                        "series": series,
                        "organ": organ,
                        "feature": feature,
                        "baseline": b,
                        "registered": r,
                    })

    args.output.mkdir(parents=True, exist_ok=True)
    for mode, rows in rows_by_mode.items():
        values = pd.DataFrame(rows)
        values.to_csv(args.output / f"feature_values_{mode}.csv", index=False)
        summary = []
        if not values.empty:
            for (organ, feature), group in values.groupby(["organ", "feature"]):
                b = group["baseline"].to_numpy(dtype=float)
                r = group["registered"].to_numpy(dtype=float)
                valid = np.isfinite(b) & np.isfinite(r)
                b, r = b[valid], r[valid]
                if len(b) == 0:
                    continue
                diff = np.abs(r - b)
                denom = np.maximum(np.abs(b), 1e-12)
                correlation = float(np.corrcoef(b, r)[0, 1]) if len(b) > 1 and np.std(b) and np.std(r) else float("nan")
                summary.append({
                    "organ": organ,
                    "feature": feature,
                    "n": len(b),
                    "mean_abs_difference": float(np.mean(diff)),
                    "median_abs_difference": float(np.median(diff)),
                    "mean_relative_abs_difference": float(np.mean(diff / denom)),
                    "baseline_cv": float(np.std(b, ddof=1) / abs(np.mean(b))) if len(b) > 1 and np.mean(b) else float("nan"),
                    "registered_cv": float(np.std(r, ddof=1) / abs(np.mean(r))) if len(r) > 1 and np.mean(r) else float("nan"),
                    "pearson_r": correlation,
                    "icc_3_1": icc_31(np.column_stack([b, r])),
                })
        pd.DataFrame(summary).to_csv(args.output / f"feature_comparison_{mode}.csv", index=False)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare baseline and registered MIRP features")
    ap.add_argument("dataset_root", type=Path, help="PhantomX/LABELED")
    ap.add_argument("registration_root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
