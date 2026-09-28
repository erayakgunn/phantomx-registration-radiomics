"""Session-aware repeatability statistics for baseline vs registered features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META = {
    "registration_pair", "analysis_method", "radiomics_mode", "sample_name",
    "image_file_name", "image_directory", "image_mask_label", "image_mask_file_name",
    "image_mask_directory", "image_modality", "image_settings_id",
}


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    result = []
    for column in frame.columns:
        if column in META or column.startswith("dcm_"):
            continue
        if pd.to_numeric(frame[column], errors="coerce").notna().any():
            result.append(column)
    return result


def pairwise_relative_difference(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    differences = []
    for index in range(len(values)):
        for other in range(index + 1, len(values)):
            denominator = max(abs(values[index]), abs(values[other]), 1e-12)
            differences.append(abs(values[index] - values[other]) / denominator)
    return float(np.median(differences))


def session_statistics(values: pd.Series) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "cv": np.nan,
                "median_pairwise_relative_difference": np.nan}
    mean = float(np.mean(array))
    return {
        "n": len(array),
        "mean": mean,
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "cv": float(np.std(array, ddof=1) / abs(mean)) if len(array) > 1 and mean else np.nan,
        "median_pairwise_relative_difference": pairwise_relative_difference(array),
    }


def load_values(dataset_root: Path, registration_root: Path, mode: str) -> pd.DataFrame:
    registered = pd.read_csv(registration_root / f"all_features_registered_{mode}.csv")
    rows = []
    for pair in sorted(registered["registration_pair"].unique()):
        if "__" not in pair:
            continue
        session, series = pair.split("__", 1)
        baseline_path = dataset_root / session / "features" / series / (
            "features.csv" if mode == "3d" else "features_2d.csv"
        )
        if not baseline_path.exists():
            continue
        baseline = pd.read_csv(baseline_path)
        current = registered[registered["registration_pair"] == pair]
        for organ in ("spleen", "aorta"):
            base_row = baseline[baseline["image_mask_label"] == organ]
            reg_row = current[current["image_mask_label"] == organ]
            if base_row.empty or reg_row.empty:
                continue
            base_row, reg_row = base_row.iloc[0], reg_row.iloc[0]
            for feature in numeric_columns(current):
                if feature not in baseline.columns:
                    continue
                rows.append({
                    "session": session,
                    "series": series,
                    "organ": organ,
                    "feature": feature,
                    "baseline": pd.to_numeric(pd.Series([base_row[feature]]), errors="coerce").iloc[0],
                    "registered": pd.to_numeric(pd.Series([reg_row[feature]]), errors="coerce").iloc[0],
                })
    return pd.DataFrame(rows)


def analyze(values: pd.DataFrame, output: Path, mode: str) -> None:
    rows = []
    for (session, organ, feature), group in values.groupby(["session", "organ", "feature"]):
        for method in ("baseline", "registered"):
            stats = session_statistics(group[method])
            rows.append({
                "session": session,
                "organ": organ,
                "feature": feature,
                "method": method,
                **stats,
            })
    session_stats = pd.DataFrame(rows)
    session_stats.to_csv(output / f"session_repeatability_{mode}.csv", index=False)

    summary_rows = []
    for (organ, feature, method), group in session_stats.groupby(["organ", "feature", "method"]):
        summary_rows.append({
            "organ": organ,
            "feature": feature,
            "method": method,
            "n_sessions": int(group["session"].nunique()),
            "median_cv": float(group["cv"].median()),
            "median_pairwise_relative_difference": float(
                group["median_pairwise_relative_difference"].median()
            ),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / f"repeatability_summary_{mode}.csv", index=False)

    baseline = summary[summary["method"] == "baseline"].drop(columns="method")
    registered = summary[summary["method"] == "registered"].drop(columns="method")
    comparison = baseline.merge(registered, on=["organ", "feature"], suffixes=("_baseline", "_registered"))
    comparison["delta_median_cv"] = comparison["median_cv_registered"] - comparison["median_cv_baseline"]
    comparison["delta_median_pairwise_difference"] = (
        comparison["median_pairwise_relative_difference_registered"]
        - comparison["median_pairwise_relative_difference_baseline"]
    )
    comparison["cv_improved"] = comparison["delta_median_cv"] < 0
    comparison["pairwise_difference_improved"] = comparison["delta_median_pairwise_difference"] < 0
    comparison.to_csv(output / f"repeatability_comparison_{mode}.csv", index=False)


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in ("3d", "2d"):
        values = load_values(args.dataset_root, args.registration_root, mode)
        values.to_csv(args.output / f"feature_values_for_repeatability_{mode}.csv", index=False)
        analyze(values, args.output, mode)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Session-aware radiomics repeatability analysis")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("registration_root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
