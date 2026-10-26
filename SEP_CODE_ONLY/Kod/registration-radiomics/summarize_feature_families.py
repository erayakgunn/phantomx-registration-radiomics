"""Aggregate V6 repeatability and FDR results by radiomic feature family."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FAMILY = {
    "morph": "Shape",
    "loc": "Local intensity peaks",
    "stat": "First-order statistics",
    "ih": "Intensity histogram",
    "ivh": "Intensity-volume histogram",
    "cm": "GLCM",
    "rlm": "GLRLM",
    "szm": "GLSZM",
    "dzm": "GLDZM",
    "ngt": "NGTDM",
    "ngl": "NGLDM",
}


def family(feature: str) -> str:
    return FAMILY.get(feature.split("_", 1)[0], "Other")


def main(args: argparse.Namespace) -> None:
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("3d", "2d"):
        comparisons = pd.read_csv(args.analysis_dir / f"repeatability_comparison_{mode}.csv")
        significance = pd.read_csv(args.analysis_dir / f"significance_{mode}.csv")
        comparisons["family"] = comparisons["feature"].map(family)
        significance["family"] = significance["feature"].map(family)
        groups = []
        for (organ, feature_family), group in comparisons.groupby(["organ", "family"]):
            sig = significance[
                (significance["organ"] == organ)
                & (significance["family"] == feature_family)
                & significance["significant_fdr_05"]
            ]
            groups.append({
                "organ": organ,
                "family": feature_family,
                "n_features": len(group),
                "cv_improved": int(group["cv_improved"].sum()),
                "pairwise_difference_improved": int(group["pairwise_difference_improved"].sum()),
                "both_improved": int((group["cv_improved"] & group["pairwise_difference_improved"]).sum()),
                "median_delta_cv": float(group["delta_median_cv"].median()),
                "median_delta_pairwise_relative_difference": float(
                    group["delta_median_pairwise_difference"].median()
                ),
                "fdr_significant_tests": len(sig),
                "fdr_significant_improved": int((sig["direction"] == "improved").sum()),
                "fdr_significant_worsened": int((sig["direction"] != "improved").sum()),
            })
        pd.DataFrame(groups).sort_values(["organ", "family"]).to_csv(
            args.analysis_dir / f"feature_family_summary_{mode}.csv", index=False
        )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Summarize repeatability by feature family")
    ap.add_argument("analysis_dir", type=Path)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
