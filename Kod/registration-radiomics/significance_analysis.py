"""Paired significance testing for session-level repeatability summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def test_mode(analysis_dir: Path, mode: str) -> pd.DataFrame:
    path = analysis_dir / f"session_repeatability_{mode}.csv"
    data = pd.read_csv(path)
    rows = []
    for (organ, feature), group in data.groupby(["organ", "feature"]):
        wide = group.pivot(index="session", columns="method", values=["cv", "median_pairwise_relative_difference"])
        for measure in ("cv", "median_pairwise_relative_difference"):
            if "baseline" not in wide[measure] or "registered" not in wide[measure]:
                continue
            pair = wide[measure][["baseline", "registered"]].dropna()
            difference = pair["registered"] - pair["baseline"]
            if len(pair) < 3 or np.allclose(difference.to_numpy(), 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    pair["registered"],
                    pair["baseline"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append({
                "organ": organ,
                "feature": feature,
                "measure": measure,
                "n_sessions": len(pair),
                "median_delta_registered_minus_baseline": float(difference.median()),
                "mean_delta_registered_minus_baseline": float(difference.mean()),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value_fdr_bh"] = benjamini_hochberg(result["p_value"])
        result["significant_fdr_05"] = result["q_value_fdr_bh"] < 0.05
        result["direction"] = np.where(
            result["median_delta_registered_minus_baseline"] < 0,
            "improved",
            "worsened_or_unchanged",
        )
    return result


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in ("3d", "2d"):
        result = test_mode(args.analysis_dir, mode)
        result.to_csv(args.output / f"significance_{mode}.csv", index=False)
        print(
            f"{mode}: {int(result['significant_fdr_05'].sum()) if not result.empty else 0} "
            "FDR-significant tests"
        )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Test baseline vs registered repeatability")
    ap.add_argument("analysis_dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
