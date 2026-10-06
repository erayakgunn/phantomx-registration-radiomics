"""Paired session bootstrap confidence intervals for V6 repeatability changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "cv": "coefficient_of_variation",
    "median_pairwise_relative_difference": "median_pairwise_relative_difference",
}


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator,
                        iterations: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    sampled = rng.choice(values, size=(iterations, values.size), replace=True)
    medians = np.median(sampled, axis=1)
    return (
        float(np.median(values)),
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    results = []
    for mode in ("3d", "2d"):
        source = pd.read_csv(args.analysis_dir / f"session_repeatability_{mode}.csv")
        for metric, metric_name in METRICS.items():
            wide = source.pivot(
                index=["session", "organ", "feature"],
                columns="method",
                values=metric,
            ).reset_index()
            wide["delta_registered_minus_baseline"] = wide["registered"] - wide["baseline"]
            for (organ, feature), group in wide.groupby(["organ", "feature"]):
                delta = group["delta_registered_minus_baseline"].to_numpy(dtype=float)
                median, lower, upper = bootstrap_median_ci(delta, rng, args.iterations)
                results.append({
                    "mode": mode,
                    "metric": metric_name,
                    "organ": organ,
                    "feature": feature,
                    "n_sessions": int(np.isfinite(delta).sum()),
                    "median_delta_registered_minus_baseline": median,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "ci_excludes_zero": bool(upper < 0.0 or lower > 0.0),
                    "direction": "improved" if upper < 0.0 else "worsened" if lower > 0.0 else "uncertain",
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Bootstrap CIs for paired session repeatability")
    ap.add_argument("analysis_dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261006)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
