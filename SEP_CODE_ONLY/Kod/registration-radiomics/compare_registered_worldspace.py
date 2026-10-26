"""Isolate registration effects from identity/world-space resampling effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META = {"registration_pair", "analysis_method", "radiomics_mode", "sample_name", "image_mask_label"}


def numeric_features(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if column in META or column.startswith(("image_", "sample_", "dcm_")):
            continue
        if pd.to_numeric(frame[column], errors="coerce").notna().any():
            columns.append(column)
    return columns


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in ("3d", "2d"):
        registered = pd.read_csv(args.registered_root / f"all_features_registered_{mode}.csv")
        worldspace_root = args.worldspace_root or args.registered_root
        worldspace = pd.read_csv(worldspace_root / f"all_features_worldspace_{mode}.csv")
        if not args.keep_flagged:
            for frame in (registered, worldspace):
                if "qc_flag" in frame.columns:
                    frame.drop(frame.index[frame["qc_flag"].fillna("") != ""], inplace=True)
        rows = []
        common = sorted(set(registered["registration_pair"]) & set(worldspace["registration_pair"]))
        features = sorted(set(numeric_features(registered)) & set(numeric_features(worldspace)))
        for pair in common:
            reg_pair = registered[registered["registration_pair"] == pair]
            world_pair = worldspace[worldspace["registration_pair"] == pair]
            for organ in ("spleen", "aorta"):
                reg_row = reg_pair[reg_pair["image_mask_label"] == organ]
                world_row = world_pair[world_pair["image_mask_label"] == organ]
                if reg_row.empty or world_row.empty:
                    continue
                reg_row, world_row = reg_row.iloc[0], world_row.iloc[0]
                for feature in features:
                    registered_value = pd.to_numeric(pd.Series([reg_row[feature]]), errors="coerce").iloc[0]
                    world_value = pd.to_numeric(pd.Series([world_row[feature]]), errors="coerce").iloc[0]
                    if pd.isna(registered_value) or pd.isna(world_value):
                        continue
                    difference = abs(float(registered_value) - float(world_value))
                    rows.append({
                        "pair": pair,
                        "organ": organ,
                        "feature": feature,
                        "worldspace": world_value,
                        "registered": registered_value,
                        "absolute_difference": difference,
                        "relative_difference": difference / max(abs(float(world_value)), 1e-12),
                    })
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output / f"worldspace_vs_registered_values_{mode}.csv", index=False)
        summary = frame.groupby(["organ", "feature"]).agg(
            n=("absolute_difference", "size"),
            mean_absolute_difference=("absolute_difference", "mean"),
            median_absolute_difference=("absolute_difference", "median"),
            mean_relative_difference=("relative_difference", "mean"),
        ).reset_index()
        summary.to_csv(args.output / f"worldspace_vs_registered_{mode}.csv", index=False)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare identity world-space and registered features")
    ap.add_argument("registered_root", type=Path)
    ap.add_argument("--worldspace-root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--keep-flagged", action="store_true",
                    help="QC-flagged pair'leri (qc_flag dolu) analize dahil et")
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
