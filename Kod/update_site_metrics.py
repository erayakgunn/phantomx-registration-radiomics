"""Refresh the site data.js residual metrics (HD95/ASSD) from the recomputed
metrics.csv files, preserving the existing Dice scores and assets.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SITE = Path(r"C:/Users/eray/Desktop/phantomx-overlay-site")
DATA_JS = SITE / "data.js"
PCCT = Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX")
SEP = next(Path(r"C:/Users/eray/Desktop").glob("SEP -*")) / "PhantomX"
ZURICH = {
    "zpcct": Path(r"C:/Users/eray/Desktop/zurich_pcct_work"),
    "zsep": Path(r"C:/Users/eray/Desktop/zurich_sep_work"),
}


def read_residual(metrics_path: Path) -> dict:
    out = {}
    with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                label = int(row["label"])
            except (KeyError, ValueError):
                continue
            organ = "spleen" if label in (1, 3) else "aorta" if label in (52, 6) else None
            if organ is None:
                continue
            out[organ] = {
                "hd95": round(float(row["after_hd95_mm"]), 4),
                "assd": round(float(row["after_assd_mm"]), 4),
                "centroid": round(float(row["after_centroid_displacement_mm"]), 4),
            }
    return out


def main() -> None:
    text = DATA_JS.read_text(encoding="utf-8")
    obj = json.loads(text[text.index("{"):text.rindex("}") + 1])

    for case in obj["sep"]["cases"]:
        session = f"{case['day']}_SEP_Aorta_Spleen"
        p = SEP / "REGISTRATION" / "batch_to_20260805_v6_skip2" / f"{session}__{case['seriesFull']}" / "metrics.csv"
        if p.exists():
            case["residual"] = read_residual(p)

    for case in obj["pcct"]["cases"]:
        session = f"{case['day']}_PCCT_Aorta_Spleen"
        p = PCCT / "REGISTRATION" / "batch_to_20260805_v1.0.0" / f"{session}__{case['seriesFull']}" / "metrics.csv"
        if p.exists():
            case["residual"] = read_residual(p)

    for case in obj["zurich"]["cases"]:
        prefix = "zpcct" if case["id"].startswith("zpcct") else "zsep"
        work = ZURICH[prefix]
        index = int(case["id"].split("_")[1])
        groups = sorted(p for p in (work / "REGISTRATION").iterdir() if p.is_dir())
        group = groups[index - 1]
        p = group / f"{group.name}__{case['seriesFull']}" / "metrics.csv"
        if p.exists():
            case["residual"] = read_residual(p)

    new = "window.DATASETS = " + json.dumps(obj, ensure_ascii=False, indent=2) + ";\n"
    new += 'const DATASET_KEY = new URLSearchParams(location.search).get("dataset") || "sep";\n'
    new += "window.CASES = window.DATASETS[DATASET_KEY] || window.DATASETS.sep;\n"
    DATA_JS.write_text(new, encoding="utf-8")
    print("site residual metrics updated")


if __name__ == "__main__":
    main()
