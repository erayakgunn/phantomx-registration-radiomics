"""Annotate the site data with kidney segmentation artifacts.

Reads KIDNEY_CC_SCAN.csv and adds ``artifact`` (isolated component) or
``segFailure`` (single component but genuine poor segmentation) to matching
cases in phantomx-overlay-site/data.js.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SITE = Path(r"C:/Users/eray/Desktop/phantomx-overlay-site")
DATA_JS = SITE / "data.js"
SCAN = Path(r"C:/Users/eray/Desktop/KIDNEY_CC_SCAN.csv")

DATASET_KEY = {
    "PCCT cross-date": "pcct",
    "SEP cross-date": "sep",
    "PCCT test-retest": "zurich",
    "SEP test-retest": "zurich",
}
ORGAN_LABEL = {
    "kidney_right": "Sağ böbrek",
    "kidney_left": "Sol böbrek",
}


def main() -> None:
    if not SCAN.exists():
        print(f"[skip] {SCAN} yok")
        return
    frame = pd.read_csv(SCAN)
    text = DATA_JS.read_text(encoding="utf-8")
    obj = json.loads(text[text.index("{"):text.rindex("}") + 1])

    index = {}
    for key in ("pcct", "sep", "zurich"):
        for case in obj[key]["cases"]:
            case.pop("artifact", None)
            case.pop("segFailure", None)
            index[(key, str(case.get("seriesFull", "")))] = case

    flagged = 0
    for _, row in frame.iterrows():
        is_island = int(row["After_components"]) > 1
        is_failure = (
            int(row["After_components"]) == 1
            and float(row["Raw_ASSD_mm"]) > 1.0
        )
        if not (is_island or is_failure):
            continue

        key = DATASET_KEY.get(str(row["Dataset"]))
        if key is None:
            continue
        case = index.get((key, str(row["Pair"])))
        if case is None:
            print(f"[skip] {row['Dataset']} {row['Pair']}")
            continue

        organ = str(row["Organ"])
        label = ORGAN_LABEL.get(organ, organ)
        if is_island:
            bucket = case.setdefault("artifact", {"organs": {}, "summary": ""})
            bucket["organs"][organ] = {
                "components": int(row["After_components"]),
                "largest_fraction": float(row["Largest_fraction"]),
                "small_volume_mm3": float(row["Small_volume_mm3"]),
                "raw_assd": float(row["Raw_ASSD_mm"]),
                "largest_assd": float(row["Largest_ASSD_mm"]),
                "raw_mssd": float(row["Raw_MSSD_mm"]),
                "largest_mssd": float(row["Largest_MSSD_mm"]),
            }
            part = (
                f"{label}: {int(row['After_components'])} bileşen, "
                f"~{float(row['Small_volume_mm3']):.0f} mm³ adacık; "
                f"ASSD {float(row['Raw_ASSD_mm']):.2f}→"
                f"{float(row['Largest_ASSD_mm']):.2f} mm"
            )
            bucket["summary"] = (
                f"{bucket['summary']}; {part}" if bucket["summary"] else part
            )
        else:
            bucket = case.setdefault("segFailure", {"organs": {}, "summary": ""})
            bucket["organs"][organ] = {
                "components": int(row["After_components"]),
                "raw_dice": float(row["Raw_Dice"]),
                "raw_assd": float(row["Raw_ASSD_mm"]),
                "raw_mssd": float(row["Raw_MSSD_mm"]),
                "raw_hd95": float(row["Raw_HD95_mm"]),
            }
            part = (
                f"{label}: tek bileşen, gerçek segmentasyon/recon hatası; "
                f"Dice {float(row['Raw_Dice']):.3f}, "
                f"ASSD {float(row['Raw_ASSD_mm']):.2f} mm"
            )
            bucket["summary"] = (
                f"{bucket['summary']}; {part}" if bucket["summary"] else part
            )
        flagged += 1

    new = "window.DATASETS = " + json.dumps(obj, ensure_ascii=False, indent=2) + ";\n"
    new += 'const DATASET_KEY = new URLSearchParams(location.search).get("dataset") || "sep";\n'
    new += "window.CASES = window.DATASETS[DATASET_KEY] || window.DATASETS.sep;\n"
    DATA_JS.write_text(new, encoding="utf-8")

    cases = sum(
        1 for key in ("pcct", "sep", "zurich") for case in obj[key]["cases"]
        if case.get("artifact") or case.get("segFailure")
    )
    print(f"flagged rows: {flagged} | unique cases: {cases}")


if __name__ == "__main__":
    main()
