"""Measure slice-to-slice fluctuation of texture/star ROI centers in
the labeled output.

For each STAR section in each labeled series, reads the metadata.json
and computes the standard deviation of each ROI's (center_x, center_y)
across the section's slices. Sub-pixel std (< ~0.1 px) means smooth /
no fluctuation. Multi-pixel std means visible wobble.
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict


def main():
    output_root = sys.argv[1] if len(sys.argv) > 1 else "C:/output_pcct"
    series_dirs = sorted([
        os.path.join(output_root, d) for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
        and os.path.exists(os.path.join(output_root, d, "metadata.json"))
    ])

    for sd in series_dirs:
        with open(os.path.join(sd, "metadata.json")) as fh:
            md = json.load(fh)
        print(f"\n{os.path.basename(sd)[:80]}")
        for sec_name in ("STAR16", "STAR8"):
            sec = md.get("sections", {}).get(sec_name)
            if not sec:
                continue
            by_label = defaultdict(list)  # label -> [(slice_idx, cx, cy), ...]
            for roi in sec["rois"]:
                by_label[roi["label"]].append(
                    (roi["slice_idx"], roi["center_x"], roi["center_y"])
                )
            print(f"  {sec_name}:")
            for label, items in sorted(by_label.items()):
                items.sort()
                cxs = np.array([it[1] for it in items])
                cys = np.array([it[2] for it in items])
                # Consecutive-slice differences (max absolute step)
                if len(cxs) > 1:
                    diffs_x = np.diff(cxs)
                    diffs_y = np.diff(cys)
                    max_step = float(np.max(np.sqrt(diffs_x**2 + diffs_y**2)))
                else:
                    max_step = 0.0
                print(f"    {label:14s}  n={len(items):3d}  "
                      f"std=(x:{cxs.std():.3f}, y:{cys.std():.3f})  "
                      f"max consecutive step={max_step:.2f} px")


if __name__ == "__main__":
    main()
