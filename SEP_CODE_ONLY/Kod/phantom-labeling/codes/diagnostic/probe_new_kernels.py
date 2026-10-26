"""Survey star + texture polish on different kernel families (Qr36 /
Qr40 / Qr64) using the labeled output from the new dataset.

For each labeled series, parse the per-section polish line from the
labeler's run.log and report:
  - protocol (kVp / mAs / Q-level / kernel)
  - star polish shift magnitude
  - whether shift is consistent with other series of same kernel
"""
import os, re, sys
from collections import defaultdict

LOG = "C:/output_pcct_v2/run.log"

annotating_re = re.compile(r"Annotating:\s+(\S+)")
polish_re = re.compile(
    r"(STAR1?6|STAR8)\s+mean-image star centre polish:\s+"
    r"\(([-\d\.]+),\s*([-\d\.]+)\)\s+\[shifted\s+\(([+\-\d\.]+),\s*([+\-\d\.]+)\)"
)
texture_polish_re = re.compile(
    r"(STAR1?6|STAR8)\s+mean-image texture polish:\s+refined (\d+)/(\d+) textures"
)
pve_re = re.compile(r"(STAR1?6|STAR8):\s+adaptive PVE guard trimmed (\d+) lead \+ (\d+) trail")

def parse_protocol(name: str):
    # Folder name: ..._Qr40f_120_100_300_1_recon1mm_MonoE70keV_take1_recon1mm
    # Tokens: ..._<kernel>_<kvp>_<mas>_<fov>_<slice>_recon<X>mm_<reconType>_takeN_recon<X>mm
    m = re.search(r"_(Qr\d+\w?)_(\d+)_(\d+)_\d+_(\d+)_", name)
    if not m:
        return None
    kernel, kvp, mas, slice_thk = m.groups()
    # Q-level: _QN_
    qm = re.search(r"_Q(\d+)_", name)
    q = qm.group(1) if qm else "?"
    return f"kVp={kvp} mAs={mas} thk={slice_thk}mm  Q{q}  {kernel}"


def main():
    if not os.path.exists(LOG):
        print(f"No log at {LOG}")
        return
    series_data = {}
    cur = None
    with open(LOG, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = annotating_re.search(line)
            if m:
                cur = m.group(1)
                series_data[cur] = {
                    "polish": {},
                    "texture_polish": {},
                    "pve": {},
                }
                continue
            if cur is None:
                continue
            m = polish_re.search(line)
            if m:
                sec, _, _, sx, sy = m.groups()
                series_data[cur]["polish"][sec] = (float(sx), float(sy))
            m = texture_polish_re.search(line)
            if m:
                sec, refined, total = m.groups()
                series_data[cur]["texture_polish"][sec] = (
                    int(refined), int(total))
            m = pve_re.search(line)
            if m:
                sec, ld, tr = m.groups()
                series_data[cur]["pve"][sec] = (int(ld), int(tr))

    print(f"\n{'='*78}\nPipeline summary across {len(series_data)} series\n{'='*78}\n")
    by_kernel = defaultdict(list)
    for name, data in series_data.items():
        proto = parse_protocol(name)
        if not proto:
            continue
        kernel = proto.split()[-1]
        by_kernel[kernel].append((proto, name, data))

    for kernel in sorted(by_kernel.keys()):
        items = by_kernel[kernel]
        print(f"\n--- Kernel {kernel} ({len(items)} series) ---")
        for proto, name, data in sorted(items):
            star16 = data["polish"].get("STAR16", (None, None))
            star8 = data["polish"].get("STAR8", (None, None))
            tex16 = data["texture_polish"].get("STAR16", (None, None))
            tex8 = data["texture_polish"].get("STAR8", (None, None))
            pve16 = data["pve"].get("STAR16", (0, 0))
            pve8 = data["pve"].get("STAR8", (0, 0))
            if star16[0] is None:
                print(f"  {proto}: (incomplete)")
                continue
            print(f"  {proto:55s}")
            print(f"    STAR16: star shift=({star16[0]:+5.2f},{star16[1]:+5.2f}) "
                  f"tex={tex16[0]}/{tex16[1]} pve=L{pve16[0]}T{pve16[1]}")
            print(f"    STAR8:  star shift=({star8[0]:+5.2f},{star8[1]:+5.2f}) "
                  f"tex={tex8[0]}/{tex8[1]} pve=L{pve8[0]}T{pve8[1]}")


if __name__ == "__main__":
    main()
