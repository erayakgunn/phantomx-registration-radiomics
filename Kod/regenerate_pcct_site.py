"""Regenerate the PCCT site assets/data from the current batch.

Assets and a temporary standalone data.js are produced in a temp folder, then
copied into the multi-dataset site while preserving sep/zurich.
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "registration-radiomics"))
import annotate_site_artifacts as artifact_annotator  # noqa: E402
import generate_all_assets as g  # noqa: E402

PCCT_ROOT = Path(
    r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX"
)
TEMP = Path(r"C:/Users/eray/AppData/Local/Temp/opencode/pcct_site_v1.0.0")
SITE = Path(r"C:/Users/eray/Desktop/phantomx-overlay-site")
DATA_JS = SITE / "data.js"

g.LAB = PCCT_ROOT / "LABELED"
g.BATCH = PCCT_ROOT / "REGISTRATION" / "batch_to_20260805_v1.0.0"
g.OUT = TEMP
g.ASSETS = TEMP / "assets"
g.QC_FLAGS = {}
g.FIXED_DAY = "20260805"
g.SESSION_SUFFIX = "_PCCT_Aorta_Spleen"
g.FIXED_BASE = (
    "USZ_Siemens_Healthineers_NAEOTOM_Alpha_20260805_Q3_Qr40f_120_100_300_1_"
    "PolyE_take1"
)
g.FIXED_LABEL = "20260805_t1"

print("generating PCCT assets...")
g.main()

print("copying assets into site...")
for child in (TEMP / "assets").iterdir():
    if child.is_dir():
        shutil.copytree(child, SITE / "assets" / child.name, dirs_exist_ok=True)

print("merging pcct into site data.js...")
temp_payload = json.loads(
    (TEMP / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
)
text = DATA_JS.read_text(encoding="utf-8")
obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
obj["pcct"] = {
    "fixed": temp_payload["fixed"],
    "beforeLabel": temp_payload["beforeLabel"],
    "afterLabel": temp_payload["afterLabel"],
    "label": "PCCT",
    "cases": temp_payload["cases"],
}
new = "window.DATASETS = " + json.dumps(obj, ensure_ascii=False, indent=2) + ";\n"
new += 'const DATASET_KEY = new URLSearchParams(location.search).get("dataset") || "sep";\n'
new += "window.CASES = window.DATASETS[DATASET_KEY] || window.DATASETS.sep;\n"
DATA_JS.write_text(new, encoding="utf-8")
print(f"PCCT site updated: {len(temp_payload['cases'])} cases")
artifact_annotator.main()
