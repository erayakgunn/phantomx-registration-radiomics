"""Regenerate the SEP site assets/data from the current 2 mm batch.

Assets + a temporary standalone data.js are produced in a temp folder, then
copied into the multi-dataset site while preserving pcct/zurich.
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

SEP_ROOT = next(Path(r"C:/Users/eray/Desktop").glob("SEP -*")) / "PhantomX"
TEMP = Path(r"C:/Users/eray/AppData/Local/Temp/opencode/sep_site_v6_skip2")
SITE = Path(r"C:/Users/eray/Desktop/phantomx-overlay-site")
DATA_JS = SITE / "data.js"

g.LAB = SEP_ROOT / "LABELED"
g.BATCH = SEP_ROOT / "REGISTRATION" / "batch_to_20260805_v6_skip2"
g.OUT = TEMP
g.ASSETS = TEMP / "assets"
g.QC_FLAGS = {}
g.FIXED_DAY = "20260805"
g.FIXED_BASE = (
    "USZ_SIEMENS_SOMATOM_Edge_Plus_20260805_S3_Br40f_120_100_300_1_"
    "PolyE_take1_recon1mm"
)
g.FIXED_LABEL = "20260805_t1"

print("generating SEP assets...")
g.main()

print("copying assets into site...")
for child in (TEMP / "assets").iterdir():
    if child.is_dir():
        shutil.copytree(child, SITE / "assets" / child.name, dirs_exist_ok=True)

print("merging sep into site data.js...")
temp_payload = json.loads(
    (TEMP / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
)
text = DATA_JS.read_text(encoding="utf-8")
obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
obj["sep"] = {
    "fixed": temp_payload["fixed"],
    "beforeLabel": temp_payload["beforeLabel"],
    "afterLabel": temp_payload["afterLabel"],
    "label": "SEP",
    "cases": temp_payload["cases"],
}
new = "window.DATASETS = " + json.dumps(obj, ensure_ascii=False, indent=2) + ";\n"
new += 'const DATASET_KEY = new URLSearchParams(location.search).get("dataset") || "sep";\n'
new += "window.CASES = window.DATASETS[DATASET_KEY] || window.DATASETS.sep;\n"
DATA_JS.write_text(new, encoding="utf-8")
print(f"SEP site updated: {len(temp_payload['cases'])} cases")
artifact_annotator.main()
