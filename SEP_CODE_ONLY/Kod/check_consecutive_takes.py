"""Check consecutive takes: timestamps, physical coordinates, and table position."""

from pathlib import Path
import json
import SimpleITK as sitk
import numpy as np
import pandas as pd
from datetime import datetime

labeled_dir = Path("PhantomX/LABELED")
sessions = sorted([p for p in labeled_dir.iterdir() if p.is_dir()])

records = []

for sess_path in sessions:
    sess_name = sess_path.name.replace("_PCCT_Aorta_Spleen", "")
    meta_files = sorted(list((sess_path / "nifti").glob("*/dicom_metadata.json")))
    
    if len(meta_files) < 2:
        continue
        
    takes_meta = []
    for mf in meta_files:
        with open(mf, "r", encoding="utf-8") as f:
            d = json.load(f)
        series_name = mf.parent.name
        acq_time = str(d.get("AcquisitionTime", "")).split(".")[0].zfill(6)
        acq_date = str(d.get("AcquisitionDate", ""))
        dt = datetime.strptime(f"{acq_date}_{acq_time}", "%Y%m%d_%H%M%S") if acq_date and acq_time else None
        
        ipp_str = d.get("ImagePositionPatient", "")
        ipp = [float(x) for x in ipp_str.split("\\")] if isinstance(ipp_str, str) and "\\" in ipp_str else [0, 0, 0]
        
        takes_meta.append({
            "series": series_name,
            "take": series_name.split("_")[-1],
            "dt": dt,
            "table_height": float(d.get("TableHeight", 0)) if d.get("TableHeight") is not None else 0,
            "ipp": ipp,
            "irrad_uid": d.get("IrradiationEventUID"),
            "seg_path": sess_path / "segmentation" / series_name / "seg.nii.gz"
        })
        
    takes_meta = sorted(takes_meta, key=lambda x: x["dt"] if x["dt"] else datetime.min)
    
    for i in range(len(takes_meta) - 1):
        t1 = takes_meta[i]
        t2 = takes_meta[i+1]
        
        delta_sec = (t2["dt"] - t1["dt"]).total_seconds() if t1["dt"] and t2["dt"] else None
        th_diff = abs(t2["table_height"] - t1["table_height"])
        ipp_x_diff = abs(t2["ipp"][0] - t1["ipp"][0])
        ipp_y_diff = abs(t2["ipp"][1] - t1["ipp"][1])
        ipp_z_diff = abs(t2["ipp"][2] - t1["ipp"][2])
        
        try:
            im1 = sitk.ReadImage(str(t1["seg_path"]))
            im2 = sitk.ReadImage(str(t2["seg_path"]))
            
            arr1 = sitk.GetArrayFromImage(im1)
            arr2 = sitk.GetArrayFromImage(im2)
            
            # Spleen (1)
            pts1_s = np.argwhere(arr1 == 1)
            pts2_s = np.argwhere(arr2 == 1)
            if pts1_s.size and pts2_s.size:
                c1_idx = pts1_s.mean(axis=0)[::-1]
                c2_idx = pts2_s.mean(axis=0)[::-1]
                p1_s = np.array(im1.TransformContinuousIndexToPhysicalPoint(c1_idx))
                p2_s = np.array(im2.TransformContinuousIndexToPhysicalPoint(c2_idx))
                spleen_disp = np.linalg.norm(p1_s - p2_s)
            else:
                spleen_disp = None
                
            # Aorta (52)
            pts1_a = np.argwhere(arr1 == 52)
            pts2_a = np.argwhere(arr2 == 52)
            if pts1_a.size and pts2_a.size:
                c1_idx = pts1_a.mean(axis=0)[::-1]
                c2_idx = pts2_a.mean(axis=0)[::-1]
                p1_a = np.array(im1.TransformContinuousIndexToPhysicalPoint(c1_idx))
                p2_a = np.array(im2.TransformContinuousIndexToPhysicalPoint(c2_idx))
                aorta_disp = np.linalg.norm(p1_a - p2_a)
            else:
                aorta_disp = None
        except Exception:
            spleen_disp, aorta_disp = None, None
            
        records.append({
            "session": sess_name,
            "pair": f"{t1['take']} -> {t2['take']}",
            "delta_sec": delta_sec,
            "table_height_diff_mm": th_diff,
            "ipp_x_diff_mm": round(ipp_x_diff, 3),
            "ipp_y_diff_mm": round(ipp_y_diff, 3),
            "ipp_z_diff_mm": round(ipp_z_diff, 3),
            "spleen_disp_mm": round(spleen_disp, 4) if spleen_disp is not None else None,
            "aorta_disp_mm": round(aorta_disp, 4) if aorta_disp is not None else None,
            "same_irrad_uid": t1["irrad_uid"] == t2["irrad_uid"]
        })

df_res = pd.DataFrame(records)
print(df_res.to_string())

# Overall Summary Statistics
print("\n=== GENEL ISTATISTIKLER ===")
print("Toplam ardisik cift sayisi:", len(df_res))
print("Masa yuksekligi farki > 0 olan cift sayisi:", (df_res["table_height_diff_mm"] > 0).sum())
print("Z-orijin farki > 0 olan cift sayisi:", (df_res["ipp_z_diff_mm"] > 0).sum())
print("Z-orijin maksimum fark (mm):", df_res["ipp_z_diff_mm"].max())
print("Ortalama zaman araligi (sn):", df_res["delta_sec"].mean(), "Medyan:", df_res["delta_sec"].median())
print("Dalak fiziksel agirlik merkezi kaymasi (mm): Ortalama =", df_res["spleen_disp_mm"].mean(), "Maksimum =", df_res["spleen_disp_mm"].max())
print("Aorta fiziksel agirlik merkezi kaymasi (mm): Ortalama =", df_res["aorta_disp_mm"].mean(), "Maksimum =", df_res["aorta_disp_mm"].max())
