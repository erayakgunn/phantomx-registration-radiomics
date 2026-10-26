# CalibraMics Labeling Codes

PCCT projesinin (NAEOTOM Alpha, USZ) etiketleme ve radiomics kodları. Üç bağımsız ağaç, her biri kendi README'si ve testleriyle. Ayrıntılı bağlam: `D:\PCCT\PCCT_BILGI_TABANI.md` §4.

| Klasör | Rol | Giriş noktası | Son güncelleme |
|---|---|---|---|
| `phantom-labeling\` | CalibraMics fantomunda ROI tespiti (6 faz) ve DICOM patch export; NAEOTOM Alpha, Philips 7500, SOMATOM Force desteği | `codes\label_series.py` (etiketle + annotate), `codes\main.py`, `codes\organize_siemens_pcct.py` | 2026-08-07 |
| `phantom-radiomics\` | Etiketlenmiş patch'lerden MIRP 2.5.0 / IBSI-1 feature çıkarımı, Excel workbook; PyInstaller dağıtımı | `extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py`, `PhantomXPro_Radiomics.spec`, `KULLANIM.txt` | 2026-08-07 |
| `organ-radiomics\` | PhantomX (aorta/spleen): seri keşfi → NIfTI → TotalSegmentator 2.12.0 → MIRP → workbook + doğrulama | `batch_120kvp.py`, `run_and_verify.py`, `config.py` | 2026-08-24 |

Ek dosyalar:
- `IBSI_UYUM_RAPORU.md` — 29 Tem 2026 IBSI uyum ve tekrarlanabilirlik denetimi (anti-aliasing düzeltmesi, reseg aralığı değişimi, B1–B6 bulguları). Her iki ekstraksiyon hattı için beyan metinleri içerir.
- `phantom-labeling\README_AcibademKent_SOMATOM_Force.md` — SOMATOM Force (Acıbadem Kent) etiketleme notları; kodu `phantom-labeling` zaten içeriyor (`labeling\phantom_instances.py`, `--institution`).
- `phantom-radiomics\KSW_Radiomics_eski.spec` — eski PyInstaller reçetesi (KSW adıyla); güncel olan `PhantomXPro_Radiomics.spec`.
- `phantom-radiomics\dist\PhantomXPro_Radiomics\` — dağıtılabilir exe klasörü (1.06 GB; `pyinstaller PhantomXPro_Radiomics.spec --noconfirm` ile yeniden üretilir).

Bu klasörde **veri yok**: DICOM, NIfTI ve radiomics çıktıları `D:\PCCT\PCCT * - LABELED\` ve `* - RADIOMICS\` altındadır.

## 2026-09-06 temizliği

`CalibraMics Phantom Label and Feature Extraction\` şemsiye klasörü (28 Tem 2026'da Masaüstü'nden kopyalanmış eski sürümler) Geri Dönüşüm Kutusu'na taşındı. İçindekiler ve üst seviyedeki karşılıkları (hash ile doğrulandı; farklı olan her dosyada üst seviye daha yeni):

| Kaldırılan | Karşılığı | Not |
|---|---|---|
| `labeling\pcct-sep-error-corrected\` (git geçmişi dahil) | `phantom-labeling` | 57 dosya birebir, 7 dosya Temmuz sürümü |
| `labeling\pcct-labeling\` (git geçmişi dahil) | `phantom-labeling` | 54 birebir, 7 eski + eski README |
| `labeling\acibadem-kent\` | `phantom-labeling` | 52 birebir, 7 eski; README'si kurtarıldı |
| `feature-extraction\pcct-radiomics-mirp\` | `phantom-radiomics` | script 132 KB (eski) vs 171 KB; `KSW_Radiomics.spec` kurtarıldı |
| `feature-extraction\phantomxpro-mirp\` | `organ-radiomics` | 14 dosya eski sürüm |
| `README.md`, `IBSI_UYUM_RAPORU.md` | — | IBSI raporu buraya alındı; README kopyası `D:\PCCT\work\code_cleanup_20260906\umbrella_README_orijinal.md` |

Aynı temizlikte `phantom-radiomics\build\` (142 MB PyInstaller ara dosyaları) ve tüm `__pycache__` / `.pytest_cache` klasörleri Geri Dönüşüm Kutusu'na taşındı. Tam manifest (405 dosya, md5, durum): `D:\PCCT\work\code_cleanup_20260906\umbrella_manifest.csv`; üst ağaçların temizlik öncesi hash listesi: `before_top_level.csv`.

## Testler

```bash
"C:/Python314/python.exe" -m pytest phantom-radiomics/tests -q -m "not slow"
"C:/Python314/python.exe" -m pytest phantom-labeling/tests -q
"C:/Python314/python.exe" -m pytest organ-radiomics/tests -q
```
