# SEP Registration Code Bundle — Windows Handoff

This bundle contains code, reports, and configuration examples only. It contains
no DICOM, NIfTI, segmentation, JPEG, PNG, `.tfm`, `.venv`, or `assets/` files.

## Target Windows workspace

Existing data is expected at the project root, for example:

```text
C:\Users\<user>\Desktop\PCCT - Q3_Qr40f_120_100_300_1_PolyE\
  PhantomX\LABELED\<session>\nifti\<series>\ct.nii.gz
  PhantomX\LABELED\<session>\segmentation\<series>\seg.nii.gz
  PhantomX\REGISTRATION\batch_to_20260805_v6\
  .venv\Scripts\python.exe
```

The audited Windows dataset has 89 `seg.nii.gz` files across 22 PhantomX
sessions. The existing `batch_to_20260805_v6` has fixed
`20260805_PCCT_Aorta_Spleen` / `..._take1` and 88 moving pairs.

## Canonical code in this bundle

- `Kod/registration-radiomics/`: registration, metrics, batch, reports.
- `registration_overlay_ui/`: gallery/detail website and asset generator.
- `run_site.ps1`: sets the target paths and runs the website generator.

The target project uses top-level `Kod/`, not `PCCT/Kod/`. The UI import has a
fallback for both layouts.

## Batch selection

`batch_to_20260805_v6` is not automatically the final result. It is only the
latest existing batch folder and may have been produced by older code. Validate
it before publishing, or preferably create a fresh batch from the current code.
Do not use the old `RAPOR_AKADEMIK_REGISTRATION_RADIOMICS_V6.docx` as evidence
that the v6 batch is current; they are separate artifacts.

The site runner requires an explicit batch path:

```powershell
powershell -ExecutionPolicy Bypass -File .\SEP_CODE_ONLY\run_site.ps1 `
  -ProjectRoot . `
  -BatchPath .\PhantomX\REGISTRATION\batch_to_20260805_v7
```

Use v6 only for a UI smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File .\SEP_CODE_ONLY\run_site.ps1 `
  -ProjectRoot . `
  -BatchPath .\PhantomX\REGISTRATION\batch_to_20260805_v6
```

Validate a batch before using it:

```powershell
$b = ".\PhantomX\REGISTRATION\batch_to_20260805_v6"
Get-Item $b | Select-Object FullName, LastWriteTime
Get-Item "$b\batch_manifest.json" | Select-Object LastWriteTime
Get-Content "$b\batch_manifest.json" -Raw | ConvertFrom-Json |
  Select-Object fixed_session, fixed_series, n_candidates
Get-ChildItem $b -Recurse -Filter metrics.csv | Measure-Object
Get-ChildItem $b -Recurse -Filter metrics.csv | Select-Object -First 1 |
  ForEach-Object { Get-Content $_.FullName -TotalCount 2 }
```

Before running, verify the packages in the existing Windows virtual environment:

```powershell
.\.venv\Scripts\python.exe -c "import SimpleITK, numpy, scipy, PIL, skimage; print('UI packages OK')"
```

If needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\SEP_CODE_ONLY\requirements-overlay.txt
```

## Critical path differences

The Mac project names sessions `*_SEP_Aorta_Spleen` and uses SOMATOM series
names. The audited Windows project uses:

- session suffix: `PCCT_Aorta_Spleen`
- fixed series:
  `USZ_Siemens_Healthineers_NAEOTOM_Alpha_20260805_Q3_Qr40f_120_100_300_1_PolyE_take1`
- batch output: `PhantomX\REGISTRATION\batch_to_20260805_v6`

These are supplied through environment variables in `run_site.ps1`; do not
hard-code Mac paths.

## Fixed take other than take1

The batch runner now accepts an explicit fixed series:

```powershell
.\.venv\Scripts\python.exe .\Kod\registration-radiomics\batch.py `
  ".\PhantomX\LABELED" `
  --fixed-session 20260805_PCCT_Aorta_Spleen `
  --fixed-series "<exact series folder name>" `
  --output ".\PhantomX\REGISTRATION\batch_to_20260805_<take>_v1" `
  --labels 1 52 --workers 2
```

After that, set `SEP_BATCH`, `SEP_FIXED_DAY`, `SEP_FIXED_BASE`, and
`SEP_FIXED_LABEL` to the new output. The current UI numbering assumes the fixed
take is take1; if another take is selected, check the generated take labels and
the featured-card configuration before publishing. Do not delete v6.

## Website generation

The generator writes:

```text
registration_overlay_ui\assets\
registration_overlay_ui\data.js
```

Only these static files are needed for hosting:

```text
index.html
styles.css
app.js
data.js
assets/
.nojekyll
```

Do not publish Python files, `PhantomX`, `DICOM`, batch outputs, or `.venv`.
Publish the new site to a separate GitHub Pages repository; do not overwrite the
existing site until the new output has been checked.

## Notes

- The bundled `data.js`, `all_pairs_report*.xlsx`, and Word reports are Mac-side
  examples for structure/format only. They describe the older 67/66-series Mac
  run, not the audited Windows 88-pair run. Regenerate the site and reports from
  the Windows batch.
- `SEP_FEATURED=0` disables the old Mac-specific featured cards. All 88 cases
  will use generic cards. Re-enable/configure featured cases only after checking
  the new metrics.
- `qc_flags.json` is optional; if absent, QC tags are simply omitted.
- The academic Word/Excel report builders are still specialized; the website
  before/after generation is the portable part of this bundle.
- Do not copy the Mac `.venv`; use the existing Windows `.venv` or create a new
  one.
