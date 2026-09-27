# Registration-Radiomics Pilot

This module currently runs one explicit fixed/moving PhantomX pair. It writes a
rigid transform, a registered CT, a nearest-neighbour registered multi-label
mask, and pre/post geometric metrics. Existing data and radiomics outputs are
not modified.

The first pilot should use `20260805 take1` as fixed and `20260805 take2` as
moving. The exact paths can be copied from the existing `PhantomX/LABELED`
tree. Labels are `1=spleen` and `52=aorta`.

To start the desktop monitor:

```powershell
..\.venv\Scripts\python.exe gui.py
```

The GUI runs the pilot in a background process, streams its log, and keeps all
outputs in the selected output folder.

In the GUI, select the session folder containing `nifti` and `segmentation`,
for example `PhantomX\\LABELED\\20260805_PCCT_Aorta_Spleen`, then click
`Auto-fill take1/take2`. The CT and mask paths are derived from that folder.

The pilot is resumable. It stores `pilot_state.json`, `metrics.partial.csv` and
the registered artifacts in the output folder. Re-running with the same inputs
continues from the first missing stage and refuses a second simultaneous run in
the same output folder.

For each pair, the pilot first compares masks in common DICOM world space. If
the maximum selected-organ centroid displacement is at most 5 mm, it keeps the
identity/world-coordinate result and skips rigid registration. Larger
displacements use deterministic correlation-based rigid registration,
initialized from vertebra/rib/sternum labels at 2 mm registration resolution;
the final CT and masks remain on the original fixed grid.

Registration is accepted only when the independent stable-structure mask
(vertebrae, ribs and sternum) has a non-decreasing Dice score. The decision is
recorded in `registration_qc.json`; target organ Dice values are not used to
select a transform.

For a sequential batch run, use `batch.py`. It processes one pair at a time and
reuses completed pair folders:

```powershell
..\.venv\Scripts\python.exe batch.py ..\..\PhantomX\LABELED `
  --fixed-session 20260805_PCCT_Aorta_Spleen `
  --output ..\..\PhantomX\REGISTRATION\batch_to_20260805 `
  --limit 2
```
