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
