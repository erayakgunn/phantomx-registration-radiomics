# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the PhantomXPro radiomics extractor.

Build:
    "C:/Python314/python.exe" -m PyInstaller PhantomXPro_Radiomics.spec --noconfirm

Output:
    dist/PhantomXPro_Radiomics/PhantomXPro_Radiomics.exe   (+ its DLL folder)

One-FOLDER build, not one-file. A one-file build unpacks the whole itk /
SimpleITK / MIRP payload into a temp directory on every launch, which costs
tens of seconds each run and breaks the multiprocessing workers on some
machines. The folder is what gets zipped and handed out.

Naming: this replaces KSW_Radiomics.spec. The old name suggested it only
handled the Philips/KSW acquisition, while the same binary processes the
NAEOTOM (PCCT), Philips (KSW) and SOMATOM (SEP) exports without a flag.
"""

from PyInstaller.utils.hooks import collect_all

# KULLANIM.txt travels INSIDE the distributed folder. The people receiving this
# get a zip, not a repository, so the instructions have to be in the zip.
# PyInstaller 6 puts datas under _internal/, which nobody opening the zip will
# look in, so the build step also copies it next to the .exe (see COLLECT below).
datas = [('KULLANIM.txt', '.')]
binaries = []
# tkinter is listed explicitly: the folder chooser imports it lazily inside a
# function, and the whole point of that dialog is that picking a folder is the
# only thing the user has to do. If it were dropped the .exe would silently fall
# back to asking for a typed path.
hiddenimports = ['openpyxl', 'pydicom', 'tkinter', 'tkinter.filedialog']

# MIRP reaches into all three at import time or during feature computation, and
# each of them loads submodules dynamically, so the static analysis alone misses
# files. skimage in particular is NOT optional here: mirp imports it in
# morph_3d_data.py and dzm_matrix.py, both of which back feature families this
# pipeline reports. matplotlib is likewise imported by mirp's core
# generic_image.py — excluding it to save space breaks the build at runtime
# (the extractor sets MPLBACKEND=Agg so it never looks for a GUI).
for package in ('itk', 'mirp', 'SimpleITK', 'skimage'):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    ['extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # matplotlib and skimage are NOT here: they are real MIRP dependencies (see
    # above) and trimming them is how a build that looks fine ends up raising
    # ImportError on a colleague's machine. tkinter is likewise required.
    #
    # torch IS excluded, and it matters: itk offers optional tensor interop in
    # itk/support/helpers.py, so collect_all('itk') followed the import and
    # dragged the whole deep-learning stack in. Measured on the first build:
    # 5402 MB total, of which torch 4230 MB, cv2 99 MB, torchvision 25 MB — about
    # 4.3 GB of payload that nothing in this pipeline touches, on a binary meant
    # to be copied around a hospital network. The interop is guarded by
    # `_HAVE_TORCH`, set by a try/except at itk import time (helpers.py:34-38),
    # so with torch absent the flag stays False and the branch never runs.
    # Verified by running the frozen .exe end to end on all three acquisitions.
    excludes=[
        'pytest', '_pytest', 'pluggy',
        'torch', 'torchvision', 'torchaudio', 'cv2', 'xarray',
    ],
    noarchive=False,
    optimize=0,
)
# Excluding a package's CODE is not enough — its *-dist-info must go too.
#
# itk does not detect optional interop by importing it. It asks the package
# metadata (itk/support/helpers.py:34-38):
#
#     try:
#         metadata('torch')
#         _HAVE_TORCH = True
#     except PackageNotFoundError:
#         pass
#
# and `accept_array_like_xarray_torch` — a decorator applied at import time in
# itkImageSourcePython.py — then does `import torch` whenever that flag is set.
# PyInstaller's `excludes` drops the torch modules but still collects the
# torch-*.dist-info directory as data, so inside the .exe the metadata lookup
# SUCCEEDS, the flag comes out True, and the import that follows fails. The
# first build with excludes alone produced exactly that: every ROI died with
# "No module named 'torch'" on all three acquisitions while --dry-run passed,
# because the failure only appears once itk's filter machinery is touched.
#
# Dropping the metadata makes the detection do what excluding the package was
# supposed to mean: the flag is False and the decorator never imports anything.
_METADATA_TO_DROP = ('torch', 'torchvision', 'torchaudio', 'xarray',
                     'opencv_python', 'opencv-python')


def _is_excluded_metadata(destination: str) -> bool:
    head = destination.replace('\\', '/').split('/')[0]
    if not head.endswith(('.dist-info', '.egg-info')):
        return False
    name = head.split('-')[0].lower().replace('_', '-')
    return any(name == candidate.lower().replace('_', '-')
               for candidate in _METADATA_TO_DROP)


a.datas = [entry for entry in a.datas if not _is_excluded_metadata(entry[0])]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PhantomXPro_Radiomics',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX is OFF deliberately. Compressing the numpy / itk / SimpleITK native
    # DLLs is a known source of silent load failures and of antivirus false
    # positives, and this binary is going to be copied around a hospital
    # network. Disk is cheaper than a colleague debugging a corrupt DLL.
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PhantomXPro_Radiomics',
)

# Put the instructions where somebody unzipping the folder will actually see
# them: beside the .exe, not buried in _internal/.
import shutil
from pathlib import Path

_dist = Path(DISTPATH) / 'PhantomXPro_Radiomics'
if _dist.is_dir():
    shutil.copyfile('KULLANIM.txt', _dist / 'KULLANIM.txt')
