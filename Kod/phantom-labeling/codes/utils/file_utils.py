"""File system utilities."""

import os
from pathlib import Path
from typing import List


# Extensions that are definitely not DICOM — skip the magic-byte probe for these.
_NON_DICOM_SUFFIXES = {
    ".txt", ".json", ".xml", ".log", ".csv", ".tsv", ".md",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif",
    ".py", ".pyc", ".yml", ".yaml", ".ini", ".cfg", ".toml",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".html", ".htm",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib",
}


def is_dicom_file(path: str) -> bool:
    """Cheap check for DICOM magic bytes at offset 128."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except (OSError, ValueError):
        return False


def find_dicom_files(directory: str) -> List[str]:
    """Find DICOM files (with or without .dcm extension) in directory.

    Recurses through ``directory``. Files ending in ``.dcm`` are accepted
    without content inspection; extension-less files (common in DICOMDIR
    exports) are accepted after a fast 'DICM' magic byte check. Files with
    obviously non-DICOM extensions (.txt, .json, .png, etc.) are skipped.
    Hidden files and ``DICOMDIR`` index files are always skipped.
    """
    dcm_files = []
    for root, _dirs, files in os.walk(directory, followlinks=False):
        for f in files:
            if f.startswith("."):
                continue
            if f.upper() == "DICOMDIR":
                continue
            lower = f.lower()
            path = os.path.join(root, f)
            if lower.endswith(".dcm"):
                dcm_files.append(path)
                continue
            ext = os.path.splitext(lower)[1]
            if ext in _NON_DICOM_SUFFIXES:
                continue
            if is_dicom_file(path):
                dcm_files.append(path)
    return sorted(dcm_files)


def list_series_dirs(data_root: str) -> List[str]:
    """List all series directories under the data root."""
    if not os.path.isdir(data_root):
        return []
    dirs = []
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full):
            dirs.append(full)
    return dirs


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path
