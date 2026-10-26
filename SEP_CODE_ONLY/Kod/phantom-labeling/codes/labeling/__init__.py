"""DICOM automatic labeling module.

Reads DICOM header metadata and generates standardized folder names
in the format:
    {Merkez}_{Marka}_{Cihaz}_{Tarih}_{ReconKernel}_{ConvKernel}_{KVP}_{mAs}_{FOV}_{SliceThickness}
"""

from labeling.header_extractor import LabelMetadata, extract_from_file, UNKNOWN
from labeling.name_builder import build_name, sanitize
from labeling.labeler import LabelResult, label_folder

__all__ = [
    "LabelMetadata",
    "LabelResult",
    "extract_from_file",
    "build_name",
    "sanitize",
    "label_folder",
    "UNKNOWN",
]
