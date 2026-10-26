"""Detect from DICOM headers whether a series is dual-energy / spectral CT
or a single-energy reconstruction, and whether the file set contains
duplicate-anonymization pairs.

Two independent classifications:

1. Spectral type (from a single header)
   - PolyE / SingleEnergy: standard polychromatic reconstruction
   - MonoE / VMI:          Virtual Monoenergetic Image at a specific keV
   - MaterialMap:          iodine / water / calcium base-material image
   - DECT-RhoZ:            Rho-Z effective-atomic-number map
   - SpectralOther:        ImageType signals spectral but no specific kind
   - Unknown:              no spectral markers found

2. Duplicate-pair status (from inspecting a sample of files)
   - True iff identical SliceLocations appear N > 1 times AND the second
     copy differs only in OriginalAttributesSequence /
     AttributeModificationDateTime (= same scan re-anonymized).
"""
import os
import sys
import hashlib
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pydicom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Spectral tokens that appear in ImageType[2..N] for various
# multi-energy reconstructions across vendors.
_VMI_TOKENS = {"VMI", "MONOENERGETIC", "MONO_E", "MONOE"}
_MATERIAL_TOKENS = {
    "MATERIAL", "BASE", "BASE_MATERIAL", "IODINE", "WATER",
    "CALCIUM", "URIC_ACID", "FAT", "BONE_REMOVAL",
}
_RHOZ_TOKENS = {"RHO_Z", "RHOZ", "Z_EFF", "EFFECTIVE_Z", "PCK"}
_GENERIC_SPECTRAL = {
    "SPECTRAL", "THRESHOLD", "MULTIENERGY", "MULTI_ENERGY",
    "DUAL_ENERGY", "DECT", "ENERGY_SUBTRACTION",
}


def classify_spectral_type(ds: pydicom.Dataset) -> str:
    """Classify the spectral reconstruction type from a single DICOM
    header. Returns a human-readable label."""
    image_type_upper = [str(t).upper() for t in (getattr(ds, "ImageType", []) or [])]
    desc_upper = str(getattr(ds, "SeriesDescription", "") or "").upper()
    comments_upper = str(getattr(ds, "ImageComments", "") or "").upper()

    # 1. Explicit multi-energy flag (newer DICOMs)
    me_flag = getattr(ds, "MultienergyCTAcquisition", None)
    if me_flag in ("YES", "Y", 1, "1", True):
        likely_spectral = True
    else:
        likely_spectral = False

    # 2. Check ImageType tokens
    tokens_present = set(image_type_upper)

    if tokens_present & _VMI_TOKENS:
        return "MonoE / VMI"
    if tokens_present & _MATERIAL_TOKENS:
        return "MaterialMap"
    if tokens_present & _RHOZ_TOKENS:
        return "DECT-RhoZ"
    if tokens_present & _GENERIC_SPECTRAL:
        return "SpectralOther"

    # 3. Fallback: keywords in SeriesDescription / ImageComments
    if "VMI" in desc_upper or "MONO" in desc_upper or "MONOE" in desc_upper:
        return "MonoE / VMI"
    if any(k in comments_upper for k in ("KEV", "ME70", "MONOENERGETIC")):
        return "MonoE / VMI"
    if "SPECTRAL" in desc_upper:
        return "SpectralOther"

    # 4. Explicit single-energy
    if likely_spectral:
        return "SpectralOther"
    return "SingleEnergy (PolyE)"


def detect_duplicate_pairs(
    series_dir: str, sample_n: int = 60
) -> Tuple[bool, Dict[str, int], List[str]]:
    """Read a sample of files and return:
      * has_duplicates: True iff any SliceLocation appears >1 time AND
        the duplicate's pixel data is identical.
      * multiplicity_histogram: dict of {N: count of unique-z with N
        files at that z}.
      * sample_diff_tags: list of DICOM tag keywords that differ
        between paired files (excluding OriginalAttributesSequence
        and modification timestamps).
    """
    files = sorted(glob.glob(os.path.join(series_dir, "**", "*.dcm"), recursive=True))
    if not files:
        return False, {}, []
    sampled = files[:sample_n]
    by_z = defaultdict(list)
    for f in sampled:
        try:
            ds = pydicom.dcmread(f)
            sl = getattr(ds, "SliceLocation", None)
            if sl is not None:
                by_z[round(float(sl), 4)].append((f, ds))
        except Exception:
            continue
    multiplicity = defaultdict(int)
    for z, files_at_z in by_z.items():
        multiplicity[len(files_at_z)] += 1
    has_dup = False
    diff_tags: List[str] = []
    for z, files_at_z in by_z.items():
        if len(files_at_z) < 2:
            continue
        a, b = files_at_z[0][1], files_at_z[1][1]
        # Check pixel data identity
        try:
            h_a = hashlib.md5(a.PixelData).hexdigest()
            h_b = hashlib.md5(b.PixelData).hexdigest()
            pixel_same = (h_a == h_b)
        except Exception:
            pixel_same = False
        # Diff tags (excluding metadata-modification tags)
        a_keys = {elem.tag: elem for elem in a.iterall()}
        b_keys = {elem.tag: elem for elem in b.iterall()}
        for tag in set(a_keys) | set(b_keys):
            ea = a_keys.get(tag)
            eb = b_keys.get(tag)
            va = str(ea.value) if ea else None
            vb = str(eb.value) if eb else None
            if va == vb:
                continue
            name = (ea.keyword if ea else eb.keyword) or str(tag)
            if name in ("OriginalAttributesSequence",
                        "AttributeModificationDateTime",
                        "ModifyingSystem", "ReasonForTheAttributeModification"):
                continue
            diff_tags.append(name)
        if pixel_same:
            has_dup = True
        break  # one pair is enough for the diff report
    return has_dup, dict(multiplicity), sorted(set(diff_tags))


def main():
    if len(sys.argv) < 2:
        print("Usage: spectral_detector.py SERIES_DIR [SERIES_DIR ...]")
        sys.exit(1)
    for series_dir in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(series_dir, "**", "*.dcm"), recursive=True))
        if not files:
            print(f"\n{series_dir}: NO DICOM files")
            continue
        print(f"\n=== {os.path.basename(series_dir) or series_dir} ({len(files)} files) ===")
        ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)
        print(f"  Manufacturer:        {getattr(ds0, 'Manufacturer', '?')}")
        print(f"  ManufacturerModel:   {getattr(ds0, 'ManufacturerModelName', '?')}")
        print(f"  SeriesDescription:   {getattr(ds0, 'SeriesDescription', '?')}")
        print(f"  ImageType:           {list(getattr(ds0, 'ImageType', []))}")
        print(f"  ImageComments:       {getattr(ds0, 'ImageComments', '?')}")
        print(f"  MultienergyCTAcquisition: {getattr(ds0, 'MultienergyCTAcquisition', '?')}")
        spectral = classify_spectral_type(ds0)
        print(f"  Spectral type:       {spectral}")
        has_dup, mult, diff_tags = detect_duplicate_pairs(series_dir, sample_n=60)
        print(f"  Slice multiplicity:  {mult}")
        print(f"  Duplicate-pair scan: {has_dup}")
        if has_dup:
            non_meta_diffs = [t for t in diff_tags
                              if t not in ("OriginalAttributesSequence",
                                            "AttributeModificationDateTime")]
            if not non_meta_diffs:
                print("    -> Duplicate pairs differ ONLY in OriginalAttributesSequence + "
                      "AttributeModificationDateTime.")
                print("    -> This is RE-ANONYMIZATION duplication, NOT dual-energy.")
                print("    -> Recommend: deduplicate input files before analysis.")
            else:
                print(f"    -> Duplicate pairs differ in: {non_meta_diffs}")
                print("    -> Investigate before trusting either copy.")


if __name__ == "__main__":
    main()
