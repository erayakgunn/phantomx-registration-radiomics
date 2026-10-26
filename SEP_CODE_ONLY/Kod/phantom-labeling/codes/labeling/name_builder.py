"""Standardized filename generator from DICOM metadata.

Converts a :class:`LabelMetadata` into a Windows-safe, single-token name
of the form::

    Merkez_Marka_Cihaz_Tarih_Recon_Conv_KVP_mAs_FOV_SliceThickness

Decimal points become ``p`` (so ``0.67`` -> ``0p67``) to avoid any
ambiguity with file extensions on Windows.

Siemens PCCT (Naeotom Alpha) cohort uses a different downstream naming
scheme — see :func:`compute_recon_type`, :func:`is_siemens_pcct`, and
:func:`build_pcct_base` below. The take_idx component is appended by the
labeler since it requires cross-series ordering by AcquisitionDateTime.
"""

import re
from typing import List

from labeling.header_extractor import LabelMetadata, UNKNOWN


# Windows forbids <>:"/\|?* and control characters in filenames.
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_SPACE_COLLAPSE = re.compile(r"\s+")

# Reserved device names on Windows. Using them as a file/folder name
# makes the filesystem behave unexpectedly (e.g. opening NUL locks).
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize(value: str) -> str:
    """Return a Windows-safe token derived from ``value``.

    Rules applied in order:
      * Empty / None  -> ``UNKNOWN``
      * Collapse runs of whitespace (incl. ``\\t``, ``\\n``) to a single ``_``
      * Strip invalid filename characters (``<>:"/\\|?*`` and ctrl chars)
      * Strip leading/trailing dots, spaces and underscores
      * Replace ``.`` with ``p`` (0.67 -> 0p67)
      * Replace ``-`` with ``neg`` (-5 -> neg5)
      * Prefix reserved device names with ``_`` (NUL -> _NUL)
    """
    if value is None or value == "":
        return UNKNOWN

    # IMPORTANT: collapse whitespace BEFORE stripping control characters,
    # otherwise \t and \n are silently deleted instead of becoming "_".
    v = _SPACE_COLLAPSE.sub("_", str(value))
    v = _INVALID_CHARS.sub("", v).strip("._ ")
    v = v.replace(".", "p").replace("-", "neg")

    if not v:
        return UNKNOWN

    if v.upper() in _WINDOWS_RESERVED:
        v = f"_{v}"

    return v


def build_name(meta: LabelMetadata) -> str:
    """Build the standardized series name from label metadata.

    The resulting string is a single underscore-separated token that is
    safe to use as a folder name on Windows, macOS and Linux.

    NOTE: the ``reconXmm`` reconstruction-spacing token is appended at
    the END of the final folder name (after ``takeN`` for PCCT scans),
    not inside ``build_name`` itself — the labeler concatenates it
    after the take index so the structure stays
    ``base_reconType_takeN_reconXmm``.
    """
    parts = [
        sanitize(meta.institution),
        sanitize(meta.manufacturer),
        sanitize(meta.device_model),
        sanitize(meta.study_date),
        sanitize(meta.recon_kernel),
        sanitize(meta.conv_kernel),
        sanitize(meta.kvp),
        sanitize(meta.mas),
        sanitize(meta.fov),
        sanitize(meta.slice_thickness),
    ]
    return "_".join(parts)


# --- Siemens PCCT (Naeotom Alpha) recon-type & take-idx naming ----------------
#
# These helpers replicate the post-processing logic of rename_pcct.py so the
# labeler can produce the final ``<base>_<ReconType>_take<N>`` name in one
# pass instead of relying on a separate rename step.

_ME_KEV_RE = re.compile(r"ME[_ ]?(\d+)\s*keV", re.IGNORECASE)
_RADIOLOGIE_PREFIX = "Radiologie_"


def format_recon_spacing(value: str) -> str:
    """Format a recon-spacing value (mm) into a filename token like
    ``recon1mm`` or ``recon0p5mm``.

    Accepts a numeric string ("1.0", "0.5", "0,5") or UNKNOWN. Returns
    ``reconUNKNOWNmm`` for empty / UNKNOWN values so the caller never
    has to special-case the missing case. Integers stay integer
    ("1.0" -> "recon1mm"), fractional values get the dot-to-p replace
    that sanitize() applies elsewhere ("0.5" -> "recon0p5mm").
    """
    if value is None or value == "" or value == UNKNOWN:
        return f"recon{UNKNOWN}mm"
    s = str(value).strip().replace(",", ".")
    try:
        v = float(s)
    except (TypeError, ValueError):
        return f"recon{sanitize(s)}mm"
    # Strip trailing ".0" for clean integer display.
    if abs(v - round(v)) < 1e-6:
        token = str(int(round(v)))
    else:
        # Round to 2 decimals, drop trailing zeros, then '.' -> 'p'.
        token = f"{v:.2f}".rstrip("0").rstrip(".")
        token = token.replace(".", "p")
    return f"recon{token}mm"


def compute_recon_type(image_type: List[str], image_comments: str) -> str:
    """Classify the reconstruction type from DICOM ImageType + ImageComments.

    Returns one of:
      * ``"PolyE"``                     — ImageType contains ``THRESHOLD``
      * ``"MonoE<N>keV"``               — ImageType contains ``VMI`` and
                                          ImageComments matches ``ME[_ ]?(\\d+)\\s*keV``
      * ``"VMI_unknown"``               — ``VMI`` present but no keV match
      * ``"OTHER_<image_type[3]>"``     — neither pattern matched (token
                                          taken from ImageType[3] when present)

    Mirrors ``determine_recon_type`` in rename_pcct.py.
    """
    if "VMI" in image_type:
        m = _ME_KEV_RE.search(image_comments or "")
        if m:
            return f"MonoE{m.group(1)}keV"
        return "VMI_unknown"
    if "THRESHOLD" in image_type:
        return "PolyE"
    tail = image_type[3] if len(image_type) > 3 else ""
    return f"OTHER_{tail}"


def is_siemens_pcct(meta: LabelMetadata) -> bool:
    """True iff the series is from a Siemens NAEOTOM (PCCT) scanner."""
    mfr = (meta.manufacturer or "").upper()
    model = (meta.device_model or "").upper()
    return "SIEMENS" in mfr and "NAEOTOM" in model


# --- Siemens SOMATOM Force (Dual-Source, energy-integrating) -------------------
#
# Unlike the NAEOTOM PCCT there are NO spectral reconstructions here: no
# ``VMI`` in ImageType, no ``ME NNkeV`` in ImageComments, no ``THRESHOLD``.
# All images are conventional polychromatic recons, so the MonoE/PolyE slot
# in the standardized name is fixed to ``PolyE``.
#
# The tags that DO differentiate SOMATOM Force series (verified on the
# 20251023 CalibraMics acquisition):
#   * ConvolutionKernel   — Br40d / Br59d (recon kernel; ADMIRE strength is
#                           only in the CSA private header, not exposed)
#   * KVP                 — 80 / 100 / 120 (real tube kV — unlike PCCT where
#                           the physical KVP tag is constant and the VMI kVp
#                           must be parsed out of SeriesDescription)
#   * Exposure            — mAs
#   * ReconstructionDiameter / SliceThickness — FOV & slice thickness
#   * AcquisitionDateTime + IrradiationEventUID + AcquisitionNumber —
#     distinguish REPEATED same-protocol acquisitions (e.g. the two 120 kV
#     runs 23 s apart): same SeriesDescription, different irradiation event.
#     The labeler's existing take-index grouping keys on
#     ``acquisition_datetime`` which maps 1:1 to IrradiationEventUID here,
#     so repeated acquisitions become take1/take2.
#
# InstitutionName on these exports is anonymized ("Ser.Nr.NNNN"), so the
# base name starts with manufacturer + model instead.


def is_siemens_force(meta: LabelMetadata) -> bool:
    """True iff the series is from a Siemens SOMATOM Force (Dual-Source CT)."""
    mfr = (meta.manufacturer or "").upper()
    model = (meta.device_model or "").upper()
    return "SIEMENS" in mfr and "SOMATOM" in model and "FORCE" in model


# Anonymized Siemens service-export institution tags look like
# "Ser.Nr.246663" — carrying them into names is useless noise.
_ANON_INSTITUTION_RE = re.compile(r"^ser\.?\s*nr\.?", re.IGNORECASE)


def build_force_base(meta: LabelMetadata) -> str:
    """Base name for a SOMATOM Force series.

    ``<Institution>_SIEMENS_SOMATOM_Force_<StudyDate>_<ConvKernel>_<kV>_<mAs>_<FOV>_<thick>``

    Mirrors the PCCT base structure (center, scanner, date, kernel, kV,
    mAs, FOV, thickness) so downstream tooling that parses the underscore
    layout keeps working. The recon-type (always ``PolyE`` here), take
    index and ``reconXmm`` tokens are appended by the labeler, exactly
    like PCCT. The institution token is dropped when the tag is missing
    or anonymized ("Ser.Nr.NNNN") and no ``--institution`` override was
    given. There is no Q-level equivalent (ADMIRE strength lives only in
    private tags).
    """
    parts = []
    inst = (meta.institution or "").strip()
    if inst and inst != UNKNOWN and not _ANON_INSTITUTION_RE.match(inst):
        parts.append(sanitize(inst))
    parts += [
        sanitize(meta.manufacturer),
        sanitize(meta.device_model),
        sanitize(meta.study_date),
        sanitize(meta.conv_kernel),
        sanitize(meta.kvp),
        sanitize(meta.mas),
        sanitize(meta.fov),
        sanitize(meta.slice_thickness),
    ]
    return "_".join(parts)


def build_pcct_base(meta: LabelMetadata) -> str:
    """Standard build_name output with the leading ``Radiologie_`` stripped.

    The take_idx and recon_type are NOT appended here — the labeler does
    that after grouping by AcquisitionDateTime across series.
    """
    base = build_name(meta)
    if base.startswith(_RADIOLOGIE_PREFIX):
        base = base[len(_RADIOLOGIE_PREFIX):]
    return base
