"""DICOM metadata'sına göre insan-okunur seri adı üretimi.

Şema (kullanıcı isteği):
    <tarih>_<kurum>_<üretici>_<model>_<kernel>_<kVp>_<mAs>_<FOV>_<kesit>
Örnek (EID / SOMATOM Edge Plus):
    20260629_USZ_SIEMENS_SOMATOM_Edge_Plus_Br36f_120_100_300_3

Tüm alanlar YALNIZCA DICOM metadata'sından okunur (klasör/dosya adına bakılmaz):
    tarih   = StudyDate (0008,0020)   [yoksa SeriesDate]
    kurum   = InstitutionName (0008,0080) içinden kısa büyük-harf kod (USZ)
    üretici = Manufacturer (0008,0070)
    model   = ManufacturerModelName (0008,1090)
    kernel  = ConvolutionKernel (0018,1210)  [çok-değerli ise birleştirilir]
    kVp     = KVP etiketi (0018,0060) — HER cihazda (PCCT dahil); SeriesDescription'a bakılmaz
    mAs     = Exposure (0018,1152)
    FOV     = ReconstructionDiameter (0018,1100)  [rekon FOV, mm]
    kesit   = SliceThickness (0018,0050)  [mm]

PCCT (foton-sayan) serilerine ayrıca rekonstrüksiyon eki (native / VMI70keV)
eklenir: aksi halde aynı (kVp,mAs,FOV,kesit,kernel) parametreli native ve VMI
rekonstrüksiyonları AYNI ada düşer ve biri diğerini ezerdi. EID serileri hep
'native' olduğundan EID adları örnekle birebir kalır (rekon eki yok).
"""

from __future__ import annotations

import re
from typing import Any

import pydicom

import config
import sort_spatially as ss


# --------------------------------------------------------------------------- #
# Alan okuyucular (hepsi sağlam / eksik etikete dayanıklı)
# --------------------------------------------------------------------------- #
def _kernel_token(ds: pydicom.Dataset) -> str | None:
    """ConvolutionKernel → tek jeton. Çok-değerli (['Br36f','3']) ise '-' ile
    birleştirilir; böylece 'Br36f' ve 'Br36f 3' FARKLI serileri ayrışır (kayıp yok)."""
    k = getattr(ds, "ConvolutionKernel", None)
    if k is None or k == "":
        return None
    if isinstance(k, (list, tuple, pydicom.multival.MultiValue)):
        parts = [str(x).strip() for x in k if str(x).strip()]
        return "-".join(parts) if parts else None
    return str(k).strip()


def _institution_code(ds: pydicom.Dataset) -> str | None:
    """InstitutionName içinden kısa büyük-harf kod (ör. 'Radiologie USZ' → 'USZ').

    config.INSTITUTION_OVERRIDE ile elle sabitlenebilir. Aksi halde 2-6 harfli
    büyük-harf kısaltma aranır; yoksa tam ad sanitize edilir.
    """
    override = getattr(config, "INSTITUTION_OVERRIDE", None)
    if override:
        return ss._sanitize(override)
    name = str(getattr(ds, "InstitutionName", "") or "").strip()
    if not name:
        return None
    m = re.findall(r"\b[A-Z]{2,6}\b", name)
    if m:
        return m[0]
    return ss._sanitize(name)


def _fmt_number(value: Any) -> str | None:
    """Sayıyı trailing-zero'suz jetona çevir: 3.0→'3', 0.6→'0.6', 300.0→'300', 1.5→'1.5'.

    Sayı değilse sanitize edilmiş string döner (bilgi kaybolmasın)."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ss._sanitize(value)
    if f == int(f):
        return str(int(f))
    return f"{f:g}"


def _recon_token(recon: str) -> str:
    """'VMI_70keV' → 'VMI70keV', 'native' → 'native' (ad içindeki '_' ayırıcısıyla
    karışmaması için iç alt çizgi kaldırılır)."""
    return recon.replace("_", "")


# 'Qr40' (kernel) İÇİNDEKİ Q'yu YAKALAMAZ: Q rakama bitişik ve alnum ile çevrili değil.
# ' Qr40 Q3' → 'Q3' (Qr40'taki Q, 'r' harfi izlediği için eşleşmez).
_QIR_RE = re.compile(r"(?<![A-Za-z0-9])Q(\d)(?![A-Za-z0-9])")


def _qir_level(ds: pydicom.Dataset) -> str | None:
    """PCCT iteratif rekonstrüksiyon (QIR) seviyesi: 'Q1'..'Q4'.

    ÖNEMLİ: Foton-sayan CT'de Q-seviyesi YALNIZCA SeriesDescription'da (0008,103E)
    güvenilir. Private etiketler yanıltıcı: (0029,1011) hep 'QIR3' (sabit),
    (0019,1003) seri numarasını gömer — ikisi de Q-seviyesini YANSITMAZ. Bu yüzden
    tam olarak kVp gibi SeriesDescription (bir DICOM metadata tag'i) esas alınır;
    klasör/dosya adına bakılmaz.
    """
    m = _QIR_RE.search(str(getattr(ds, "SeriesDescription", "") or ""))
    return f"Q{m.group(1)}" if m else None


# --------------------------------------------------------------------------- #
# Alanlar + ad
# --------------------------------------------------------------------------- #
def parse_fields(ds: pydicom.Dataset,
                 kvp: float | None = None,
                 recon: str | None = None,
                 is_pcct: bool | None = None) -> dict[str, Any]:
    """Adlandırma alanlarını DICOM header'ından çıkarır (ham + biçimli).

    kvp/recon/is_pcct verilmezse ds'den (sort_spatially ile) hesaplanır."""
    if is_pcct is None:
        is_pcct = ss.is_photon_counting(ds)
    if kvp is None:
        kvp = ss.effective_kvp(ds)
    if recon is None:
        recon = ss.recon_type(ds)

    date = str(getattr(ds, "StudyDate", "") or getattr(ds, "SeriesDate", "") or "")
    return {
        "date": ss._sanitize(date, "NA"),
        "institution": _institution_code(ds) or "NA",
        "manufacturer": ss._sanitize(getattr(ds, "Manufacturer", None), "NA"),
        "model": ss._sanitize(getattr(ds, "ManufacturerModelName", None), "NA"),
        "kernel": ss._sanitize(_kernel_token(ds) or "NA"),
        "kvp": _fmt_number(kvp) or "NA",
        "mas": _fmt_number(getattr(ds, "Exposure", None)) or "NA",
        "fov": _fmt_number(getattr(ds, "ReconstructionDiameter", None)) or "NA",
        "slice": _fmt_number(getattr(ds, "SliceThickness", None)) or "NA",
        "recon": recon,
        "qir": _qir_level(ds),          # PCCT QIR seviyesi (Q1..Q4) veya None
        "is_pcct": bool(is_pcct),
    }


# Ada giren alanların sırası (şema).
_NAME_ORDER = ("date", "institution", "manufacturer", "model",
               "kernel", "kvp", "mas", "fov", "slice")


def build_name(ds: pydicom.Dataset,
               kvp: float | None = None,
               recon: str | None = None,
               is_pcct: bool | None = None,
               take: int | None = None) -> str:
    """DICOM header'ından insan-okunur seri adı üretir (çakışma çözümü hariç).

    - Temel: tarih_kurum_üretici_model_kernel_kvp_mas_fov_kesit
    - PCCT ise sonuna rekon eki (native/VMI70keV) — VMI/native çakışmasını önler.
    - Test-retest ise sonuna _takeN.
    Nihai benzersizlik için apply_naming.resolve_collisions çağrılır.
    """
    f = parse_fields(ds, kvp=kvp, recon=recon, is_pcct=is_pcct)
    parts = [f[k] for k in _NAME_ORDER]
    if f["is_pcct"]:
        # PCCT rekon ayırıcıları (aksi halde aynı kVp/mAs/FOV/kesit/kernel'li farklı
        # rekonstrüksiyonlar çakışır): önce QIR seviyesi (Q1..Q4), sonra native/VMIkeV.
        if f["qir"]:
            parts.append(f["qir"])                # Q1..Q4 (SeriesDescription'dan)
        parts.append(_recon_token(f["recon"]))    # native / VMI70keV
    name = "_".join(parts)
    if take:
        name += f"_take{take}"
    # Son bir güvenlik: dosya sistemi için sanitize (alt çizgileri koruyarak)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "NA"


# --------------------------------------------------------------------------- #
# Tablo (Excel/CSV) için düzenli seri adı ve ayrıştırılmış parametre sütunları
# --------------------------------------------------------------------------- #
# Şema:
#   <kurum>_<üretici>_<model>_<tarih>_<güç>_<kernel>_<kVp>_<mAs>_<FOV>_<kesit>
#   _<rekon>_take<n>_recon<kesit>mm
# Örnek:
#   USZ_Siemens_Healthineers_NAEOTOM_Alpha_20260714_Q3_Qr40f_120_100_300_1
#   _MonoE70keV_take1_recon1mm
#
# `series_key` (hasta__klasör__uid) izlenebilirlik için gerekli ama tabloda
# okunmuyor: klasör adı ve UID kuyruğu protokolü göstermiyor. Bu ad, satırın
# HANGİ ÇEKİM PARAMETRELERİNDEN geldiğini doğrudan okutur ve Excel'de sıralama
# yapıldığında aynı protokolün varyantları yan yana gelir.

# 'Br40f\3' → kernel 'Br40f', güç '3'  (EID'de kernel gücü çok-değerli tag'te)
# 'Qr40f'   → kernel 'Qr40f', güç None (PCCT'de güç QIR seviyesinde)
def split_kernel(value: Any) -> tuple[str | None, str | None]:
    """ConvolutionKernel → (kernel_tabanı, kernel_gücü)."""
    if value is None or (isinstance(value, float) and value != value):
        return None, None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None, None
    # Ayırıcı kaynağa göre değişiyor: dicom_metadata çok-değerli etiketi '\' ile,
    # _kernel_token '-' ile birleştiriyor. İkisi de aynı ada çözülmeli.
    parts = [p.strip() for p in re.split(r"[\\/,-]", text) if p.strip()]
    if not parts:
        return None, None
    base = parts[0]
    strength = parts[1] if len(parts) > 1 and parts[1].isdigit() else None
    return base, strength


# '120kVp_100mAs_1mm_300FOV 1,00 Qr40 Q3' → '120'
_NOMINAL_KVP_RE = re.compile(r"(\d{2,3})\s*kVp", re.IGNORECASE)


def nominal_kvp(description: Any) -> str | None:
    """SeriesDescription'daki NOMİNAL kVp etiketi (protokol süpürmesinin değişkeni).

    PCCT'de tüp voltajı hep 120 kV'dir; açıklamadaki 80/100/140 farklı PolyE
    rekonstrüksiyonlarını gösterir. Gerçek kVp ayrı sütunda (`kvp`) durur; bu
    yalnızca protokolü adlandırmak için — aksi halde bütün PolyE varyantları
    aynı ada düşer ve tabloda ayırt edilemez.
    """
    m = _NOMINAL_KVP_RE.search(str(description or ""))
    return m.group(1) if m else None


_RECON_LABEL = {"native": "PolyE", "VMI_45keV": "MonoE45keV",
                "VMI_70keV": "MonoE70keV"}


def _text(value: Any, default: str = "NA") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return default
    text = str(value).strip()
    return ss._sanitize(text) if text and text.lower() != "nan" else default


def tidy_fields(row: dict[str, Any], institution_default: str = "NA") -> dict[str, Any]:
    """Birleşik tablo satırından düzenli kimlik/parametre alanları üretir.

    Girdi ham DICOM sütunlarıdır (`dcm_*`); DICOM dosyaları yeniden okunmaz.
    """
    kernel_base, kernel_strength = split_kernel(row.get("dcm_ConvolutionKernel"))
    description = row.get("dcm_SeriesDescription")
    recon = str(row.get("recon_type") or "native")

    # Rekonstrüksiyon gücü: PCCT'de QIR (Q1..Q4, açıklamada), EID'de kernel gücü.
    qir = None
    m = _QIR_RE.search(str(description or ""))
    if m:
        qir = m.group(1)
    strength = qir or kernel_strength

    override = getattr(config, "INSTITUTION_OVERRIDE", None)
    if override:
        institution = ss._sanitize(override)
    else:
        institution = _text(row.get("dcm_InstitutionName"), "")
        if institution:
            codes = re.findall(r"\b[A-Z]{2,6}\b",
                               str(row.get("dcm_InstitutionName")))
            institution = codes[0] if codes else institution
        else:
            # Bazı dışa aktarımlarda (bu projede NAEOTOM) InstitutionName boş.
            # Sırayla: yapılandırılmış yedek → çağıranın verdiği (aynı
            # çalışmanın dolu kayıtlarından türetilmiş) varsayılan.
            institution = (ss._sanitize(getattr(config, "INSTITUTION_FALLBACK", None))
                           if getattr(config, "INSTITUTION_FALLBACK", None)
                           else institution_default)

    thick = _fmt_number(row.get("dcm_SliceThickness")) or "NA"
    return {
        "institution": institution or "NA",
        "manufacturer": _text(row.get("dcm_Manufacturer")),
        "model": _text(row.get("dcm_ManufacturerModelName")),
        "date": _text(row.get("dcm_StudyDate")),
        "strength": f"Q{strength}" if qir else (f"S{strength}" if strength else None),
        "k_strength": strength,
        "recon_k": kernel_base or "NA",
        "kvp_nominal": nominal_kvp(description) or (_fmt_number(row.get("kvp")) or "NA"),
        "kvp": _fmt_number(row.get("dcm_KVP")) or _fmt_number(row.get("kvp")) or "NA",
        "mas": _fmt_number(row.get("dcm_Exposure")) or "NA",
        "fov": _fmt_number(row.get("dcm_ReconstructionDiameter")) or "NA",
        "thick_mm": thick,
        "recon_label": _RECON_LABEL.get(recon, _recon_token(recon)),
        "me_keV": {"VMI_45keV": "45", "VMI_70keV": "70"}.get(recon),
        "take": row.get("take") or 1,
        "series_no": _text(row.get("dcm_SeriesNumber"), ""),
    }


def tidy_fields_from_ds(ds: pydicom.Dataset,
                        kvp: float | None = None,
                        recon: str | None = None,
                        take: int | None = None,
                        institution_default: str = "NA") -> dict[str, Any]:
    """tidy_fields ile AYNI alanları, doğrudan DICOM header'ından üretir.

    Keşif aşamasında (tablo henüz yokken) çıktı klasörünü adlandırmak için.
    Tablo tarafıyla aynı şemayı vermesi şart: aksi halde klasör adı ile
    tablodaki `series` sütunu birbirini tutmaz.
    """
    if recon is None:
        recon = ss.recon_type(ds)
    if kvp is None:
        kvp = ss.effective_kvp(ds)

    row = {
        "dcm_ConvolutionKernel": _kernel_token(ds),
        "dcm_SeriesDescription": getattr(ds, "SeriesDescription", None),
        "dcm_InstitutionName": getattr(ds, "InstitutionName", None),
        "dcm_Manufacturer": getattr(ds, "Manufacturer", None),
        "dcm_ManufacturerModelName": getattr(ds, "ManufacturerModelName", None),
        "dcm_StudyDate": (getattr(ds, "StudyDate", None)
                          or getattr(ds, "SeriesDate", None)),
        "dcm_KVP": getattr(ds, "KVP", None),
        "dcm_Exposure": getattr(ds, "Exposure", None),
        "dcm_ReconstructionDiameter": getattr(ds, "ReconstructionDiameter", None),
        "dcm_SliceThickness": getattr(ds, "SliceThickness", None),
        "dcm_SeriesNumber": getattr(ds, "SeriesNumber", None),
        "recon_type": recon,
        "kvp": kvp,
        "take": take,
    }
    return tidy_fields(row, institution_default)


def build_series_name(f: dict[str, Any]) -> str:
    """tidy_fields çıktısından düzenli seri adı kurar."""
    parts = [f["institution"], f["manufacturer"], f["model"], f["date"]]
    if f.get("strength"):
        parts.append(f["strength"])
    parts += [f["recon_k"], f["kvp_nominal"], f["mas"], f["fov"], f["thick_mm"],
              f["recon_label"], f"take{f['take']}", f"recon{f['thick_mm']}mm"]
    name = "_".join(str(p) for p in parts if p not in (None, ""))
    return re.sub(r"_+", "_", name).strip("_") or "NA"


if __name__ == "__main__":
    import sys
    ds = pydicom.dcmread(sys.argv[1], stop_before_pixels=True, force=True)
    print(build_name(ds))
    for k, v in parse_fields(ds).items():
        print(f"  {k:12s} = {v!r}")
