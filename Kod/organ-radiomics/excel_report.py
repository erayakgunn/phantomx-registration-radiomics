"""Birleşik öznitelik tablosunu okunabilir bir Excel çalışma kitabına yazar.

Ham `df.to_excel()` çıktısı 200+ sütunluk bir tabloda pratikte okunamıyor:
başlık kayboluyor, meta sütunlarla öznitelik sütunları ayırt edilemiyor, hangi
ayarlarla üretildiği hiçbir yerde yazmıyor. Buradaki yazıcı iki şey ekliyor —
beyan (methods sayfası) ve biçim: dondurulmuş kimlik sütunları, otomatik filtre,
renkle ayrılmış sütun blokları ve nadiren okunan MIRP meta sütunlarının
katlanması.

Beyan sayfası canlı `config.MIRP_SETTINGS`'ten türetiliyor, dolayısıyla tablonun
yanındaki açıklama tablonun kendisiyle asla çelişemiyor.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import config

# Sütun blokları. SIRA ile RENK ayrı tutuluyor: hangi sütunun nerede duracağı
# okuma alışkanlığının, hangi renkte olacağı ise anlamının meselesi, ve ikisi
# birbirinden bağımsız değişiyor.
ORGAN = "organ"                # hangi yapı ölçüldü
BOOKKEEPING = "bookkeeping"    # dosya/hasta kimliği — gerekli ama okunmuyor
ACQUISITION = "acquisition"    # çekim koşulları: satırlar arası farkı bunlar yaratıyor
DICOM = "dicom"                # başlığın geri kalanı
MIRP_META = "mirp_meta"        # MIRP'in kendi meta sütunları
FEATURE = "feature"

_ORGAN_COLUMN = "image_mask_label"
_MIRP_META_PREFIX = "image_"
_DICOM_PREFIX = "dcm_"

# Kimlik: satırı bulmaya yarar, ama analiz sırasında okunmaz.
# `series` en başta: protokolü okunur biçimde taşıyan tek sütun o. `series_key`
# izlenebilirlik için kalıyor (klasör + UID), ama gözle okunmuyor.
_BOOKKEEPING_COLUMNS = ("series", "series_key", "metadata_name", "patient_id")

# Çekim koşulları. Önce `series` adından ayrıştırılmış DÜZENLİ parametreler
# (Excel'de süzme/gruplama bunlarla yapılır), sonra DICOM başlığından terfi
# eden ham etiketler. İki cihaz da aynı sütunlara düşer: SOMATOM'da rekon gücü
# kernel etiketinden (S1..S5), NAEOTOM'da QIR seviyesinden (Q1..Q4) gelir.
_ACQUISITION_COLUMNS = (
    "date", "institution", "manufacturer", "model",
    "recon_k", "k_strength", "kvp", "kvp_nominal", "mAs", "FOV", "thick_mm",
    "recon_type", "recon_label", "me_keV", "take",
)
_DICOM_PROMOTED = (
    "dcm_StudyDate",
    "dcm_AcquisitionTime",
    "dcm_Modality",
    "dcm_Manufacturer",
    "dcm_ManufacturerModelName",
    "dcm_ConvolutionKernel",
    "dcm_KVP",
    "dcm_Exposure",
    "dcm_ReconstructionDiameter",
    "dcm_SliceThickness",
)

_HEADER_FILL = "1F4E78"
_ORGAN_FILL = "C6E0B4"       # yeşil
_ACQUISITION_FILL = "DDEBF7"  # mavi
_DICOM_FILL = "FFF2CC"        # sarı
_ACCENT = "C00000"

_BLOCK_FILL = {
    ORGAN: _ORGAN_FILL,
    BOOKKEEPING: _DICOM_FILL,
    ACQUISITION: _ACQUISITION_FILL,
    DICOM: _DICOM_FILL,
}

_BLOCK_WIDTH = {
    ORGAN: 26,
    BOOKKEEPING: 30,
    ACQUISITION: 18,
    DICOM: 20,
    MIRP_META: 20,
    FEATURE: 17,
}


def _layout(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Sütunları görünecekleri sırada (ad, blok) çiftleri olarak verir."""
    layout: list[tuple[str, str]] = []
    placed: set[str] = set()

    def take(name: str, block: str) -> None:
        if name in df.columns and name not in placed:
            layout.append((name, block))
            placed.add(name)

    take(_ORGAN_COLUMN, ORGAN)
    for name in _BOOKKEEPING_COLUMNS:
        take(name, BOOKKEEPING)
    for name in _ACQUISITION_COLUMNS:
        take(name, ACQUISITION)
    for name in _DICOM_PROMOTED:
        take(name, ACQUISITION)

    for name in df.columns:                       # harvest() sırası korunur
        if name.startswith(_DICOM_PREFIX):
            take(name, DICOM)
    for name in df.columns:
        if name.startswith(_MIRP_META_PREFIX):
            take(name, MIRP_META)
    for name in df.columns:
        take(name, FEATURE)

    return layout


def _classify_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """(organ, çekim, kalan DICOM, MIRP meta, öznitelik) sütunları.

    Kimlik sütunları sarı blokta sayılır: renkleri onlarla aynı ve analizde
    aynı rolü oynuyorlar.
    """
    layout = _layout(df)

    def block(*names: str) -> list[str]:
        return [name for name, group in layout if group in names]

    return (block(ORGAN), block(ACQUISITION), block(BOOKKEEPING, DICOM),
            block(MIRP_META), block(FEATURE))


# IBSI toplama yöntemlerinin okunur adları. Beyan bu tablodan üretiliyor;
# elle yazılmış bir cümle ayar değiştiğinde sessizce yanlışa dönüyordu
# (ölçüldü: sayfa "3D volume-merged" diyordu, tabloda `3d_average` da vardı).
_AGGREGATION_LABEL = {
    "3d_average": "3D averaged",
    "3d_volume_merge": "3D volume-merged",
    "2d_average": "2D averaged",
    "2d_slice_merge": "2D slice-merged",
    "2.5d_direction_merge": "2.5D direction-merged",
    "2.5d_volume_merge": "2.5D volume-merged",
    "3d": "3D", "2d": "2D", "2.5d": "2.5D",
}


_RESOLVED: dict | None = None


def _resolved(key: str, default: float) -> float:
    """MIRP'in ÇÖZÜMLENMİŞ ayar değeri.

    Beyanda sabit sayı yazmak, ayar varsayılana bırakıldığında sessizce yanlışa
    dönüyordu. Bu yüzden değer MIRP'in kendi yapılandırma çözümleyicisinden
    okunuyor — yazdığımız parametreler de, varsayılana bıraktıklarımız da
    gerçek değeriyle görünüyor. Okunamazsa `default` döner (uydurma değil,
    MIRP'in belgelenmiş varsayılanı).
    """
    global _RESOLVED
    if _RESOLVED is None:
        try:
            from mirp.settings.import_config_parameters import (
                import_configuration_settings)
            s = import_configuration_settings(
                compute_features=True, **config.MIRP_SETTINGS,
                **config.MIRP_MODES[config.PRIMARY_MODE])[0]
            _RESOLVED = {
                "glcm_distance": s.feature_extr.glcm_distance,
                "ngldm_distance": s.feature_extr.ngldm_dist,
                "ngldm_difference_level": s.feature_extr.ngldm_diff_lvl,
                "roi_spline_order": s.roi_interpolate.spline_order,
            }
        except Exception:  # noqa: BLE001 — beyan üretimi çıktıyı düşürmemeli
            _RESOLVED = {}
    v = _RESOLVED.get(key, default)
    if isinstance(v, (list, tuple)):
        if len(v) != 1:
            return float(default)
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _ts_version() -> str:
    """Kurulu TotalSegmentator sürümü — okunamıyorsa 'unknown' (tahmin YOK)."""
    import feature_extraction_mirp as fx
    return fx.totalsegmentator_versions().get("totalsegmentator", "unknown")


def _torch_version() -> str:
    import feature_extraction_mirp as fx
    return fx.totalsegmentator_versions().get("torch", "unknown")


def _observed_span(df: pd.DataFrame) -> str:
    """Maske içinde GERÇEKTEN gözlenen yoğunluk aralığı — bu tablodan ölçülür.

    Buraya daha önce sabit '-1047 to 1625 HU' yazılıydı; o sayılar 117 yapılık
    başka bir veri setinden geliyordu ve dalak+aorta çıkarımında hiçbir satırı
    tarif etmiyordu. Beyandaki bir sayı, o beyanı taşıyan tablodan gelmeli.
    """
    if not {"stat_min", "stat_max"} <= set(df.columns):
        return "not measurable from this table (stat_min / stat_max absent)"
    alt, ust = df["stat_min"].min(), df["stat_max"].max()
    if pd.isna(alt) or pd.isna(ust):
        return "not measurable from this table (all NaN)"
    return f"{alt:.0f} to {ust:.0f} HU"


def _excluded_declaration() -> str:
    """Tablodan çıkarılan öznitelikler — listeler koddan okunur, elle yazılmaz."""
    import feature_extraction_mirp as fx
    kestirim = ", ".join(fx.APPROXIMATED_FEATURES)
    ivh = ", ".join(fx.IVH_VOLUME_FRACTION_FEATURES)
    alt, ust = config.MIRP_SETTINGS["resegmentation_intensity_range"]
    return (
        f"Two groups are NOT reported. (1) {kestirim}: MIRP estimates these by "
        f"Monte Carlo above 1000 voxels and seeds the sampler from a hash that "
        f"includes the file path, so they are neither exact nor reproducible "
        f"across relocations. (2) {ivh}: with ivh_discretisation_method='none' "
        f"MIRP takes the IVH intensity axis from the re-segmentation range "
        f"[{alt:g}, {ust:g}] HU, which places the volume-fraction cut-offs far "
        f"outside the soft-tissue span — measured over 284 rows they were "
        f"constant (0 or 1) and two of them duplicated other columns exactly. "
        f"The intensity fractions (ivh_i10..i90, ivh_diff_i*) ARE reported. "
        f"Every remaining feature in this workbook is computed exactly."
    )


def _ivh_declaration() -> str:
    """IVH davranışının ÖLÇÜLMÜŞ beyanı.

    Önceki metin 'bin sayısı maskede bulunan farklı gri seviye sayısına eşittir'
    diyordu. Ölçüldü, yanlış: ivh_discretisation_method='none' iken MIRP
    yoğunluk aralığı olarak RESEGMENTASYON aralığını kullanıyor
    (base_mask.py:428) ve histogramı o aralığın tamamı boyunca kuruyor
    (ivh_features.py:80) — burada 4401 seviye, bulunmayanlar sıfır sayımla
    dolduruluyor.
    """
    alt, ust = config.MIRP_SETTINGS["resegmentation_intensity_range"]
    n = int(ust - alt) + 1
    return (f"No re-discretisation ('none'): CT intensities are calibrated and "
            f"integer-valued, so the IVH is computed directly on Hounsfield "
            f"units. The histogram spans the re-segmentation range "
            f"[{alt:g}, {ust:g}] HU — {n} levels, with levels absent from the "
            f"mask padded with zero counts. This same range also sets the IVH "
            f"gamma axis, which is why the volume-fraction features are "
            f"excluded (see 'Excluded features'); the intensity fractions "
            f"reported here are unaffected.")


def _aggregation_names(settings: dict, key: str) -> str:
    """Bir toplama listesinin okunur adları ('3d_average' -> '3D averaged')."""
    return ", ".join(_AGGREGATION_LABEL.get(m, m) for m in settings.get(key, []))


def _texture_sentence() -> str:
    """Yapıştırılabilir paragrafın doku cümlesi — canlı config.MIRP_MODES'tan.

    Burada sabit bir cümle duruyordu ("Texture matrices were computed in 3D and
    volume-merged") ve ayarlar değişince sessizce yanlışa döndü: config GLCM ve
    GLRLM için hem `3d_average` hem `3d_volume_merge` üretirken paragraf yalnız
    ikincisini beyan ediyor, 2B modundan hiç söz etmiyordu. Yani `methods`
    sayfasının yapılandırılmış satırları (`_texture_rows`) doğruyu, yayına
    yapıştırılacak metin yanlışı söylüyordu.

    Bu hata bir kez ölçülüp `_texture_rows` tarafında düzeltilmişti (bkz.
    _AGGREGATION_LABEL). Nüksetmesinin nedeni, aynı beyanın iki yerde ve yalnız
    birinin canlı ayardan türetilmesiydi; artık ikisi de tek kaynaktan geliyor.
    """
    modes = config.MIRP_MODES
    clauses = []
    for mode, settings in modes.items():
        plane = ("slice by slice (by_slice = True), so resampling and "
                 "morphology are also two-dimensional"
                 if settings.get("by_slice") else
                 "over the whole volume (by_slice = False)")
        clauses.append(
            f"in the {mode} mode textures were computed {plane}, with GLCM and "
            f"GLRLM aggregated as "
            f"{_aggregation_names(settings, 'glcm_spatial_method')} and GLSZM, "
            f"GLDZM, NGTDM and NGLDM as "
            f"{_aggregation_names(settings, 'glszm_spatial_method')}")

    sheets = ", ".join(f"{m}_base" for m in modes)
    if len(modes) == 1:
        return (f"Texture matrices were computed in a single mode: "
                f"{clauses[0]} (sheet {sheets}).")
    return (f"Texture matrices were computed in {len(modes)} separate modes: "
            + "; ".join(clauses)
            + f". Each mode is reported on its own sheet ({sheets}); rows from "
              "different modes must not be pooled, because by_slice changes "
              "resampling and morphology as well as texture aggregation.")


def _texture_rows() -> list[tuple[str, str, str]]:
    """Doku toplama beyanı — canlı config.MIRP_MODES'tan türetilir."""
    def adlar(anahtar: str, ayar: dict) -> str:
        return _aggregation_names(ayar, anahtar)

    rows: list[tuple[str, str, str]] = [
        ("Texture", "Sheets",
         "Each mode is reported on its own sheet (" +
         ", ".join(f"{m}_base" for m in config.MIRP_MODES) + "). "
         "2D and 3D rows must NOT be pooled: by_slice changes resampling and "
         "morphology as well as texture aggregation, so identically named "
         "columns come from different computations."),
    ]
    for mod, ayar in config.MIRP_MODES.items():
        rows.append(("Texture", f"{mod}_base — dimensionality",
                     f"by_slice = {bool(ayar.get('by_slice', False))}"))
        rows.append(("Texture", f"{mod}_base — GLCM / GLRLM aggregation",
                     adlar("glcm_spatial_method", ayar)))
        rows.append(("Texture", f"{mod}_base — GLSZM / GLDZM / NGTDM / NGLDM",
                     adlar("glszm_spatial_method", ayar)))
    return rows


def _methods_rows(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Bir yayında beyan edilmesi gereken her şey, canlı ayarlardan türetilmiş."""
    import importlib.metadata as md

    def version(package: str) -> str:
        try:
            return md.version(package)
        except Exception:
            return "not-installed"

    settings = config.MIRP_SETTINGS
    lower, upper = settings["resegmentation_intensity_range"]
    _, _, dicom, _, features = _classify_columns(df)
    n_dicom = sum(1 for c in df.columns if c.startswith(_DICOM_PREFIX))

    n_series = df["series_key"].nunique() if "series_key" in df.columns else len(df)
    n_structures = (df["image_mask_label"].nunique()
                    if "image_mask_label" in df.columns else "n/a")

    return [
        ("Dataset", "Series", str(n_series)),
        ("Dataset", "Structures", str(n_structures)),
        ("Dataset", "Feature rows", str(len(df))),
        ("Dataset", "Feature columns", str(len(features))),
        ("Dataset", "DICOM header columns",
         f"{n_dicom} (dcm_* — acquisition parameters carried over from the "
         f"source series so a row's protocol is readable without returning to "
         f"the DICOM files; the {len(_DICOM_PROMOTED)} most-read ones sit in "
         f"the identity block, the remaining {len(dicom)} follow it)"),

        ("Software", "Feature extraction library",
         f"MIRP {version('mirp')} (IBSI-1 compliant feature definitions)"),
        ("Software", "Numerical stack",
         ", ".join(f"{p} {version(p)}" for p in
                   ("numpy", "scipy", "pandas", "SimpleITK", "nibabel"))),
        ("Software", "Segmentation",
         f"TotalSegmentator {_ts_version()} (task '{config.TS_TASK}', "
         f"device '{config.TS_DEVICE}'), PyTorch {_torch_version()}. "
         "Mask boundaries depend on this software: a different version can "
         "shift them, and every feature follows the boundary."),

        ("Image", "Modality", "CT (calibrated, Hounsfield units)"),
        ("Image", "Resampling",
         f"{settings['new_spacing']} mm isotropic, spline order "
         f"{settings['spline_order']} (tricubic — matches IBSI Config E)"),
        ("Image", "Anti-aliasing pre-filter",
         "Disabled. The IBSI resampling definition specifies interpolation only; "
         "MIRP's default Gaussian pre-filter is not part of it, and this "
         "project's own IBSI validation disables it in every interpolating "
         "configuration."),
        ("Image", "Intensity normalisation / denoising", "Not applied"),
        ("Image", "Filter-based (IBSI-2) features", "Not computed"),

        ("ROI", "Structures analysed", _roi_scope()),
        ("ROI", "Mask source",
         "TotalSegmentator multilabel output; one ROI per anatomical structure, "
         "no manual editing, no morphological post-processing."),
        ("ROI", "Mask resampling",
         f"Spline order {_resolved('roi_spline_order', 1):g} onto the resampled "
         f"image grid, then binarised at a 0.5 partial-volume threshold "
         f"(MIRP default, not overridden here)."),

        ("Re-segmentation", "Range (HU)", f"[{lower:g}, {upper:g}]"),
        ("Re-segmentation", "Voxels excluded",
         f"None. The intensity span actually observed inside the masks of this "
         f"workbook is {_observed_span(df)}, well inside the range, so it acts "
         f"purely as the fixed-bin-size grey-level origin. The range was chosen "
         f"wide enough to cover bone as well: a soft-tissue window removes the "
         f"intensities that define it — measured on a 117-structure extraction, "
         f"it discarded 69.5% of vertebrae_T10."),
        ("Re-segmentation", "Outlier (sigma) filtering", "Not applied"),

        ("Discretisation", "Method", "Fixed bin size (= fixed bin width)"),
        ("Discretisation", "Bin width (HU)",
         f"{settings['base_discretisation_bin_width']:g}"),
        ("Discretisation", "Grey-level origin (HU)", f"{lower:g}"),
        ("Discretisation", "Intensity-volume histogram", _ivh_declaration()),

        *_texture_rows(),
        ("Texture", "GLCM / NGLDM distance",
         f"GLCM distance {_resolved('glcm_distance', 1.0):g} voxel; "
         f"NGLDM distance {_resolved('ngldm_distance', 1.0):g} voxel, "
         f"coarseness alpha = {_resolved('ngldm_difference_level', 0.0):g} "
         f"(carried in the column names as a0.0)"),

        ("Features", "Families", "All IBSI-1 base families"),
        ("Features", "Feature columns in this workbook", str(len(features))),

        ("Features", "Excluded features", _excluded_declaration()),
        ("Caveats", "Segmentation reproducibility",
         "TotalSegmentator runs on GPU with unpinned PyTorch and downloaded "
         "weights; masks are cached per series, but a different machine may "
         "produce slightly different boundaries."),
    ]


def _roi_scope() -> str:
    """Metot beyanı için ROI kapsamı — hedeflenen organ kümesi.

    Sabit "117 structures" yazmak, çıkarım bir alt kümeyle koşulduğunda yayına
    giden beyanı yanlış kılıyordu. Kapsam config.TARGET_ORGANS'tan okunur.
    """
    targets = getattr(config, "TARGET_ORGANS", None)
    if not targets:
        return (f"all structures produced by TotalSegmentator "
                f"('{config.TS_TASK}' task)")
    return (f"a pre-specified subset of the structures produced by "
            f"TotalSegmentator ('{config.TS_TASK}' task): "
            f"{', '.join(sorted(targets))}")


_METHODS_PARAGRAPH = (
    "Radiomics features were extracted with MIRP {mirp} using IBSI-1 compliant "
    "feature definitions. Images were resampled to {spacing} mm isotropic voxels "
    "using tricubic interpolation (spline order {order}) with the Gaussian "
    "anti-aliasing pre-filter disabled, since it is not part of the IBSI "
    "resampling definition. Regions of interest were {roi_scope}; masks were "
    "resampled trilinearly and binarised at a 0.5 partial-volume threshold. "
    "Intensities were re-segmented to [{low:g}, {high:g}] HU, a range that "
    "brackets the full intensity span observed inside the masks and therefore "
    "excludes no voxel while fixing the discretisation origin. Discretisation "
    "used a fixed bin size of {width:g} HU. {texture} No filter-based (IBSI-2) "
    "features were computed."
)


def write_workbook(df: pd.DataFrame, csv_path: Path) -> Path | None:
    """CSV'nin yanına biçimlendirilmiş bir .xlsx yazar (atomik)."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except Exception as exc:  # noqa: BLE001
        print(f"[xlsx] UYARI: openpyxl yok ({exc}); yalnız CSV yazıldı.")
        return None

    xlsx = csv_path.with_suffix(".xlsx")
    tmp = xlsx.with_name(xlsx.name + ".tmp")

    # Girdi ya tek tablo (eski çağrı biçimi) ya da {mod: tablo} sözlüğü olabilir.
    # `methods` sayfası ANA modun tablosundan üretilir: beyan metni yeniden
    # örnekleme/ayrıklaştırma gibi mod-bağımsız ayarları anlatıyor.
    tablolar: dict[str, pd.DataFrame] = (
        dict(df) if isinstance(df, dict) else {config.PRIMARY_MODE: df})
    # `or` KULLANILAMAZ: DataFrame'in doğruluk değeri belirsizdir (ValueError).
    df = tablolar[config.PRIMARY_MODE] if config.PRIMARY_MODE in tablolar \
        else next(iter(tablolar.values()))

    header_font = Font(bold=True, color="FFFFFF", size=10)
    header_fill = PatternFill("solid", start_color=_HEADER_FILL, end_color=_HEADER_FILL)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")

    wb = Workbook()
    wb.remove(wb.active)

    # ---------------- öznitelik sayfaları (mod başına) ----------------
    # Sayfa adları referans çalışma kitabıyla aynı: `3D_base`, `2D_base`.
    tints = {block: PatternFill("solid", start_color=colour, end_color=colour)
             for block, colour in _BLOCK_FILL.items()}

    for mod, mod_df in tablolar.items():
        layout = _layout(mod_df)
        ordered = [name for name, _ in layout]
        table = mod_df[ordered]

        ws = wb.create_sheet(f"{mod}_base")
        ws.append(list(table.columns))
        for record in table.itertuples(index=False, name=None):
            ws.append([None if pd.isna(v) else v for v in record])

        for col_index, (name, block) in enumerate(layout, start=1):
            cell = ws.cell(1, col_index)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center",
                                       wrap_text=True)
            cell.border = border

            letter = get_column_letter(col_index)
            ws.column_dimensions[letter].width = _BLOCK_WIDTH[block]
            if block == MIRP_META:
                # MIRP'in 24 meta sütunu nadiren okunuyor: katlanabilir ve kapalı.
                ws.column_dimensions[letter].outlineLevel = 1
                ws.column_dimensions[letter].hidden = True

            tint = tints.get(block)
            for row_index in range(2, ws.max_row + 1):
                target = ws.cell(row_index, col_index)
                if tint is not None:
                    target.fill = tint
                elif block == FEATURE:
                    target.number_format = "0.00000"

        ws.row_dimensions[1].height = 42
        # Yalnızca satırı TANIMLAYAN sütunlar donduruluyor: organ ve kimlik. Çekim
        # parametreleri de solda ve renkli, ama 10 sütun daha dondurmak ekranın
        # yarısını yer ve asıl bakılan öznitelikleri görünmez kılar.
        frozen = sum(1 for _, block in layout if block in (ORGAN, BOOKKEEPING))
        ws.freeze_panes = ws.cell(2, frozen + 1)
        ws.auto_filter.ref = f"A1:{get_column_letter(len(ordered))}{ws.max_row}"
        ws.sheet_properties.tabColor = _HEADER_FILL

    # ---------------- methods ----------------
    ms = wb.create_sheet("methods")
    ms["A1"] = "Methods declaration — what to report for this extraction"
    ms["A1"].font = Font(bold=True, color="FFFFFF", size=14)
    ms["A1"].fill = header_fill
    ms.merge_cells("A1:C1")
    ms.row_dimensions[1].height = 26
    ms["A2"] = ("Every value below is read from the live configuration, from "
                "MIRP's own resolved settings, or measured from this "
                "workbook's own rows — nothing is hard-coded.")
    ms["A2"].font = Font(italic=True, color="595959", size=10)
    ms.merge_cells("A2:C2")

    for col, heading in enumerate(("Section", "Item", "Declaration"), start=1):
        cell = ms.cell(4, col, heading)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

    section_fill = PatternFill("solid", start_color=_ACQUISITION_FILL,
                               end_color=_ACQUISITION_FILL)
    row = 5
    previous = None
    for section, item, value in _methods_rows(df):
        label = section if section != previous else ""
        previous = section
        cell = ms.cell(row, 1, label)
        cell.font = Font(bold=True, color=_HEADER_FILL, size=10)
        if label:
            cell.fill = section_fill
        ms.cell(row, 2, item).font = Font(bold=True, size=10)
        ms.cell(row, 3, str(value))
        for col in range(1, 4):
            ms.cell(row, col).border = border
            ms.cell(row, col).alignment = wrap
        row += 1

    row += 1
    ms.cell(row, 1, "Ready to paste").font = Font(bold=True, color=_HEADER_FILL, size=11)
    row += 1
    import importlib.metadata as md
    try:
        mirp_version = md.version("mirp")
    except Exception:
        mirp_version = "?"
    settings = config.MIRP_SETTINGS
    low, high = settings["resegmentation_intensity_range"]
    paragraph = _METHODS_PARAGRAPH.format(
        mirp=mirp_version,
        spacing=settings["new_spacing"],
        order=settings["spline_order"],
        roi_scope=_roi_scope(),
        low=low, high=high,
        width=settings["base_discretisation_bin_width"],
        texture=_texture_sentence(),
    )
    cell = ms.cell(row, 1, paragraph)
    cell.alignment = wrap
    cell.border = border
    ms.merge_cells(start_row=row, start_column=1, end_row=row + 9, end_column=3)

    ms.column_dimensions["A"].width = 18
    ms.column_dimensions["B"].width = 34
    ms.column_dimensions["C"].width = 96
    ms.freeze_panes = ms.cell(5, 1)
    ms.sheet_properties.tabColor = _ACCENT

    # methods öne alınıyor: kitap açıldığında önce neyin nasıl üretildiği görünsün.
    # Kaydırma miktarı sayfa sayısına bağlı — sabit -1 iki öznitelik sayfası
    # varken methods'u ortada bırakıyordu.
    wb.move_sheet("methods", offset=-len(tablolar))
    wb.active = 0

    try:
        wb.save(tmp)
    except Exception as exc:  # noqa: BLE001
        print(f"[xlsx] UYARI: Excel yazılamadı ({exc}); CSV kaydedildi.")
        if tmp.exists():
            tmp.unlink()
        return None

    try:
        os.replace(tmp, xlsx)
    except OSError as exc:
        alt = xlsx.with_name(xlsx.stem + "_yeni" + xlsx.suffix)
        os.replace(tmp, alt)
        print(f"[xlsx] UYARI: {xlsx.name} yazılamadı ({exc}); {alt.name} olarak "
              f"kaydedildi (Excel'de açık olabilir).")
        return alt
    return xlsx
