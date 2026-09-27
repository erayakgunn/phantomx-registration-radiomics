"""Aşama 7 — üretilmiş çalışma kitabını doğrula.

Önceki aşamalar girdiyi ve hattı denetliyor. Bu dosya ÇIKTIYA bakıyor: bir
çalıştırma bittikten sonra diskteki `radiomics_combined.xlsx` gerçekten
beklenen şeyi mi içeriyor?

Bunun ayrı bir aşama olmasının sebebi somut: hat exit 0 dönebilir, tüm satırları
toplayabilir ve yine de kullanılamaz bir kitap yazabilir — eksik bir sayfa,
kaymış bir kolon düzeni, tamamı NaN bir feature bloğu, ya da bir cihazda hiç
dolmayan bir kolon. Bunların hiçbiri çalıştırma logunda görünmez.

Çalışma kitabı yoksa testler atlanıyor, yani çıkarım yapılmamış bir makinede
paket yine çalışır.
"""

from __future__ import annotations

import math

import pytest

from conftest import DEVICE_DATASETS, PIPELINE_PATH

pytestmark = pytest.mark.slow

DEVICES = list(DEVICE_DATASETS)

EXPECTED_SERIES = {"USZ_PCCT": 80, "KSW": 72, "EID": 100}
ROIS_PER_SERIES = 90                      # 40 insert x 2 boyut + 2 star + 8 texture
EXPECTED_SHEETS = ["methods", "series", "insert_ranking", "2D_base", "3D_base"]

# Feature sayfalarının sabit sol bloğu.
META_BLOCK = ["label", "insert_size", "series", "n_vox", "recon_k", "k_strength",
              "kVp", "mAs", "FOV", "SliceThickness", "me_keV", "repeat"]

# Insert'ler yalnızca first-order `statistics` alıyor; star/texture tüm aileler.
MIN_INSERT_FEATURES = 15
MIN_TEXTURE_FEATURES = 150


def workbook_path(device: str):
    """Bu cihazın çalışma kitabı — yoksa ya da KODDAN ESKİYSE testi atlar.

    Eskilik kontrolü şart. Bu dosyanın testleri diskteki ARTEFAKTI doğruluyor,
    hattın kendisini değil. Yazım katmanı düzeltildikten sonra eski bir kitap
    hâlâ orada duruyor ve testler onu okursa iki kötü sonuçtan biri çıkıyor:
    ya düzeltilmiş kodu suçlayan yanlış bir başarısızlık, ya da o kitabı üreten
    hatanın hâlâ orada olduğunu gizleyen sessiz bir geçiş. İkisi de kodun
    değişmiş, artefaktın değişmemiş olduğu gerçeğinden daha kötü bir mesaj verir.

    Karşılaştırma dosya zamanına göre: kitap hattın kaynağından eskiyse
    yeniden üretilmesi gerekiyor. Checkpoint'ler durduğu için bu ucuz — aynı
    komut yeniden çalıştırıldığında çıkarım atlanır, yalnızca kitap yazılır.
    """
    path = DEVICE_DATASETS[device] / "radiomics_output" / "radiomics_combined.xlsx"
    if not path.is_file():
        pytest.skip(f"{device}: çalışma kitabı yok ({path})")
    # Truncated artefact. An interrupted run leaves a partial zip behind, and
    # every test in this file then fails on an unreadable file — eighteen
    # confusing failures that say nothing about the pipeline. The actionable fact
    # is "the artefact is incomplete, regenerate it", so say that once.
    import zipfile

    if not zipfile.is_zipfile(path):
        pytest.skip(
            f"{device}: çalışma kitabı bozuk/yarım "
            f"({path.stat().st_size / 1e6:.1f} MB, geçerli bir xlsx değil) — "
            f"yarıda kesilmiş bir çalıştırmadan kalmış olmalı. Yeniden üret:\n"
            f'  "C:/Python314/python.exe" '
            f"extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py "
            f'--dataset-root "{DEVICE_DATASETS[device]}"'
        )
    if path.stat().st_mtime < PIPELINE_PATH.stat().st_mtime:
        pytest.skip(
            f"{device}: çalışma kitabı hattın kaynağından eski, yani şu anki "
            f"yazıcıyla üretilmemiş. Yeniden üret (checkpoint'ler sayesinde "
            f"çıkarım atlanır):\n"
            f'  "C:/Python314/python.exe" '
            f"extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py "
            f'--dataset-root "{DEVICE_DATASETS[device]}"'
        )
    return path


@pytest.fixture(scope="module")
def loaded():
    """device -> {sheet: DataFrame}, modül başına bir kez okunur (dosyalar büyük)."""
    import pandas as pd

    cache = {}

    def load(device: str):
        if device not in cache:
            path = workbook_path(device)
            cache[device] = {
                "3D_base": pd.read_excel(path, sheet_name="3D_base"),
                "2D_base": pd.read_excel(path, sheet_name="2D_base"),
                "series": pd.read_excel(path, sheet_name="series"),
                "insert_ranking": pd.read_excel(path, sheet_name="insert_ranking"),
            }
        return cache[device]

    return load


# --------------------------------------------------------------------------- #
# Yapı
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", DEVICES)
def test_workbook_has_every_expected_sheet_with_methods_first(device):
    from openpyxl import load_workbook

    wb = load_workbook(workbook_path(device), read_only=True)
    assert wb.sheetnames[0] == "methods", (
        "kitap açıldığında önce sayıların NASIL üretildiği görünmeli"
    )
    assert sorted(wb.sheetnames) == sorted(EXPECTED_SHEETS), wb.sheetnames
    wb.close()


@pytest.mark.parametrize("device", DEVICES)
def test_row_count_is_series_times_rois(device, loaded):
    """Satır sayısı tam olmalı — eksik bir ROI yüzlerce satır arasında görünmez."""
    expected = EXPECTED_SERIES[device] * ROIS_PER_SERIES
    for sheet in ("3D_base", "2D_base"):
        frame = loaded(device)[sheet]
        assert len(frame) == expected, (
            f"{device}/{sheet}: {len(frame)} satır, {expected} bekleniyordu "
            f"({EXPECTED_SERIES[device]} seri x {ROIS_PER_SERIES} ROI)"
        )
        assert frame["series"].nunique() == EXPECTED_SERIES[device]


@pytest.mark.parametrize("device", DEVICES)
def test_meta_block_is_in_the_documented_order(device, loaded):
    """Sol blok sabit: konumsal okuyan downstream analiz buna dayanıyor."""
    for sheet in ("3D_base", "2D_base"):
        columns = list(loaded(device)[sheet].columns)
        assert columns[:len(META_BLOCK)] == META_BLOCK, columns[:len(META_BLOCK)]
        assert columns[-1] == "section", "renklendirme yardımcısı en sağda olmalı"


@pytest.mark.parametrize("device", DEVICES)
def test_every_roi_appears_once_per_series(device, loaded):
    """(seri, label, insert_size) tekil olmalı — pivot hiçbir satırı çökertmemeli."""
    frame = loaded(device)["3D_base"]
    key = frame[["series", "label", "insert_size"]].astype(str)
    duplicated = key.duplicated().sum()
    assert duplicated == 0, f"{device}: {duplicated} yinelenen ROI satırı"
    assert frame["label"].nunique() == 50, "40 insert + 2 star + 8 texture"


# --------------------------------------------------------------------------- #
# İçerik — kolonlar gerçekten dolu mu
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", DEVICES)
def test_feature_values_are_present_for_every_roi_group(device, loaded):
    """Insert'lerde first-order, star/texture'da tüm aileler dolu olmalı.

    Tamamı NaN bir feature bloğu logda görünmez: MIRP çağrısı başarılı sayılır
    ve satır yazılır.
    """
    frame = loaded(device)["3D_base"]
    meta_and_tags = set(META_BLOCK) | {"section"}
    features = [c for c in frame.columns
                if c not in meta_and_tags and not str(c).startswith("dcm_")]
    assert features, f"{device}: hiç feature kolonu yok"

    inserts = frame[frame["insert_size"].astype(str).isin(["5mm", "10mm"])]
    textures = frame[frame["label"] == "Honeylarge"]
    assert len(inserts) == EXPECTED_SERIES[device] * 80
    assert len(textures) == EXPECTED_SERIES[device]

    n_insert = int(inserts[features].notna().any().sum())
    n_texture = int(textures[features].notna().any().sum())
    assert n_insert >= MIN_INSERT_FEATURES, (
        f"{device}: insert'lerde yalnızca {n_insert} dolu feature"
    )
    assert n_texture >= MIN_TEXTURE_FEATURES, (
        f"{device}: texture'da yalnızca {n_texture} dolu feature"
    )
    # Insert politikası first-order: texture'dan belirgin şekilde az olmalı.
    assert n_insert < n_texture


@pytest.mark.parametrize("device", DEVICES)
def test_dropped_features_are_absent_from_the_output(device, pipeline, loaded):
    """Tekrarlanamaz feature'lar kitaba hiç girmemeli."""
    for sheet in ("3D_base", "2D_base"):
        columns = [str(c) for c in loaded(device)[sheet].columns]
        for dropped in pipeline._DROPPED_FEATURES:
            offenders = [c for c in columns
                         if c == dropped or c.startswith(dropped + "_")]
            assert not offenders, f"{device}/{sheet}: {offenders}"


@pytest.mark.parametrize("device", DEVICES)
def test_no_padding_sentinel_reached_the_measurements(device, loaded):
    """Hiçbir ölçüm exporter'ın −33792 HU dolgusunu görmemiş olmalı.

    Sentinel MIRP'e girseydi ortalama HU o ROI'de tanınmaz biçimde düşerdi.
    """
    frame = loaded(device)["3D_base"]
    stat_cols = [c for c in frame.columns if str(c).startswith("stat_mean")]
    assert stat_cols, f"{device}: stat_mean kolonu yok"
    values = frame[stat_cols[0]].dropna()
    assert values.min() > -2000, (
        f"{device}: en düşük stat_mean {values.min()} — sentinel sızmış olabilir"
    )
    assert values.max() < 31743


# --------------------------------------------------------------------------- #
# Cihaza özel meta — boş kolon bırakmadan
# --------------------------------------------------------------------------- #

EXPECTED_STRENGTHS = {
    "USZ_PCCT": {"1", "2", "3", "4"},
    "KSW": {"1", "2", "3", "4"},
    "EID": {"", "1", "2", "3", "4", "5"},
}


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruction_strength_reached_the_workbook(device, loaded):
    """k_strength kitapta gerçekten dolu olmalı, sadece kodda değil."""
    frame = loaded(device)["3D_base"]
    got = {"" if str(v) in ("nan", "None") else str(v).strip()
           for v in frame["k_strength"].unique()}
    got = {v.removesuffix(".0") for v in got}
    assert got == EXPECTED_STRENGTHS[device], (
        f"{device}: kitapta k_strength = {sorted(got)}, "
        f"{sorted(EXPECTED_STRENGTHS[device])} bekleniyordu"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_kernel_column_is_groupable(device, loaded):
    """recon_k Python liste yazımı içermemeli, yoksa Excel'de gruplanamaz."""
    frame = loaded(device)["3D_base"]
    kernels = {str(v) for v in frame["recon_k"].unique()}
    for kernel in kernels:
        assert not kernel.startswith("["), f"{device}: {kernel!r}"
        assert "'" not in kernel, f"{device}: {kernel!r}"


@pytest.mark.parametrize("device", DEVICES)
def test_repeat_column_is_numbered_from_one(device, loaded):
    """Test-retest indeksi her protokol için 1'den başlamalı."""
    frame = loaded(device)["3D_base"]
    per_series = frame.drop_duplicates(subset=["series"])
    values = sorted({int(v) for v in per_series["repeat"]})
    assert values[0] == 1, f"{device}: repeat {values}"
    assert values == list(range(1, max(values) + 1)), f"{device}: {values} boşluklu"
    if device == "USZ_PCCT":
        assert max(values) == 3, "USZ'de 3 tekrarlı protokoller var"
        assert int((per_series["repeat"] > 1).sum()) == 16


@pytest.mark.parametrize("device", DEVICES)
def test_discriminating_tags_are_present_and_populated(device, pipeline, loaded):
    """Mavi bloğa yükseltilen ayırıcı tag'ler kitapta ve dolu olmalı.

    Cihaza özgü olanlar (Siemens IQ level, Philips iDose) diğer cihazlarda
    doğal olarak boş; cihazdan bağımsız olanlar her satırda dolu olmalı.
    """
    frame = loaded(device)["3D_base"]
    for column in ("dcm_SeriesDescription", "dcm_SeriesNumber",
                   "dcm_AcquisitionNumber", "dcm_AcquisitionTime", "dcm_CTDIvol"):
        assert column in frame.columns, f"{device}: {column} yok"
        assert frame[column].notna().all(), f"{device}: {column} boş satır içeriyor"

    vendor_column = {"USZ_PCCT": "dcm_(0029,1043)",
                     "KSW": "dcm_(00E1,1040)_Image_Label",
                     "EID": None}[device]
    if vendor_column:
        assert vendor_column in frame.columns, f"{device}: {vendor_column} yok"
        assert frame[vendor_column].notna().all()


@pytest.mark.parametrize("device", DEVICES)
def test_every_series_row_is_uniquely_identifiable(device, pipeline, loaded):
    """Kitapta hiçbir iki seri, seri adı hariç tutulduğunda ayırt edilemez kalmamalı.

    Seri adı bu veri setlerinde ölçülerek yanıltıcı bulundu, dolayısıyla
    ayırt edicilik ona dayanamaz.
    """
    frame = loaded(device)["3D_base"]
    identity = [c for c in frame.columns
                if c in META_BLOCK or str(c).startswith("dcm_")]
    identity = [c for c in identity if c != "series"]
    per_series = frame.drop_duplicates(subset=["series"])[["series"] + identity]
    # str() per value rather than .astype(str): under pandas 3.0's string dtype a
    # NaN survives astype(str) as a float and "|".join then raises.
    fingerprint = per_series[identity].apply(
        lambda row: "|".join(str(v) for v in row), axis=1)
    assert fingerprint.nunique() == len(per_series), (
        f"{device}: {len(per_series) - fingerprint.nunique()} seri, seri adı "
        f"olmadan ayırt edilemiyor"
    )


# --------------------------------------------------------------------------- #
# Beyan ile içerik çelişmemeli
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", DEVICES)
def test_methods_roi_count_equals_the_feature_sheet_row_count(device, loaded):
    """`ROIs measured` yanındaki sayfanın satır sayısıyla aynı olmalı.

    Sayım `insert_size` olmadan yapıldığında her insert'in 5 mm ve 10 mm ROI'si
    tek gruba çöküyordu: seri başına 90 yerine 50, yani %44 eksik — ve bu, tam
    olarak metod bölümüne kopyalanmak için var olan sayfada.
    """
    import pandas as pd

    frame = pd.read_excel(workbook_path(device), sheet_name="methods", header=None)
    declared = None
    for row in frame.itertuples(index=False):
        cells = [str(v) for v in row if pd.notna(v)]
        if any(c == "ROIs measured" for c in cells):
            declared = int(cells[cells.index("ROIs measured") + 1])
    assert declared is not None, "methods sayfasında 'ROIs measured' yok"
    assert declared == len(loaded(device)["3D_base"]), (
        f"{device}: beyan {declared}, sayfada {len(loaded(device)['3D_base'])} satır"
    )
    assert declared == EXPECTED_SERIES[device] * ROIS_PER_SERIES


@pytest.mark.parametrize("device", DEVICES)
def test_series_sheet_repeat_column_is_populated(device, loaded):
    """methods sayfası "see the 'repeat' column" diyor; o sütun boş olmamalı.

    `series_meta` frame'in kopyası olarak repeat atanmadan ÖNCE alınıyordu, yani
    sayfa checkpoint'lerin taşıdığı boş placeholder'ı yazıyordu ve beyandaki
    çapraz referans boşluğa çıkıyordu.
    """
    frame = loaded(device)["series"]
    assert "repeat" in frame.columns, f"{device}: series sayfasında repeat yok"
    assert frame["repeat"].notna().all(), f"{device}: repeat boş satır içeriyor"
    values = sorted({int(v) for v in frame["repeat"]})
    assert values == list(range(1, max(values) + 1))
    if device == "USZ_PCCT":
        assert int((frame["repeat"] > 1).sum()) == 16


@pytest.mark.parametrize("device", DEVICES)
def test_displayed_precision_does_not_hide_protocol_differences(device):
    """Ekranda görünen değer, protokoller arası farkı gizlememeli.

    Sabit "0.000" biçimi ölçüldü: bir doku ROI'sinde 207 öznitelik kolonunun
    102'si gerçekte taşıdığından az sayıda değer gösteriyordu ve
    ``rlm_lrlge_3d_v_mrg`` 80 farklı protokol değerini iki string olarak
    yazıyordu — üstelik o kolonun kendi yayılımı ortalamasının %71'i. Saklanan
    değer tam hassasiyette olduğu için formüller etkilenmiyordu, ama bu kitap
    Excel'de okunmak için var: ekranda sabit görünen bir öznitelik incelenmez.
    """
    from openpyxl import load_workbook

    wb = load_workbook(workbook_path(device), read_only=True)
    try:
        ws = wb["3D_base"]
        rows = ws.iter_rows(values_only=False)
        header = [c.value for c in next(rows)]
        skip = set(META_BLOCK) | {"section"}
        watched = {i: h for i, h in enumerate(header)
                   if h not in skip and not str(h).startswith("dcm_")}
        # Her ROI ayrı ayrı: bir karşılaştırma tek bir ROI'yi protokoller
        # boyunca okur, ve biçim TÜM kolondan seçildiği için bir ROI'de yeterli
        # olan çözünürlük başka bir ROI'de yetmeyebilir.
        label_i = header.index("label")
        per_roi: dict = {}
        for row in rows:
            label = row[label_i].value
            bucket = per_roi.setdefault(label, {i: [] for i in watched})
            for i in watched:
                cell = row[i]
                value = cell.value
                # Ints as well as floats. Many HU-derived features land on whole
                # numbers and Excel stores those as integers; sampling only
                # floats measured three values out of 160 for stat_p10 and
                # invented a 1.0-wide "spread" that does not exist.
                if (isinstance(value, (int, float))
                        and not isinstance(value, bool)):
                    bucket[i].append((float(value), cell.number_format))
    finally:
        wb.close()

    def rendered(value: float, fmt: str) -> str:
        if "E+" in fmt.upper():
            decimals = len(fmt.split("E")[0].split(".")[1]) if "." in fmt else 0
            return f"{value:.{decimals}e}"
        decimals = len(fmt.split(".")[1]) if "." in fmt else 0
        return f"{value:.{decimals}f}"

    # Ölçüt: her float'ın ekranda ayrı görünmesi DEĞİL. 80 protokolün bir kısmı
    # 12. anlamlı hanede ayrışıyor; onu göstermek sinyal değil gürültü basmak
    # olur. İstenen, kolonun YAYILIMININ okunabilir olması: gösterim adımı
    # yayılımın yüzde birinden küçük olmalı, yani aralık en az 100 ayırt
    # edilebilir basamağa bölünmeli.
    #
    # Bağıl yayılımı 1e-9'un altında kalan kolonlar (ör.
    # morph_vol_dens_conv_hull, kutu maskesi yüzünden yapısal olarak 1,0 ve
    # yalnızca 2e-16 oynuyor) dışarıda: orada gösterilecek bir fark yok.
    # 50 basamak: eşik ölçülerek seçildi, keyfi değil. Kusurun kendisi 2 basamak
    # gösteriyordu; en dar gerçek durum (Insert_12 / stat_var, 6 haneli tam sayı
    # kolonu) 95 basamak veriyor. 50, kusurdan 25 kat sıkı ve bir eğilimi
    # okumak için fazlasıyla yeterli — daha yükseği stat_var gibi kolonlara
    # bilgi taşımayan bir ondalık eklemekten başka iş yapmaz.
    LEVELS = 50
    NOISE_FLOOR = 1e-9
    unreadable = []
    for label, values in per_roi.items():
        for i, name in watched.items():
            pairs = values[i]
            if len(pairs) < 2:
                continue
            distinct = {v for v, _ in pairs}
            if len(distinct) < 2:
                continue
            spread = max(distinct) - min(distinct)
            scale = max(abs(v) for v in distinct) or 1.0
            if spread / scale < NOISE_FLOOR:
                continue

            fmt = str(pairs[0][1])
            if "E+" in fmt.upper():
                mantissa = (len(fmt.split("E")[0].split(".")[1])
                            if "." in fmt else 0)
                step = 10.0 ** (math.floor(math.log10(scale)) - mantissa)
            else:
                decimals = len(fmt.split(".")[1]) if "." in fmt else 0
                step = 10.0 ** -decimals

            if step > spread / LEVELS:
                unreadable.append((label, name, fmt,
                                   f"levels={spread / step:.1f}"))

    assert not unreadable, (
        f"{device}: {len(unreadable)} (ROI, kolon) çiftinin yayılımı ekranda "
        f"{LEVELS} basamağa bölünemiyor: {unreadable[:3]}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_meta_columns_have_a_single_excel_type(device):
    """Bir kolonda hem metin hem sayı hücresi bulunmamalı.

    Excel'in filtresi ve `=A2=1` gibi karşılaştırmalar hücre TİPİNE bakıyor.
    Taze okunan seriler `k_strength`'i string, checkpoint'ten dönenler int
    veriyordu; kısmi bir resume ikisini aynı kolonda karıştırıyor ve filtre
    aynı seviyeyi iki kez listeliyordu — üstelik bu, EID (ADMIRE 1..5) ve KSW
    (IMR/iDose) setlerinin ana değişkeni olan kolonda.
    """
    from collections import Counter

    from openpyxl import load_workbook

    wb = load_workbook(workbook_path(device), read_only=True)
    try:
        for sheet, columns in (("3D_base", ("k_strength", "repeat", "kVp")),
                               ("series", ("k_strength", "date", "repeat"))):
            rows = wb[sheet].iter_rows(values_only=False)
            header = [c.value for c in next(rows)]
            index = {c: header.index(c) for c in columns if c in header}
            assert index, f"{device}/{sheet}: hiçbiri bulunamadı: {columns}"
            kinds = {c: Counter() for c in index}
            for row in rows:
                for column, i in index.items():
                    cell = row[i]
                    # BOŞ hücreler sayılmaz. Yakalanmak istenen hata aynı
                    # DEĞERİN iki tipte görünmesi (metin "1" ile sayı 1 Excel'de
                    # ayrı değerlerdir). Boşluk bir değer değil, değerin
                    # yokluğu — EID'de 30 non-iteratif serinin k_strength'i
                    # meşru biçimde boş ve openpyxl boşları inlineStr yazıyor.
                    if cell.value in (None, ""):
                        continue
                    kinds[column][cell.data_type] += 1
            for column, counter in kinds.items():
                assert len(counter) <= 1, (
                    f"{device}/{sheet}.{column}: karışık Excel hücre tipi "
                    f"{dict(counter)} — filtre aynı değeri iki kez listeler"
                )
    finally:
        wb.close()


# --------------------------------------------------------------------------- #
# Yardımcı sayfalar
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", DEVICES)
def test_series_sheet_has_one_row_per_series(device, loaded):
    frame = loaded(device)["series"]
    assert len(frame) == EXPECTED_SERIES[device]
    assert frame["series"].nunique() == EXPECTED_SERIES[device]


@pytest.mark.parametrize("device", DEVICES)
def test_insert_ranking_covers_all_forty_inserts(device, loaded):
    frame = loaded(device)["insert_ranking"]
    assert len(frame) == 40, f"{device}: {len(frame)} insert sıralandı"
    assert sorted(frame["rank"]) == list(range(1, 41))
    # rank ETİKETLEME sırasıdır: Insert_01 -> 1. Sayfa ile overlay/`label`
    # numarası arasında fark kalmasin diye (eskiden HU'ya göre artan sıralanıyordu,
    # bu da tam olarak 41 - numara demekti).
    numara = frame["label"].str.split("_").str[1].astype(int)
    assert frame["rank"].tolist() == numara.tolist(), "rank insert numarası olmalı"
    hu = frame.sort_values("rank")["mean_HU_global"].tolist()
    assert hu == sorted(hu, reverse=True), "etiketleme sırasında HU azalan olmalı"


# --------------------------------------------------------------------------- #
# Cihazlar arası tutarlılık — aynı fantom, üç tarayıcı
# --------------------------------------------------------------------------- #

def _all_devices_or_skip():
    """Üç cihazın da güncel bir kitabı olmalı — karşılaştırma hepsini ister."""
    for device in DEVICES:
        workbook_path(device)


def test_insert_ranking_agrees_across_devices():
    """40 insert üç cihazda da aynı sırada olmalı.

    Bu, tüm zincirin en güçlü tek doğrulaması. Insert'ler fiziksel olarak sabit
    malzemeler; hangi tarayıcı ölçerse ölçsün yoğunluk SIRASI değişmemeli.
    Değişiyorsa sebep tarayıcı değil hattır: kayan bir ROI eşlemesi, yanlış
    numaralanmış insert klasörleri, ya da bir cihazda bozulan geometri.
    Mutlak HU cihazlar arası kayabilir (spektrum farkı) — sıra kayamaz.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    _all_devices_or_skip()
    rankings = {}
    for device in DEVICES:
        frame = pd.read_excel(workbook_path(device), sheet_name="insert_ranking")
        rankings[device] = frame.set_index("label")["rank"]

    labels = sorted(set.intersection(*(set(r.index) for r in rankings.values())))
    assert len(labels) == 40, f"ortak insert sayısı {len(labels)}"

    for a, b in [("USZ_PCCT", "KSW"), ("USZ_PCCT", "EID"), ("KSW", "EID")]:
        rho = spearmanr(rankings[a][labels], rankings[b][labels]).statistic
        assert rho > 0.97, (
            f"{a} ve {b} insert sıralaması uyuşmuyor (Spearman rho={rho:.4f}). "
            f"Aynı fiziksel insert'ler farklı sıralanıyor — bu bir tarayıcı "
            f"farkı değil, hatta bir eşleme/geometri hatası işareti."
        )
        # En büyük tek sapma da sınırlı kalmalı: yüksek rho birkaç büyük
        # yer değiştirmeyi gizleyebilir.
        shift = (rankings[a][labels] - rankings[b][labels]).abs().max()
        assert shift <= 3, (
            f"{a} vs {b}: bir insert {shift} basamak kaymış — komşu HU'lu "
            f"insert'lerin yer değiştirmesi beklenir, büyük kayma beklenmez"
        )


def test_the_same_insert_has_a_comparable_mean_hu_on_every_device():
    """Aynı insert'in mutlak HU'su cihazlar arası makul aralıkta kalmalı.

    Kalibre edilmiş bir CT'de su 0, hava −1000 HU. Farklı tarayıcı/spektrum
    orta yoğunluklarda kayma yapar ama bir insert bir cihazda −400, ötekinde
    +400 HU okunuyorsa bu kalibrasyon değil eşleme hatasıdır.
    """
    import pandas as pd

    _all_devices_or_skip()
    means = {}
    for device in DEVICES:
        frame = pd.read_excel(workbook_path(device), sheet_name="insert_ranking")
        means[device] = frame.set_index("label")["mean_HU_global"]

    table = pd.DataFrame(means)
    spread = (table.max(axis=1) - table.min(axis=1))
    worst = spread.idxmax()
    assert spread.max() < 250, (
        f"{worst}: cihazlar arası HU yayılımı {spread.max():.0f} HU "
        f"({table.loc[worst].to_dict()}) — eşleme hatası olabilir"
    )


def test_voxel_counts_follow_geometry_not_chance():
    """n_vox yalnızca ROI boyutu, FOV ve dilim kalınlığından türemeli.

    Kırpma deterministik; aynı (ROI, FOV, kalınlık) üçlüsü her cihazda aynı
    voksel sayısını vermeli. Vermiyorsa kırpma girdiye göre kayıyor demektir ve
    hacim/doku feature'ları karşılaştırılamaz.
    """
    import pandas as pd

    _all_devices_or_skip()
    frames = []
    for device in DEVICES:
        frame = pd.read_excel(workbook_path(device), sheet_name="3D_base")
        frames.append(frame.assign(device=device)[
            ["device", "label", "insert_size", "FOV", "SliceThickness", "n_vox"]])
    table = pd.concat(frames, ignore_index=True)

    grouped = table.groupby(
        ["label", "insert_size", "FOV", "SliceThickness"], dropna=False)["n_vox"]
    inconsistent = grouped.nunique()
    offenders = inconsistent[inconsistent > 1]
    assert offenders.empty, (
        f"{len(offenders)} (ROI, boyut, FOV, kalınlık) kombinasyonunda voksel "
        f"sayısı cihaza göre değişiyor, ör.\n"
        f"{table[table.set_index(['label','insert_size','FOV','SliceThickness']).index.isin(offenders.index[:1])].to_string(index=False)}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_methods_sheet_declares_the_live_parameters(device, pipeline):
    """Beyan sayfası gerçekten çalıştırmayı üreten sabitleri yazmalı."""
    import pandas as pd

    frame = pd.read_excel(workbook_path(device), sheet_name="methods", header=None)
    text = " ".join(str(v) for v in frame.to_numpy().ravel() if pd.notna(v))
    assert f"{pipeline._DISCRETISATION_ANCHOR_HU:g}" in text
    assert "fixed bin size" in text.lower()
    assert "Not computed" in text, "IBSI-2 filtre katmanı beyanı eksik"
    assert str(EXPECTED_SERIES[device]) in text
