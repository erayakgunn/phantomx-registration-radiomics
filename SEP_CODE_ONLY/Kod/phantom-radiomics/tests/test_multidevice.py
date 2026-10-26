"""Aşama 6 — üç cihaz, tek hat.

Hat aynı fantomun üç ayrı tarayıcıdaki çekimlerinde çalıştırılıyor:
foton sayan NAEOTOM Alpha, Philips Spectral CT 7500 ve enerji integre eden
SOMATOM Edge Plus. Üçü de aynı etiketleme çıktısını üretiyor ama DICOM
tarafında hiç benzemiyorlar, ve her farklılık bir şeyi bir kez bozdu:

  * ``ConvolutionKernel`` SOMATOM'da ÇOK DEĞERLİ (``['Br40f', '3']``).
    ``str()`` bunu Python liste yazımına çevirdiği için aynı çekirdek 100
    serinin 70'inde ``['Br40f', '3']``, 30'unda ``Br40f`` olarak yazılıyordu —
    Excel'de gruplanamayan bir kolon. Üstelik gücün 1..5 arasında değiştiği
    o ikinci eleman atılıyordu, yani setin ANA değişkeni hiçbir yerde yoktu.
  * Rekonstrüksiyon gücü her cihazda başka yerde: Siemens private tag'i,
    SOMATOM'da çekirdeğin içinde, Philips'te iDose/IMR private tag'lerinde.
    Yalnızca ilkini okumak 252 serinin 172'sinde ``k_strength``'i boş
    bırakıyordu.
  * Havanın tabanı cihaza göre değişiyor: NAEOTOM −1203 HU'ya kadar sürekli
    iniyor, Philips tam −1024'te KESİLİYOR. Çapa (−2000) üçünde de güvende,
    ama kesilme bir yorumlama uyarısı ve sessizce değişmemeli.
  * Ham CT uzantısı EID'de ``.IMA``, diğerlerinde ``.dcm``.
  * Aynı protokol birden çok kez çekiliyor (test-retest). Bu tekrarların
    kopya mı, gerçek yeniden çekim mi, yoksa bir tag'le ayrılmış farklı
    rekonstrüksiyon mu olduğu ayırt edilebilir olmalı.

Testler salt-okunur ve veri seti yoksa atlanıyor.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from conftest import DEVICE_DATASETS, device_root, real_hu, series_dirs

pytestmark = pytest.mark.slow

DEVICES = list(DEVICE_DATASETS)

# Beklenen envanter: 40 insert x 2 boyut + 2 yıldız + 8 doku.
EXPECTED_INSERTS, EXPECTED_STARS, EXPECTED_TEXTURES = 80, 2, 8

EXPECTED_SERIES = {"USZ_PCCT": 80, "KSW": 72, "EID": 100}

# Ham CT dosya uzantısı. EID native Siemens export'u olduğu için .IMA;
# _CT_DICOM_PATTERNS bunu kapsamazsa o cihazın HİÇBİR serisi bulunamaz.
EXPECTED_CT_SUFFIX = {"USZ_PCCT": ".dcm", "KSW": ".dcm", "EID": ".IMA"}

# Yüksek riskli ROI'ler: içlerinde hava boşluğu olanlar, yani HU tabanını
# belirleyenler. Katı insert'ler bu soruya cevap vermiyor.
AIR_ROIS = [("Textures", "Honeylarge"), ("Textures", "Honeysmall"),
            ("Textures", "Sinusoidal"), ("Stars", "Star16"), ("Stars", "Star8")]


def _read_meta(pipeline, series_dir):
    """(meta, tags) — WARN çıktısını yutarak."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        meta = pipeline._dicom_series_meta_from_header(series_dir)
        tags = pipeline.read_dicom_tags(pipeline._find_dicoms_dir(series_dir))
    return meta, tags


# --------------------------------------------------------------------------- #
# Keşif ve envanter
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", DEVICES)
def test_every_series_is_discovered_with_no_rejections(pipeline, device):
    """Hepsi bulunmalı ve hiçbiri sessizce düşmemeli."""
    root = device_root(device)
    pipeline._REJECTED_SERIES.clear()
    found = pipeline.discover_series(root)

    assert len(found) == EXPECTED_SERIES[device], (
        f"{device}: {len(found)} seri bulundu, {EXPECTED_SERIES[device]} bekleniyordu"
    )
    assert pipeline._REJECTED_SERIES == [], (
        f"{device}: reddedilen klasör var -> {pipeline._REJECTED_SERIES[:3]}"
    )
    assert len(found) == len(series_dirs(root))


@pytest.mark.parametrize("device", DEVICES)
def test_every_series_yields_the_full_roi_inventory(pipeline, device):
    """Seri başına 90 ROI. Eksik bir ROI yüzlerce satır arasında görünmez."""
    root = device_root(device)
    for s in series_dirs(root):
        n_ins = len(pipeline.discover_insert_roi_dirs(s))
        n_star = len(pipeline.discover_star_roi_dirs(s))
        n_tex = len(pipeline.discover_texture_roi_dirs(s))
        assert (n_ins, n_star, n_tex) == (
            EXPECTED_INSERTS, EXPECTED_STARS, EXPECTED_TEXTURES), (
            f"{device}/{s.name}: inserts={n_ins} stars={n_star} textures={n_tex}"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_texture_and_star_labels_match_the_section_map(pipeline, device):
    """Doku adları TEXTURE_TO_SECTION ile birebir eşleşmeli.

    Eşleşmeyen bir ad sessizce STAR16'ya düşer (``.get(name, "STAR16")``), yani
    satır yanlış renklenir ve yanlış alt-gruba sayılır.
    """
    root = device_root(device)
    s = series_dirs(root)[0]
    textures = {label for _, label, _ in pipeline.discover_texture_roi_dirs(s)}
    stars = {label for _, label, _ in pipeline.discover_star_roi_dirs(s)}

    assert textures == set(pipeline.TEXTURE_TO_SECTION), (
        f"{device}: beklenmeyen/eksik doku adı -> "
        f"{textures ^ set(pipeline.TEXTURE_TO_SECTION)}"
    )
    assert stars == {"Star16", "Star8"}


@pytest.mark.parametrize("device", DEVICES)
def test_raw_ct_extension_is_matched_by_the_reader(pipeline, device):
    """EID .IMA yazıyor; desen listesi bunu kapsamazsa cihaz tamamen kaybolur."""
    root = device_root(device)
    s = series_dirs(root)[0]
    dicom_dir = pipeline._find_dicoms_dir(s)
    assert dicom_dir is not None, f"{device}: ham CT klasörü bulunamadı"

    files = pipeline._list_ct_dicom_files(dicom_dir)
    assert files, f"{device}: _list_ct_dicom_files hiçbir dosya bulamadı"
    suffixes = {f.suffix for f in files}
    assert suffixes == {EXPECTED_CT_SUFFIX[device]}, (
        f"{device}: ham CT uzantısı {suffixes}, "
        f"{EXPECTED_CT_SUFFIX[device]} bekleniyordu"
    )
    # ROI patch'leri her cihazda .dcm — okuyucu bunu sabit kodluyor.
    patch_dir = pipeline.discover_insert_roi_dirs(s)[0][2]
    assert {f.suffix for f in patch_dir.glob("*")} == {".dcm"}


# --------------------------------------------------------------------------- #
# HU aralığı — ROI'yi reddedebilecek tek sert kural
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", DEVICES)
def test_real_hu_stays_inside_the_discretisation_range(pipeline, device):
    """Hiçbir gerçek voksel çapanın altına inmemeli.

    ``_assert_within_discretisation_range`` aralık dışına çıkan bir ROI'yi
    reddediyor — doğru davranış, ama sessiz veri kaybı değil gürültülü bir
    başarısızlık olarak. Bu test yeni bir cihaz/çekirdek getirildiğinde bunun
    çalıştırma sırasında değil BURADA fark edilmesini sağlıyor.
    """
    root = device_root(device)
    lower, upper = pipeline.DISCRETISATION_RANGE_HU
    worst_lo, worst_hi, checked = np.inf, -np.inf, 0

    for s in series_dirs(root):
        for group, label in AIR_ROIS:
            files = sorted((s / group / label).glob("*.dcm"))
            if not files:
                continue
            out = real_hu(files[len(files) // 2])
            assert out is not None, f"{device}/{s.name}/{label}: rescale yok"
            values, _ = out
            if not values.size:
                continue
            worst_lo = min(worst_lo, float(values.min()))
            worst_hi = max(worst_hi, float(values.max()))
            checked += 1

    assert checked, f"{device}: hiç hava içeren ROI okunamadı"
    assert worst_lo > lower, (
        f"{device}: en düşük gerçek voksel {worst_lo} HU, çapa {lower} HU — "
        f"bu ROI'ler reddedilir. _DISCRETISATION_ANCHOR_HU yeniden beyan edilip "
        f"TÜM seriler yeniden çıkarılmalı."
    )
    assert worst_hi < upper, f"{device}: {worst_hi} HU kodlama tavanının üstünde"


@pytest.mark.parametrize("device", DEVICES)
def test_padding_sentinel_contract_holds_on_every_device(pipeline, device):
    """Sentinel sözleşmesi üç cihazda da aynı olmalı.

    Okuyucu tek bir büyüklük eşiğine (``hu > _PADDING_HU_FLOOR``) dayanıyor.
    Bir cihaz farklı bir intercept ya da farklı bir padding değeri yazarsa,
    uydurulmuş voksel gerçek sanılır.
    """
    import pydicom

    root = device_root(device)
    s = series_dirs(root)[0]

    disc_dir = pipeline.discover_insert_roi_dirs(s)[0][2]
    ds = pydicom.dcmread(str(sorted(disc_dir.glob("*.dcm"))[0]), force=True)
    assert float(ds.RescaleSlope) == 1.0
    assert float(ds.RescaleIntercept) == -1024.0
    assert int(ds.PixelPaddingValue) == -32768
    sentinel_hu = -32768 * 1.0 + (-1024.0)
    assert sentinel_hu < pipeline._PADDING_HU_FLOOR

    # Kare doku patch'leri sentinel İÇERMEMELİ: okuyucu onları kırpmadan
    # geçiriyor, yani tek bir sentinel doğrudan MIRP'e girerdi.
    for _, label, patch_dir in pipeline.discover_texture_roi_dirs(s):
        files = sorted(patch_dir.glob("*.dcm"))
        _, n_sentinel = real_hu(files[len(files) // 2])
        assert n_sentinel == 0, (
            f"{device}/{label}: doku patch'inde sentinel var — kırpılmadan geçer"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_air_floor_is_characterised_per_device(pipeline, device):
    """Havanın tabanı: sürekli kuyruk mu, sert kesme mi?

    KARAKTERİZASYON testi, hata testi değil. Philips tam −1024 HU'da kesiyor ve
    bir petek doku patch'inin voksellerinin ~%22'si o kesmede duruyor. Hat bunu
    düzeltemez (kayıp rekonstrüksiyonda olmuş) ama yayında beyan edilmeli:
    KSW'nin hava popülasyonu TARAYICI tarafından budanmış, dolayısıyla
    birinci-derece ve düşük-gri-seviye aileleri cihazlar arası kıyaslandığında
    bu budama da kıyasa giriyor.
    """
    root = device_root(device)
    s = series_dirs(root)[0]
    files = sorted((s / "Textures" / "Honeylarge").glob("*.dcm"))
    values, _ = real_hu(files[len(files) // 2])

    floor = float(values.min())
    at_floor = float((values == floor).mean())

    if device == "KSW":
        assert floor == -1024.0, f"Philips tabanı {floor}, −1024 bekleniyordu"
        assert at_floor > 0.10, (
            f"Philips kesmesi kaybolmuş (tabanda voksel oranı {at_floor:.3%}) — "
            f"iyi haber olabilir ama beyan metni güncellenmeli"
        )
    else:
        assert at_floor < 0.02, (
            f"{device}: voksellerin {at_floor:.2%}'si tam tabanda — bu cihaz da "
            f"artık kesiyor olabilir, hava popülasyonu tek gri seviyeye çöker"
        )


# --------------------------------------------------------------------------- #
# Rekonstrüksiyon kimliği — cihaz başına başka bir tag
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", DEVICES)
def test_kernel_is_never_a_python_list_repr(pipeline, device):
    """recon_k asla ``['Br40f', '3']`` olmamalı.

    Çok değerli bir DICOM elemanına ``str()`` uygulamak Python liste yazımı
    üretiyordu. Sonuç: aynı çekirdek iki farklı string, yani protokol
    karşılaştırmasının anahtarlandığı kolon Excel'de gruplanamıyor.
    """
    root = device_root(device)
    kernels = set()
    for s in series_dirs(root):
        meta, _ = _read_meta(pipeline, s)
        kernels.add(meta["recon_kernel"])

    for k in kernels:
        assert not k.startswith("["), f"{device}: recon_k liste yazımı -> {k!r}"
        assert "'" not in k, f"{device}: recon_k tırnak içeriyor -> {k!r}"
    # Aynı çekirdek tek bir yazımla görünmeli: 'Br40f' ile "['Br40f', '3']"
    # bir arada bulunamaz.
    bases = {re.split(r"[\\,]", k)[0] for k in kernels}
    assert len(bases) <= len(kernels)


# Cihaz başına beklenen k_strength kümesi. Boş string "tarayıcı güç
# kaydetmemiş" demek ve anlamlıdır: SOMATOM'da iteratif olmayan
# rekonstrüksiyonu ADMIRE 1..5'ten ayıran şey tam olarak bu.
EXPECTED_STRENGTHS = {
    "USZ_PCCT": {"1", "2", "3", "4"},          # NAEOTOM IQ level, private tag
    "KSW": {"1", "2", "3", "4"},               # Philips IMR 1..3 + iDose 4
    "EID": {"", "1", "2", "3", "4", "5"},      # SOMATOM ADMIRE + non-iteratif
}


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruction_strength_is_recovered_from_each_vendor_tag(pipeline, device):
    """Güç, cihazın onu yazdığı yerden okunmalı.

    Yalnızca Siemens private tag'ine bakmak 252 serinin 172'sinde bu kolonu boş
    bırakıyordu — EID'de gücün 1..5 arasında değişmesi setin ana değişkeni
    olmasına rağmen.
    """
    root = device_root(device)
    found = set()
    for s in series_dirs(root):
        meta, tags = _read_meta(pipeline, s)
        found.add(meta["k_strength"])
        # series sayfası ile feature sayfaları aynı şeyi söylemeli.
        assert tags["q_strength"] == meta["k_strength"], (
            f"{device}/{s.name}: series sayfası {tags['q_strength']!r}, "
            f"feature sayfası {meta['k_strength']!r}"
        )

    assert found == EXPECTED_STRENGTHS[device], (
        f"{device}: k_strength kümesi {sorted(found)}, "
        f"{sorted(EXPECTED_STRENGTHS[device])} bekleniyordu"
    )


def test_the_somatom_strength_comes_from_the_kernel_element(pipeline):
    """EID'de güç ConvolutionKernel'in ikinci elemanı — başka kaynağı yok."""
    root = device_root("EID")
    with_strength = without = 0
    for s in series_dirs(root):
        meta, _ = _read_meta(pipeline, s)
        if meta["k_strength"]:
            with_strength += 1
        else:
            without += 1
    # 70 ADMIRE serisi + aynı çekimlerin 30 iteratif olmayan rekonstrüksiyonu.
    assert (with_strength, without) == (70, 30)


def test_the_philips_strength_comes_from_the_idose_or_imr_tag(pipeline):
    """KSW'de güç iDose Level ya da IMR etiketinin son rakamı.

    Çekirdek harfi ('B') algoritmayı söylemiyor: MonoE serileri IMR seviye 3 ile
    rekonstrüksiyon edilmiş ve bunu yalnızca (00E1,1040) söylüyor.
    """
    import pydicom

    root = device_root("KSW")
    seen = {}
    for s in series_dirs(root):
        meta, _ = _read_meta(pipeline, s)
        ds = pydicom.dcmread(
            str(pipeline._list_ct_dicom_files(pipeline._find_dicoms_dir(s))[0]),
            stop_before_pixels=True, force=True)
        label = ds.get(pipeline._PHILIPS_IMAGE_LABEL_TAG)
        seen[(meta["recon_kernel"], str(label.value) if label else "")] = \
            meta["k_strength"]

    assert seen[("B", "IMR_BR_3")] == "3", "MonoE serisi IMR seviye 3 olmalı"
    assert seen[("IMR1,Routine", "IMR_BR_1")] == "1"
    assert seen[("IMR3,SharpPlus", "IMR_LSh_3")] == "3"
    assert seen[("YA", "iDose4_Y-Sharp")] == "4", "iDose seviyesi 4 olmalı"


def test_the_imr_level_regex_does_not_misread_a_kernel_name(pipeline):
    """"Son rakamı al" kuralı 'Qr40f'i güç 40 diye okurdu; IMR öneki şart."""
    assert pipeline._IMR_LEVEL_RE.match("IMR_BR_3").group(1) == "3"
    assert pipeline._IMR_LEVEL_RE.match("IMR_LSh_3").group(1) == "3"
    for not_a_level in ("Qr40f", "Br62f", "YA", "B", "iDose4_Y-Sharp"):
        assert pipeline._IMR_LEVEL_RE.match(not_a_level) is None, not_a_level


# --------------------------------------------------------------------------- #
# Kopya mı, test-retest mi, tag ile mi ayrılmış?
# --------------------------------------------------------------------------- #

def _protocol_table(pipeline, root):
    """series_name -> (protokol anahtarı, tüm DICOM tag'leri)."""
    import contextlib
    import io

    protocols, tags_by_series = {}, {}
    for s in series_dirs(root):
        with contextlib.redirect_stdout(io.StringIO()):
            meta = pipeline._dicom_series_meta_from_header(s)
            tags = pipeline.read_dicom_tags(pipeline._find_dicoms_dir(s))
            full = pipeline.read_dicom_full_metadata(pipeline._find_dicoms_dir(s))
        protocols[s.name] = (meta["recon_kernel"], meta["k_strength"],
                             str(meta["kvp"]), str(meta["mas"]), str(meta["fov"]),
                             str(meta["slice_thickness_label"]), tags["me_keV"])
        tags_by_series[s.name] = {f"dcm_{k}": v for k, v in full.items()}
    return protocols, tags_by_series


@pytest.mark.parametrize("device", DEVICES)
def test_every_series_is_uniquely_identifiable_in_the_workbook(pipeline, device):
    """Hiçbir iki seri workbook'ta ayırt edilemez kalmamalı.

    Ayırt edici bilgi ya çözümlenmiş meta kolonlarında ya da mavi bloğa
    yükseltilmiş DICOM tag'lerinde olmalı — seri ADINDA değil, çünkü ad bu veri
    setinde ölçülerek yanlış bulundu (klasör 80/100 kVp diyor, KVP tag'i 120,
    CTDIvol %0,0003 farkla aynı).
    """
    root = device_root(device)
    protocols, tags_by_series = _protocol_table(pipeline, root)

    promoted = list(pipeline._DICOM_PROMOTED_COLUMNS)
    fingerprints = {}
    for name, protocol in protocols.items():
        extra = tuple(str(tags_by_series[name].get(c, "")) for c in promoted)
        fingerprints.setdefault(protocol + extra, []).append(name)

    collisions = {k: v for k, v in fingerprints.items() if len(v) > 1}
    assert not collisions, (
        f"{device}: {sum(len(v) for v in collisions.values())} seri meta + mavi "
        f"blok kombinasyonunda ayırt edilemiyor, ör. {list(collisions.values())[0]}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_repeats_of_one_protocol_are_numbered_in_acquisition_order(pipeline, device):
    """Aynı protokolün tekrarları 1..n olarak numaralanmalı."""
    import pandas as pd

    root = device_root(device)
    protocols, tags_by_series = _protocol_table(pipeline, root)
    frame = pd.DataFrame([
        dict(series_name=n, recon_kernel=p[0], k_strength=p[1], kvp=p[2],
             mas=p[3], fov=p[4], slice_thickness_mm=p[5], me_kev=p[6])
        for n, p in protocols.items()
    ])
    repeats = pipeline._assign_repeat_index(frame, tags_by_series)

    assert set(repeats) == set(protocols)
    by_protocol = {}
    for name, protocol in protocols.items():
        by_protocol.setdefault(protocol, []).append(name)
    for protocol, names in by_protocol.items():
        got = sorted(repeats[n] for n in names)
        assert got == list(range(1, len(names) + 1)), (
            f"{device}: {protocol} tekrarları {got} olarak numaralandı"
        )
        # Sıra çekim zamanına göre olmalı, worker bitiş sırasına göre değil.
        ordered = sorted(names, key=lambda n: repeats[n])
        times = [tags_by_series[n].get("dcm_AcquisitionTime", "") for n in ordered]
        assert times == sorted(times), f"{device}: tekrarlar zaman sırasında değil"


def test_the_naeotom_set_contains_test_retest_repeats(pipeline):
    """USZ'de 8 protokol 3 kez çekilmiş — bu bir hata değil, veri tasarımı.

    Klasör adları bu üçlüyü 80/100/120 kVp sanıyor. KVP tag'i üçünde de 120 ve
    CTDIvol %0,0003 farkla aynı; 200 mA'da 80 kVp CTDIvol'ü kabaca üçe bölerdi,
    dolayısıyla üçü de gerçekten 120 kVp. Ad yanlış, tag doğru, ve satırlar
    tekrar olarak işaretlenmeli.
    """
    import pandas as pd

    root = device_root("USZ_PCCT")
    protocols, tags_by_series = _protocol_table(pipeline, root)
    frame = pd.DataFrame([
        dict(series_name=n, recon_kernel=p[0], k_strength=p[1], kvp=p[2],
             mas=p[3], fov=p[4], slice_thickness_mm=p[5], me_kev=p[6])
        for n, p in protocols.items()
    ])
    repeats = pipeline._assign_repeat_index(frame, tags_by_series)

    assert max(repeats.values()) == 3, "3 tekrarlı protokoller bekleniyordu"
    assert sum(1 for v in repeats.values() if v > 1) == 16, (
        "8 protokol x (3-1) = 16 tekrar serisi bekleniyordu"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_look_alike_series_are_distinct_acquisitions_not_duplicates(pipeline, device):
    """Birbirine benzeyen seriler KOPYA değil, ayrı veri olmalı.

    Üç olasılık var ve ayırt edilmeleri gerekiyor: (a) aynı dosyanın kopyası,
    (b) test-retest yeniden çekim, (c) bir tag ile ayrılmış farklı
    rekonstrüksiyon. (a) sessiz bir veri hatası olurdu — aynı ölçüm iki kez
    ortalamaya girer. Piksel düzeyinde bakıldığında hiçbir çift birebir aynı
    olmamalı.
    """
    root = device_root(device)
    protocols, _ = _protocol_table(pipeline, root)
    by_protocol = {}
    for name, protocol in protocols.items():
        by_protocol.setdefault(protocol, []).append(name)
    look_alikes = [v for v in by_protocol.values() if len(v) > 1]
    if not look_alikes:
        pytest.skip(f"{device}: aynı protokolde birden fazla seri yok")

    import pydicom

    checked = 0
    for names in look_alikes:
        digests = {}
        for name in names:
            patch = sorted((root / name / "Textures" / "Honeylarge").glob("*.dcm"))
            assert patch, f"{device}/{name}: doku patch'i yok"
            arr = pydicom.dcmread(str(patch[len(patch) // 2]), force=True).pixel_array
            digests.setdefault(arr.tobytes(), []).append(name)
        assert len(digests) == len(names), (
            f"{device}: BİREBİR AYNI piksel verisi -> {list(digests.values())}; "
            f"bunlar tekrar değil kopya, ortalamaya iki kez girerler"
        )
        checked += 1
    assert checked


# --------------------------------------------------------------------------- #
# Mavi blok — ayırıcı bilgi görünür olmalı
# --------------------------------------------------------------------------- #

def test_discriminating_tags_are_promoted_to_the_acquisition_block(pipeline):
    """Ayırt edici tag'ler mavi blokta olmalı, 150 tag arasına gömülü değil."""
    promoted = pipeline._DICOM_PROMOTED_COLUMNS
    for column in ("dcm_SeriesDescription", "dcm_SeriesNumber",
                   "dcm_AcquisitionNumber", "dcm_AcquisitionTime",
                   "dcm_CTDIvol", "dcm_(0029,1043)",
                   "dcm_(00E1,1040)_Image_Label",
                   "dcm_(01F7,109B)_iDose_Level"):
        assert column in promoted, f"{column} mavi bloğa yükseltilmemiş"
    # Zaten çözümlenmiş parametreler iki kez gösterilmemeli.
    for column in ("dcm_KVP", "dcm_Exposure", "dcm_SliceThickness",
                   "dcm_ReconstructionDiameter"):
        assert column not in promoted, f"{column} zaten kendi kolonunda"


@pytest.mark.parametrize("device", DEVICES)
def test_promoted_private_tag_names_match_what_the_writer_generates(pipeline, device):
    """Yükseltilmiş private tag adları gerçekten üretilen adlar olmalı.

    Bu adlar ``_column_for_element``'in pydicom sözlüğünden türettiği
    ``(gggg,eeee)_Name`` biçiminde. Eşleşmeyen bir ad sessizce hiçbir şey
    yapmaz — kolon sarı blokta kalır ve kimse fark etmez. Bu test o sessiz
    başarısızlığı yakalar: cihazda VAR OLAN her yükseltilmiş tag için ad
    eşleşmeli.
    """
    root = device_root(device)
    s = series_dirs(root)[0]
    generated = {f"dcm_{k}"
                 for k in pipeline.read_dicom_full_metadata(
                     pipeline._find_dicoms_dir(s))}

    expected_per_device = {
        "USZ_PCCT": {"dcm_(0029,1043)"},
        "KSW": {"dcm_(00E1,1040)_Image_Label", "dcm_(01F7,109B)_iDose_Level"},
        "EID": set(),
    }
    for column in expected_per_device[device]:
        assert column in generated, (
            f"{device}: yükseltilen {column} yazıcının ürettiği adlar arasında "
            f"yok — sessizce sarı blokta kalır"
        )
    # Cihazdan bağımsız olarak var olması gerekenler.
    for column in ("dcm_SeriesDescription", "dcm_SeriesNumber",
                   "dcm_AcquisitionNumber"):
        assert column in generated, f"{device}: {column} üretilmiyor"


# --------------------------------------------------------------------------- #
# Geometri
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", DEVICES)
def test_slice_thickness_matches_the_position_delta(pipeline, device):
    """z spacing SliceThickness'ten alınıyor; örtüşmeli rekonstrüksiyon bozar.

    Hacim ve tüm 3B doku feature'ları z spacing ile ölçekleniyor. 3 mm kalınlığı
    1,5 mm aralıkla yazan bir rekonstrüksiyon hacmi iki katına çıkarırdı.
    """
    import pydicom

    root = device_root(device)
    for s in series_dirs(root)[:4]:
        files = pipeline._list_ct_dicom_files(pipeline._find_dicoms_dir(s))[:40]
        thickness, positions = None, []
        for f in files:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            thickness = float(ds.SliceThickness)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) >= 3:
                positions.append(float(ipp[2]))
        assert thickness and len(positions) > 2, f"{device}/{s.name}: geometri yok"
        deltas = np.diff(np.sort(np.array(positions)))
        assert np.allclose(deltas, thickness, atol=1e-6), (
            f"{device}/{s.name}: SliceThickness={thickness} ama dilim aralığı "
            f"medyanı {np.median(deltas)} — hacim yanlış ölçeklenir"
        )
