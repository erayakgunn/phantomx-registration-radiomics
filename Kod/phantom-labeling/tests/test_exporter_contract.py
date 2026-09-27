"""Aşama 4 — etiketleme exporter sözleşmesi.

Bu kod radiomics'in GİRDİSİNİ üretiyor: her ROI, kırpılmış bir DICOM patch'i
olarak diske yazılıyor. Burada uydurulan her piksel değeri, aşağı akıştaki
IBSI feature'larını kirletme potansiyeli taşıyor.

Üç etiketleme ağacının (`pcct-labeling`, `pcct-sep-error-corrected`,
`acibadem-kent`) tüm .py dosyaları byte-identical; testler bu ağaçta yaşıyor ve
diğerlerinin aynı kaldığını bir hash testi doğruluyor.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pydicom
import pytest

from conftest import CODES_DIR, LABELING_ROOT, synthetic_ct_slice


# --------------------------------------------------------------------------- #
# HU <-> int16 round-trip
# --------------------------------------------------------------------------- #

def test_integer_hu_round_trips_exactly(exporter):
    """Gerçek CT verisi (slope 1, tamsayı intercept) kayıpsız dönmeli."""
    hu = np.array([[-1024.0, -1000.0, -3.0, 0.0, 42.0, 3071.0]], dtype=np.float64)
    stored = exporter._hu_to_int16(hu)
    back = stored.astype(np.float64) * 1.0 + (-1024.0)
    assert np.array_equal(back, hu)


def test_nan_becomes_the_padding_sentinel(exporter):
    hu = np.array([[10.0, np.nan, 20.0]], dtype=np.float64)
    stored = exporter._hu_to_int16(hu)
    assert stored[0, 1] == exporter._DICOM_PADDING_STORED == -32768
    # Sentinel yeniden yüklendiğinde HU -33792 oluyor.
    assert stored[0, 1] * 1.0 + (-1024.0) == -33792.0


def test_real_values_never_collide_with_the_sentinel(exporter):
    """Kırpma tabanı -32767; sentinel -32768. Aradaki tek adım, gerçek bir
    vokselin padding sanılmasını önlüyor."""
    hu = np.array([[-999999.0]], dtype=np.float64)     # absürt derecede düşük
    stored = exporter._hu_to_int16(hu)
    assert stored[0, 0] == -32767
    assert stored[0, 0] != exporter._DICOM_PADDING_STORED


def test_fractional_hu_is_truncated_not_rounded(exporter):
    """`astype(np.int16)` sıfıra doğru KIRPIYOR, yuvarlamıyor.

    Bu veri setinde etkisiz (kaynak slope=1, intercept tamsayı), ama kesirli
    intercept'li ya da birim olmayan slope'lu bir seri (spektral/VMI, materyal
    haritası) gelirse 1 HU'ya kadar kayıp olur ve kırpma yönü HU -1024'te
    işaret değiştirir.
    """
    hu = np.array([[-1023.4, -1024.6]], dtype=np.float64)
    stored = exporter._hu_to_int16(hu)
    # -1023.4 -> stored +0.6 -> 0   (aşağı)
    # -1024.6 -> stored -0.6 -> 0   (yukarı!)  => -1024 civarında süreksizlik
    assert stored[0, 0] == 0
    assert stored[0, 1] == 0


# --------------------------------------------------------------------------- #
# Yazılan DICOM'un içeriği
# --------------------------------------------------------------------------- #

def _write_patch(exporter, make_roi, tmp_path, pixels):
    path = tmp_path / "patch.dcm"
    exporter._save_roi_as_dicom(make_roi(pixels), str(path), series_name="TEST")
    return pydicom.dcmread(str(path))


def test_written_patch_declares_rescale_and_padding(exporter, make_roi, tmp_path):
    hu = np.array([[10.0, np.nan], [20.0, 30.0]], dtype=np.float64)
    ds = _write_patch(exporter, make_roi, tmp_path, hu)

    assert float(ds.RescaleSlope) == 1.0
    assert float(ds.RescaleIntercept) == -1024.0
    assert ds.RescaleType == "HU"
    assert ds.PixelPaddingValue == -32768
    assert ds.PixelRepresentation == 1        # int16 için zorunlu
    assert ds.BitsAllocated == 16


def test_written_patch_round_trips_pixel_values(exporter, make_roi, tmp_path):
    hu = np.array([[-500.0, 0.0], [250.0, np.nan]], dtype=np.float64)
    ds = _write_patch(exporter, make_roi, tmp_path, hu)

    back = ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    assert back[0, 0] == -500.0
    assert back[0, 1] == 0.0
    assert back[1, 0] == 250.0
    assert back[1, 1] == -33792.0


def test_written_patch_carries_no_spatial_geometry(exporter, make_roi, tmp_path):
    """Karakterizasyon: patch'ler kendi voksel boyutlarını taşımıyor.

    Sonuç: bu dosyalar kendi kendini tanımlamıyor. Başka bir IBSI aracı bunları
    tek başına okursa 1 mm izotropik varsayar ve tüm hacim/morfoloji
    feature'larını yanlış hesaplar. Çıkarım hattı bunu ebeveyn CT klasöründen
    telafi ediyor.
    """
    ds = _write_patch(exporter, make_roi, tmp_path, np.full((5, 5), 10.0))

    for tag in ("PixelSpacing", "SliceThickness", "ImagePositionPatient",
                "ImageOrientationPatient", "SpacingBetweenSlices",
                "FrameOfReferenceUID"):
        assert tag not in ds, (
            f"{tag} artık yazılıyor — patch'ler kendini tanımlıyor olabilir. "
            f"Bu iyi bir değişiklik; çıkarım tarafındaki _patch_spacing testi "
            f"ve README güncellenmeli."
        )


def test_slice_location_is_the_array_index_not_physical_z(exporter, make_roi, tmp_path):
    """SliceLocation fiziksel z değil, dizi indeksi.

    Dolayısıyla dilim aralığı bu etiketten türetilemez (her zaman 1,0 çıkar).
    """
    path = tmp_path / "p.dcm"
    exporter._save_roi_as_dicom(make_roi(np.full((5, 5), 10.0), slice_idx=262),
                                str(path), series_name="TEST")
    ds = pydicom.dcmread(str(path))
    assert ds.SliceLocation == 262.0
    assert ds.InstanceNumber == 262


def test_each_written_file_gets_fresh_random_uids(exporter, make_roi, tmp_path):
    """Karakterizasyon: her dilim ayrı Study VE Series UID alıyor.

    Yan etki: standart bir DICOM okuyucu (SimpleITK ImageSeriesReader, dcm2niix)
    bu dosyaları tek bir 3B seri olarak toplayamaz. Çıkarım hattı sadece klasör
    yapısına güvendiği için çalışıyor.
    """
    pixels = np.full((4, 4), 10.0)
    datasets = []
    for k in range(3):
        path = tmp_path / f"s{k}.dcm"
        exporter._save_roi_as_dicom(make_roi(pixels, slice_idx=k), str(path),
                                    series_name="TEST")
        datasets.append(pydicom.dcmread(str(path)))

    assert len({str(d.SeriesInstanceUID) for d in datasets}) == 3
    assert len({str(d.StudyInstanceUID) for d in datasets}) == 3


def test_pixel_data_is_stable_across_writes(exporter, make_roi, tmp_path):
    """Başlıklar rastgele UID ve zaman damgası taşıdığı için dosyalar asla
    byte-identical olmuyor; ama PIKSEL verisi olmalı. Altın hash testi."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(-1000, 1000, size=(9, 9)).astype(np.float64)
    pixels[0, 0] = np.nan

    hashes = set()
    for k in range(3):
        path = tmp_path / f"h{k}.dcm"
        exporter._save_roi_as_dicom(make_roi(pixels), str(path), series_name="TEST")
        hashes.add(hashlib.sha256(pydicom.dcmread(str(path)).PixelData).hexdigest())

    assert len(hashes) == 1


# --------------------------------------------------------------------------- #
# Kesiciler (cutters)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("radius_px", [4.0, 8.0, 16.0])
def test_circle_cutter_leaves_corners_as_padding(roi_extractor, radius_px):
    """Dairesel ROI kare bir dizide dönüyor; köşeler tasarım gereği NaN.

    Padding oranı, kesicinin `dx**2 + dy**2 <= r**2` tam sayı testinden analitik
    olarak çıkıyor. Elle bir bant vermek yerine aynı kuralı burada yeniden
    hesaplıyoruz, böylece test kesicinin gerçek geometrisini doğruluyor.
    """
    ct = synthetic_ct_slice(80, 80, disc_hu=120.0, radius=radius_px + 4)
    patch = roi_extractor._extract_nearest_neighbor_circle(ct, 40.0, 40.0, radius_px)

    r = int(np.ceil(radius_px))
    assert patch.shape == (2 * r + 1, 2 * r + 1)

    offsets = np.arange(-r, r + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    expected_inside = (dx ** 2 + dy ** 2 <= radius_px ** 2)

    assert np.array_equal(~np.isnan(patch), expected_inside)
    assert np.isnan(patch[0, 0]), "köşe daire dışında kalmalı"
    assert not np.isnan(patch[r, r]), "merkez gerçek veri olmalı"

    # Padding hiçbir zaman ihmal edilebilir değil: karenin köşeleri her zaman dışarıda.
    assert np.isnan(patch).mean() > 0.2


def test_square_cutter_has_no_padding_when_inside_the_image(roi_extractor):
    ct = synthetic_ct_slice(64, 64)
    patch = roi_extractor._extract_nearest_neighbor_square(ct, 32.0, 32.0, 6.0)
    assert patch.shape == (13, 13)
    assert not np.isnan(patch).any()


def test_cutters_are_deterministic(roi_extractor):
    """Aynı girdi her zaman aynı patch'i vermeli — kayan nokta merkez dahil."""
    ct = synthetic_ct_slice(64, 64)
    a = roi_extractor._extract_nearest_neighbor_circle(ct, 31.7, 32.4, 7.5)
    b = roi_extractor._extract_nearest_neighbor_circle(ct, 31.7, 32.4, 7.5)
    assert np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9))


def test_out_of_bounds_crop_is_reported_not_silent(roi_extractor, caplog):
    """Görüntü dışına taşan kırpma kutusu bildirilmeli.

    Taşan pikseller NaN kalıyor ve exporter'da dairenin köşe padding'iyle AYNI
    sentinel'e (-32768) dönüşüyor; yani aşağı akışta bir geometri hatası normal
    maske kenarından ayırt edilemiyor. Kesicinin bunu duyurması, hatanın tespit
    edilebilir olmasıyla ROI'nin sessizce küçülmesi arasındaki fark.

    Bu veri setinde fantom ortalandığı için tetiklenmiyor; fantom dedektörü hacim
    merkezine geri düştüğünde (büyük FOV / yumuşak kernel) gerçek bir hata modu.
    """
    ct = synthetic_ct_slice(64, 64)
    # Merkez sol üst köşede: kutunun yarısı görüntünün dışında.
    with caplog.at_level(logging.WARNING):
        roi_extractor._extract_nearest_neighbor_circle(ct, 2.0, 2.0, 8.0)

    assert any("leaves the" in r.message for r in caplog.records), (
        "Kırpma görüntü dışına taştı ama hiçbir uyarı verilmedi."
    )


def test_in_bounds_crop_stays_quiet(roi_extractor, caplog):
    """Görüntünün içindeki normal kırpma uyarı üretmemeli — aksi halde uyarı
    gürültüye dönüşür ve gerçek bir taşma fark edilmez."""
    ct = synthetic_ct_slice(64, 64)
    with caplog.at_level(logging.WARNING):
        roi_extractor._extract_nearest_neighbor_circle(ct, 32.0, 32.0, 8.0)
        roi_extractor._extract_nearest_neighbor_square(ct, 32.0, 32.0, 6.0)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --------------------------------------------------------------------------- #
# Üç ağaç aynı kalmalı
# --------------------------------------------------------------------------- #

def test_all_three_labeling_trees_remain_identical():
    """Testler tek ağaçta yaşıyor; diğer ikisi ondan sapmamalı.

    Saptıkları an bu testler artık onları kapsamıyor demektir.
    """
    trees = sorted(LABELING_ROOT.glob("*/**/codes/output/exporter.py"))
    if len(trees) < 2:
        pytest.skip("karşılaştırılacak ikinci ağaç yok")

    digests = {}
    for exporter_path in trees:
        codes_dir = exporter_path.parent.parent
        digest = hashlib.sha256()
        for source in sorted(codes_dir.rglob("*.py")):
            digest.update(source.relative_to(codes_dir).as_posix().encode())
            digest.update(source.read_bytes())
        digests[str(codes_dir)] = digest.hexdigest()

    unique = set(digests.values())
    assert len(unique) == 1, (
        "Etiketleme ağaçları ayrıştı:\n"
        + "\n".join(f"  {h[:12]}  {p}" for p, h in sorted(digests.items()))
    )
