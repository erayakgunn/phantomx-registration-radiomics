"""Aşama 2 — padding / sentinel sözleşmesi.

Kullanıcının asıl sorusu buydu: "padding için yeni piksel verisi uyduruyor mu?"

Cevap: etiketleme exporter'ı uyduruyor (gerçek insert patch'lerinin %38-50'si
HU -33792 sentinel'i), ama çıkarım hattı bunları MASKELEMEK yerine KIRPARAK
atıyor, dolayısıyla uydurulmuş değer MIRP'e hiç girmiyor.

Bu koruma tek bir büyüklük eşiğine (`hu > _PADDING_HU_FLOOR`) dayanıyor ve
iki ayrı proje arasında paylaşılan yazısız bir sözleşme. Buradaki testler o
sözleşmeyi kalıcı hale getiriyor.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conftest import (labeling_project, make_disc_patch_dir, make_patch_dicom,
                      series_dirs)

# Exporter'ın stored padding değeri, RescaleSlope=1 / Intercept=-1024 altında.
SENTINEL_HU = -32768 * 1.0 + (-1024.0)          # = -33792.0


# --------------------------------------------------------------------------- #
# Çapraz proje sözleşmesi — sessizce kırılması en muhtemel seam
# --------------------------------------------------------------------------- #

def _exporter_source() -> str:
    """Etiketleme exporter'ının kaynağı.

    Yol `labeling_project()` üzerinden çözülüyor: sabit `../../labeling` yolu
    proje yeniden adlandırıldığında (`phantom-labeling`) sessizce kaybolmuş ve
    bu iki test aylarca atlanmış olabilir.
    """
    root = labeling_project()
    matches = sorted(root.glob("**/codes/output/exporter.py"))
    if not matches:
        pytest.skip(f"{root} içinde codes/output/exporter.py yok")
    return matches[0].read_text(encoding="utf-8")


def _constant(source: str, name: str) -> float:
    match = re.search(rf"^{name}\s*=\s*(-?[\d.]+)", source, re.MULTILINE)
    assert match, f"{name} exporter kaynağında bulunamadı"
    return float(match.group(1))


def test_extractor_floor_still_separates_exporter_padding(pipeline):
    """Exporter'ın sentinel'i ile çıkarımın eşiği uyumlu kalmalı.

    Bu iki sabit iki AYRI projede yaşıyor ve birbirlerinden habersiz. Exporter
    sentinel'i değiştirirse (ya da intercept'i), çıkarım sessizce uydurma
    değerleri gerçek voksel sanmaya başlar.
    """
    source = _exporter_source()
    stored = _constant(source, "_DICOM_PADDING_STORED")
    slope = _constant(source, "_DICOM_RESCALE_SLOPE")
    intercept = _constant(source, "_DICOM_RESCALE_INTERCEPT")

    sentinel_hu = stored * slope + intercept
    assert sentinel_hu == SENTINEL_HU

    floor = pipeline._PADDING_HU_FLOOR
    assert sentinel_hu < floor, "sentinel eşiğin altında kalmalı"
    # Eşik gerçek CT değerlerinin de altında olmalı, yoksa gerçek voksel eler.
    assert floor < -1200, "eşik hava HU'sunun (-1000 civarı) güvenle altında olmalı"


def test_exporter_clip_floor_does_not_collide_with_the_sentinel():
    """Exporter gerçek HU'ları -32767'ye kırpıyor, sentinel ise -32768.

    Aradaki tek adımlık pay, gerçek bir vokselin sentinel'e dönüşmesini önlüyor.
    """
    source = _exporter_source()
    assert "np.clip(stored_valid, -32767, 32767)" in source
    assert _constant(source, "_DICOM_PADDING_STORED") == -32768


# --------------------------------------------------------------------------- #
# Sentinel MIRP'e girmiyor
# --------------------------------------------------------------------------- #

def test_disc_patch_is_cropped_free_of_the_sentinel(pipeline, tmp_path):
    """Dairesel ROI'de köşe padding'i kırpılarak atılıyor, maskelenmiyor."""
    patch_dir = make_disc_patch_dir(tmp_path / "Insert_01", size=19, n_slices=4)
    hu, mask, _ = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))

    assert not np.any(hu <= pipeline._PADDING_HU_FLOOR), "sentinel görüntüye sızdı"
    assert hu.min() > -1000, "kırpılmış hacimde uydurma değer kalmamalı"
    # Kare, orijinal patch'ten küçük olmalı — kırpma gerçekten oldu.
    assert hu.shape[1] < 19 and hu.shape[2] < 19


def test_square_patch_passes_through_untouched(pipeline, tmp_path):
    """Texture kareleri padding içermez; hiç kırpılmadan geçmeli."""
    patch_dir = tmp_path / "Honeylarge"
    for k in range(4):
        hu = np.full((21, 21), 40.0 + k)
        make_patch_dicom(patch_dir / f"slice_{k:04d}.dcm", hu, slice_index=k)

    out, mask, _ = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))
    assert out.shape == (4, 21, 21), "kare patch kırpılmamalı"
    assert mask.all()


def test_mask_is_all_ones_not_a_segmentation(pipeline, tmp_path):
    """Maske ROI değil, kırpma kutusu.

    Karakterizasyon testi: bu 1,0'lardan çıkarsa geometri sözleşmesi değişmiş
    demektir ve morfoloji feature'larının anlamı da değişmiştir.
    """
    patch_dir = make_disc_patch_dir(tmp_path / "Insert_01", size=15, n_slices=3)
    hu, mask, _ = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))
    assert mask.shape == hu.shape
    assert np.array_equal(np.unique(mask), np.array([1], dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Kenar durumlar
# --------------------------------------------------------------------------- #

def test_all_padding_patch_is_refused(pipeline, tmp_path):
    """Tamamı sentinel olan bir patch ölçüm olarak yayımlanamaz."""
    patch_dir = tmp_path / "Insert_dead"
    for k in range(3):
        make_patch_dicom(patch_dir / f"slice_{k:04d}.dcm",
                         np.full((9, 9), np.nan), slice_index=k)

    with pytest.raises(ValueError, match="out-of-ROI padding"):
        pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))


def test_single_real_voxel_survives(pipeline, tmp_path):
    """Tek gerçek vokselli patch 1x1 kareye inmeli, sessizce boşalmamalı."""
    patch_dir = tmp_path / "Insert_tiny"
    for k in range(3):
        hu = np.full((7, 7), np.nan)
        hu[3, 3] = 120.0
        make_patch_dicom(patch_dir / f"slice_{k:04d}.dcm", hu, slice_index=k)

    hu_out, mask, _ = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))
    assert hu_out.shape == (3, 1, 1)
    assert hu_out.min() == 120.0


def test_inconsistent_slice_shapes_raise(pipeline, tmp_path):
    patch_dir = tmp_path / "Insert_ragged"
    make_patch_dicom(patch_dir / "slice_0000.dcm", np.full((9, 9), 30.0), slice_index=0)
    make_patch_dicom(patch_dir / "slice_0001.dcm", np.full((11, 11), 30.0), slice_index=1)

    with pytest.raises(ValueError, match="Inconsistent patch shapes"):
        pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))


def test_missing_rescale_raises_instead_of_assuming(pipeline, tmp_path):
    """HU türetilemiyorsa varsayılan uydurulmamalı — hata yükselmeli."""
    patch_dir = tmp_path / "Insert_norescale"
    for k in range(2):
        make_patch_dicom(patch_dir / f"slice_{k:04d}.dcm", np.full((9, 9), 30.0),
                         slice_index=k, include_rescale=False)

    with pytest.raises(ValueError, match="RescaleSlope|RescaleIntercept"):
        pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))


def test_empty_directory_raises(pipeline, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        pipeline.read_insert_patch_dicom_volume(empty, (0.6, 0.6, 1.0))


# --------------------------------------------------------------------------- #
# Kırpma determinizmi
# --------------------------------------------------------------------------- #

def test_largest_inscribed_square_is_deterministic(pipeline):
    """Aynı footprint her zaman aynı kareyi vermeli.

    Simetrik bir footprint'te birden çok aday kare eşit büyüklüktedir; seçim
    merkeze yakınlığa göre yapılıyor. Bu belirsizlik çözülmezse aynı veri
    makineler arasında farklı voksel kümesi üretebilirdi.
    """
    footprint = np.zeros((11, 11), dtype=bool)
    footprint[2:9, 2:9] = True

    results = {pipeline._largest_inscribed_square(footprint) for _ in range(20)}
    assert len(results) == 1

    r0, c0, size = results.pop()
    assert size == 7
    assert footprint[r0:r0 + size, c0:c0 + size].all()


def test_largest_inscribed_square_picks_the_centred_candidate(pipeline):
    """Eşit büyüklükte iki aday varsa merkeze yakın olan seçilmeli."""
    footprint = np.zeros((5, 9), dtype=bool)
    footprint[1:4, 1:8] = True          # 3 satır yüksek, 7 sütun geniş
    r0, c0, size = pipeline._largest_inscribed_square(footprint)
    assert size == 3
    assert footprint[r0:r0 + size, c0:c0 + size].all()
    # Merkezî aday: sütunlar 1..7 arasında ortada olan.
    assert c0 == 3


# --------------------------------------------------------------------------- #
# Gerçek veri regresyonu
# --------------------------------------------------------------------------- #

# Ölçülen değerler (2026-07-29, 20260402_USZ_PCCT).
# (FOV, ROI): (patch kenarı, gerçek voksel, kırpma sonrası voksel)
REAL_RETENTION = {
    ("_300_", "Inserts/5mm/Insert_01"):  (11, 1830, 1470),
    ("_300_", "Inserts/10mm/Insert_01"): (19, 6750, 5070),
    ("_300_", "Stars/Star16"):           (49, 53670, 32670),
    ("_300_", "Textures/Honeylarge"):    (49, 72030, 72030),
    ("_500_", "Inserts/5mm/Insert_01"):  (7, 630, 270),
    ("_500_", "Inserts/10mm/Insert_01"): (13, 2670, 1470),
    ("_500_", "Stars/Star16"):           (31, 19710, 13230),
    ("_500_", "Textures/Honeylarge"):    (29, 25230, 25230),
}


@pytest.mark.slow
@pytest.mark.parametrize("key", list(REAL_RETENTION), ids=lambda k: f"{k[0]}{k[1]}")
def test_real_data_voxel_retention_is_stable(pipeline, real_dataset, key):
    """Kırpmanın gerçek veride ne kadar voksel attığını sabitler.

    FOV 500'de 5 mm insert 630 gerçek vokselden 270'e (3x3x30 kutu) iniyor.
    Bu sayılar değişirse geometri sözleşmesi değişmiş demektir ve daha önce
    üretilmiş tüm feature'lar karşılaştırılamaz hale gelir.
    """
    fov, rel = key
    expected_side, expected_real, expected_kept = REAL_RETENTION[key]

    series = [p for p in series_dirs(real_dataset) if fov in p.name]
    if not series:
        pytest.skip(f"FOV {fov} serisi yok")
    patch_dir = series[0] / rel
    if not any(patch_dir.glob("*.dcm")):
        pytest.skip(f"{patch_dir} boş")

    import pydicom
    files = sorted(patch_dir.glob("*.dcm"))
    first = pydicom.dcmread(str(files[0]), force=True).pixel_array
    assert first.shape == (expected_side, expected_side)

    real_voxels = len(files) * int((first != -32768).sum())
    assert real_voxels == expected_real

    hu, mask, _ = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.5859375, 0.5859375, 1.0))
    assert hu.size == expected_kept
    assert not np.any(hu <= pipeline._PADDING_HU_FLOOR)


@pytest.mark.slow
def test_real_texture_patches_carry_no_sentinel(pipeline, real_dataset):
    """Texture kareleri sentinel içermemeli.

    Önemli: çıkarım hattı `real.all()` olan patch'leri hiç kırpmadan geçiriyor.
    Bir texture patch'ine sınır dışı taşma yüzünden tek bir sentinel girse,
    HU -33792 doğrudan MIRP'e giderdi. Bu test o sessiz bozulmayı yakalar.
    """
    import pydicom

    checked = 0
    for series in series_dirs(real_dataset):
        textures = series / "Textures"
        if not textures.is_dir():
            continue
        for pattern in sorted(textures.iterdir()):
            files = sorted(pattern.glob("*.dcm"))
            if not files:
                continue
            arr = pydicom.dcmread(str(files[0]), force=True).pixel_array
            assert not (arr == -32768).any(), (
                f"{pattern} içinde sentinel var — kırpılmadan MIRP'e gider"
            )
            checked += 1
    assert checked > 0, "hiç texture patch'i bulunamadı"
