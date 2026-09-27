"""Aşama 3 — geometri ve maske dürüstlüğü.

Voksel boyutu her hacim ve morfoloji feature'ını doğrudan ölçekler, dolayısıyla
spacing'in nereden geldiği IBSI raporlaması açısından birinci derece önemli.

Buradaki testlerin bir kısmı "bug testi" değil **karakterizasyon** testi:
bugünkü davranışı sabitliyorlar ki, biri değiştirdiğinde bunun bilinçli bir
karar olduğu görülsün.
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from conftest import make_disc_patch_dir, make_patch_dicom, series_dirs


# --------------------------------------------------------------------------- #
# Patch'ler kendi geometrilerini taşımıyor
# --------------------------------------------------------------------------- #

def test_patch_spacing_falls_back_when_tags_are_absent(pipeline, tmp_path):
    """Exporter patch'lere PixelSpacing/SliceThickness yazmıyor.

    Sonuç: `_patch_spacing` HER ZAMAN fallback dalını kullanıyor. Modülün
    "hiçbir değer asla ikame edilmez" doktrini bu yolda fiilen çalışmıyor —
    bugün doğru sonuç veriyor çünkü fallback ebeveyn CT'den geliyor, ama
    garanti kodda değil, çağıranın disiplininde.
    """
    patch_dir = make_disc_patch_dir(tmp_path / "Insert_01", size=13, n_slices=3)

    sentinel_spacing = (0.123, 0.123, 7.5)
    _, _, spacing = pipeline.read_insert_patch_dicom_volume(patch_dir, sentinel_spacing)

    assert spacing == sentinel_spacing, (
        "Patch'ten spacing okunabildi — exporter artık geometri etiketi yazıyor "
        "olmalı. Bu iyi bir değişiklik ama testin ve README'nin güncellenmesi gerekir."
    )


def test_patch_spacing_prefers_real_tags_when_present(pipeline, tmp_path):
    """Etiket varsa fallback yerine gerçek değer kullanılmalı."""
    import pydicom

    patch_dir = tmp_path / "Insert_tagged"
    for k in range(3):
        hu = np.full((11, 11), 45.0)
        make_patch_dicom(patch_dir / f"slice_{k:04d}.dcm", hu, slice_index=k)
    for path in sorted(patch_dir.glob("*.dcm")):
        ds = pydicom.dcmread(str(path))
        ds.PixelSpacing = [0.75, 0.75]
        ds.SliceThickness = 2.0
        ds.save_as(str(path))

    _, _, spacing = pipeline.read_insert_patch_dicom_volume(patch_dir, (9.9, 9.9, 9.9))
    assert spacing == (0.75, 0.75, 2.0)


@pytest.mark.slow
@pytest.mark.parametrize("fov,expected_xy", [("_300_", 0.5859375),
                                             ("_400_", 0.78125),
                                             ("_500_", 0.9765625)])
def test_real_parent_ct_spacing_varies_with_fov(real_dataset, fov, expected_xy):
    """Gerçek spacing FOV ile değişiyor; fallback bunu doğru taşımalı.

    FOV 300'de voksel 1,71x anizotropik, FOV 500'de neredeyse izotropik.
    İzotropik yeniden örnekleme yapılmadığı için 3B doku feature'larının
    yön yanlılığı FOV'a göre DEĞİŞİYOR — FOV karşılaştırması bu yüzden
    geometri artefaktıyla karışıyor.
    """
    import pydicom

    series = sorted(p for p in real_dataset.iterdir() if p.is_dir() and fov in p.name)
    if not series:
        pytest.skip(f"FOV {fov} serisi yok")

    files = sorted((series[0] / "dicoms").rglob("*.dcm"))
    assert files, "ebeveyn CT bulunamadı"
    ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)

    assert float(ds.PixelSpacing[0]) == pytest.approx(expected_xy)
    assert float(ds.SliceThickness) == pytest.approx(1.0)


@pytest.mark.slow
def test_real_slice_thickness_matches_image_position_delta(real_dataset):
    """z spacing SliceThickness'ten okunuyor, ImagePositionPatient farkından değil.

    Bu veri setinde ikisi birebir aynı (1,0 mm), dolayısıyla bugün hata yok.
    Test, örtüşmeli/boşluklu bir rekonstrüksiyon geldiğinde sessizce yanlış
    hacim hesaplanmasını yakalamak için var.
    """
    import pydicom

    # series_dirs() skips ``radiomics_output``. The pipeline writes its output
    # INSIDE the dataset root by default, and pathlib compares Windows paths
    # case-insensitively, so "radiomics_output" sorts BEFORE "USZ_..." — a plain
    # ``sorted(root.iterdir())[0]`` picks the output folder and finds no DICOMs.
    # The pipeline itself is immune (_try_series rejects it); this test was not.
    series = series_dirs(real_dataset)
    assert series
    files = sorted((series[0] / "dicoms").rglob("*.dcm"))[:60]
    assert len(files) > 2

    thickness = None
    positions = []
    for path in files:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        thickness = float(ds.SliceThickness)
        if hasattr(ds, "ImagePositionPatient"):
            positions.append(float(ds.ImagePositionPatient[2]))

    deltas = np.diff(np.sort(np.array(positions)))
    assert np.allclose(deltas, thickness, atol=1e-6), (
        f"SliceThickness={thickness} ile gerçek dilim aralığı "
        f"(medyan {np.median(deltas)}) uyuşmuyor — hacim ve tüm 3B doku "
        f"feature'ları yanlış ölçeklenir."
    )


# --------------------------------------------------------------------------- #
# NIfTI geometrisi
# --------------------------------------------------------------------------- #

def test_written_nifti_keeps_spacing_but_drops_patient_space(pipeline, tmp_path):
    """write_roi_nifti sadece spacing yazıyor; origin/direction varsayılan kalıyor.

    Feature değerleri açısından sorun değil (görüntü ile maske aynı çerçevede),
    ama NIfTI artık hasta uzayında değil. Karakterizasyon testi.
    """
    hu = np.linspace(-200, 200, 3 * 5 * 5, dtype=np.float32).reshape(3, 5, 5)
    mask = np.ones_like(hu, dtype=np.uint8)
    image_path = tmp_path / "img.nii.gz"
    mask_path = tmp_path / "msk.nii.gz"

    pipeline.write_roi_nifti(hu, mask, (0.6, 0.6, 1.5), image_path, mask_path)

    img = sitk.ReadImage(str(image_path))
    msk = sitk.ReadImage(str(mask_path))

    assert img.GetSpacing() == pytest.approx((0.6, 0.6, 1.5))
    assert img.GetSpacing() == msk.GetSpacing()
    assert img.GetSize() == msk.GetSize()
    # Hasta uzayı bilgisi taşınmıyor:
    assert img.GetOrigin() == pytest.approx((0.0, 0.0, 0.0))
    assert img.GetDirection() == pytest.approx((1, 0, 0, 0, 1, 0, 0, 0, 1))


def test_image_and_mask_stay_aligned(pipeline, tmp_path):
    """Görüntü ve maske aynı ızgarada olmalı — MIRP hizalamayı varsayıyor."""
    patch_dir = make_disc_patch_dir(tmp_path / "Insert_01", size=15, n_slices=4)
    hu, mask, spacing = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))

    image_path = tmp_path / "img.nii.gz"
    mask_path = tmp_path / "msk.nii.gz"
    pipeline.write_roi_nifti(hu, mask, spacing, image_path, mask_path)

    img = sitk.ReadImage(str(image_path))
    msk = sitk.ReadImage(str(mask_path))
    assert img.GetSize() == msk.GetSize()
    assert img.GetSpacing() == msk.GetSpacing()
    assert img.GetOrigin() == msk.GetOrigin()
    assert img.GetDirection() == msk.GetDirection()


# --------------------------------------------------------------------------- #
# Dilim sıralaması
# --------------------------------------------------------------------------- #

def test_slices_are_stacked_in_ascending_order(pipeline, tmp_path):
    """Dosya adı sırası ile dilim sırası ayrışsa bile z sırası korunmalı."""
    patch_dir = tmp_path / "Insert_shuffled"
    # Dosya adları kasten ters sırada numaralanıyor.
    for name_index, slice_index in enumerate([30, 10, 20]):
        hu = np.full((9, 9), float(slice_index))
        make_patch_dicom(patch_dir / f"zz_{name_index}.dcm", hu, slice_index=slice_index)

    hu_out, _, _ = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))
    first_voxel_per_slice = hu_out[:, 0, 0]
    assert list(first_voxel_per_slice) == [10.0, 20.0, 30.0]


# --------------------------------------------------------------------------- #
# Morfoloji feature'ları kutuyu ölçüyor
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_morphology_describes_the_crop_box_not_the_insert(pipeline, tmp_path):
    """Maske tümü-1 bir dikdörtgen prizma olduğu için morfoloji ailesi
    fantomu değil kırpma kutusunu ölçüyor.

    `morph_vol_dens_conv_hull` her ROI'de tam 1,0 çıkıyor: bir kutunun konveks
    örtüsü kendisidir. Bu, morfoloji feature'larının bu veri setinde fiziksel
    bir anlam taşımadığının kanıtı ve yayında beyan edilmesi gerekir.
    """
    from mirp import extract_features

    patch_dir = make_disc_patch_dir(tmp_path / "Insert_01", size=25, n_slices=12)
    hu, mask, spacing = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))

    image_path = tmp_path / "img.nii.gz"
    mask_path = tmp_path / "msk.nii.gz"
    pipeline.write_roi_nifti(hu, mask, spacing, image_path, mask_path)

    params = pipeline.get_3d_base_params()
    params["base_feature_families"] = "morphology"
    frame = extract_features(image=str(image_path), mask=str(mask_path),
                             roi_name={"1": "roi"}, image_modality="CT", **params)[0]

    assert float(frame["morph_vol_dens_conv_hull"].iloc[0]) == pytest.approx(1.0, abs=1e-6)

    # Hacim, kutunun geometrisinden analitik olarak çıkıyor.
    expected_volume = hu.size * spacing[0] * spacing[1] * spacing[2]
    assert float(frame["morph_volume"].iloc[0]) == pytest.approx(expected_volume, rel=0.05)
