"""phantomxpro-mirp — gerçek CT + TotalSegmentator maskeleri üzerinde testler.

Veri: ``~/Desktop/20260227_USZ_PCCT`` — 72 seri, her biri ct.nii.gz + seg.nii.gz
+ label_map.json. TotalSegmentator zaten çalıştırılmış olduğu için bu testler
GPU olmadan koşuyor.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import SimpleITK as sitk


# --------------------------------------------------------------------------- #
# Geometri koruması
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_every_segmentation_shares_the_ct_geometry(seg_series):
    """Maske ile görüntü aynı ızgarada olmalı.

    Pipeline bunu HİÇ doğrulamıyor: `run_totalsegmentator` yalnızca seg.nii.gz'in
    okunabildiğine bakıyor (`sitk.ReadImage` çağrısı), boyut/spacing/origin
    karşılaştırması yapmıyor. TotalSegmentator sürümü değişip farklı bir ızgara
    üretirse MIRP maskeyi sessizce yanlış hizalayabilir.

    Bugün 72/72 seri tutarlı; bu test o durumu koruyor.
    """
    mismatched = []
    for name, ct_path, seg_path, _ in seg_series:
        ct = sitk.ImageFileReader()
        ct.SetFileName(str(ct_path))
        ct.ReadImageInformation()
        seg = sitk.ImageFileReader()
        seg.SetFileName(str(seg_path))
        seg.ReadImageInformation()

        if (ct.GetSize() != seg.GetSize()
                or not np.allclose(ct.GetSpacing(), seg.GetSpacing(), atol=1e-6)
                or not np.allclose(ct.GetOrigin(), seg.GetOrigin(), atol=1e-4)):
            mismatched.append(name)

    assert not mismatched, f"CT ile maske geometrisi uyuşmayan seriler: {mismatched}"


@pytest.mark.slow
def test_label_maps_are_complete_and_consistent(seg_series):
    """Her seri aynı 117 etiketlik TotalSegmentator 'total' haritasını taşımalı.

    Harita seri başına anlık görüntüleniyor; TS sürümü değişirse etiket->isim
    eşlemesi sessizce kayabilir ve organ isimleri karışır.
    """
    reference = None
    for name, _, _, label_map_path in seg_series:
        mapping = json.loads(label_map_path.read_text(encoding="utf-8"))
        assert len(mapping) == 117, f"{name}: {len(mapping)} etiket, 117 bekleniyordu"
        if reference is None:
            reference = mapping
        else:
            assert mapping == reference, f"{name}: etiket haritası diğerlerinden farklı"

    assert reference["1"] == "spleen"


@pytest.mark.slow
def test_segmentation_labels_are_a_subset_of_the_label_map(seg_series):
    """Maskede haritada olmayan bir etiket çıkmamalı."""
    name, _, seg_path, label_map_path = seg_series[0]
    mapping = json.loads(label_map_path.read_text(encoding="utf-8"))
    labels = np.unique(sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))))

    unknown = [int(v) for v in labels if v != 0 and str(int(v)) not in mapping]
    assert not unknown, f"{name}: haritada olmayan etiketler {unknown}"


# --------------------------------------------------------------------------- #
# Resegmentation gerçek organlarda ne yapıyor?
# --------------------------------------------------------------------------- #

# Eski [-1000, 400] yumuşak doku penceresiyle ölçülen kayıp (2026-07-29):
# vertebrae_T10 %69,5 · rib_left_12 %57,5 · rib_left_11 %51,9 · rib_right_11
# %49,8 · vertebrae_T11 %38,8; buna karşılık karaciğer/dalak/autochthon %0.
# Aralık bu yüzden -1300..3100'e genişletildi.
BONE_ORGANS = ("vertebrae_T10", "vertebrae_T11", "vertebrae_L1",
               "rib_left_11", "rib_left_12", "rib_right_11")


@pytest.mark.slow
def test_resegmentation_no_longer_removes_bone(seg_series, project_config):
    """Aralık hiçbir vokseli elememeli — kemik dahil.

    Yumuşak doku penceresi 117 yapının hepsine uygulandığında kemiği tanımlayan
    yoğunlukları siliyordu; geriye kalan öznitelikler kortikal kemiği değil iliği
    tanımlıyordu. Bu test o davranışın geri gelmesini engelliyor.
    """
    lower, upper = project_config.MIRP_SETTINGS["resegmentation_intensity_range"]

    name, ct_path, seg_path, label_map_path = seg_series[0]
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
    mapping = json.loads(label_map_path.read_text(encoding="utf-8"))
    name_to_label = {v: int(k) for k, v in mapping.items()}

    checked = 0
    for organ in BONE_ORGANS:
        label = name_to_label.get(organ)
        if label is None:
            continue
        voxels = ct[seg == label]
        if voxels.size < 200:
            continue
        removed = float(((voxels < lower) | (voxels > upper)).mean() * 100.0)
        assert removed == 0.0, (
            f"{organ}: resegmentation voksellerin %{removed:.1f}'ini eliyor. "
            f"Aralık [{lower}, {upper}] kemiği kapsamıyor."
        )
        checked += 1

    assert checked > 0, "kemik organı bulunamadı — etiket haritası değişmiş olabilir"


@pytest.mark.slow
def test_resegmentation_removes_nothing_anywhere(seg_series, project_config):
    """Tüm maske genelinde sıfır kayıp — "veri değiştirilmez" ilkesinin testi."""
    lower, upper = project_config.MIRP_SETTINGS["resegmentation_intensity_range"]

    for name, ct_path, seg_path, _ in seg_series[:4]:
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
        voxels = ct[seg > 0]
        outside = int(((voxels < lower) | (voxels > upper)).sum())
        assert outside == 0, (
            f"{name}: {outside} voksel [{lower}, {upper}] dışında "
            f"(gözlenen {voxels.min():.0f}..{voxels.max():.0f} HU). "
            f"Aralık genişletilmeli ve her şey yeniden çıkarılmalı."
        )


# --------------------------------------------------------------------------- #
# Yola bağlı Monte-Carlo feature'ları
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_moran_and_geary_change_when_the_file_moves(tmp_path):
    """MIRP'in Monte-Carlo tohumu dosya yolunu içeriyor.

    Aynı voksellerden, sadece farklı bir dizinden, farklı `morph_moran_i` ve
    `morph_geary_c` çıkıyor. `apply_naming.py` çıkarımdan sonra nifti/seg
    klasörlerini yeniden adlandırdığı için, yeniden çıkarım eski değerleri
    üretmez.

    pcct hattı bu iki feature'ı atarak sorunu çözüyor; bu hat atmıyor.
    """
    from mirp import extract_features

    rng = np.random.default_rng(0)
    volume = rng.normal(40.0, 25.0, size=(16, 16, 16)).astype(np.float32)
    mask = np.zeros_like(volume, dtype=np.uint8)
    mask[3:13, 3:13, 3:13] = 1                       # 1000 voksel: MC eşiği
    assert int(mask.sum()) >= 1000

    def run(directory):
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / "img.nii.gz"
        mask_path = directory / "msk.nii.gz"
        for array, path, dtype in ((volume, image_path, np.float32),
                                   (mask, mask_path, np.uint8)):
            image = sitk.GetImageFromArray(array.astype(dtype))
            image.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(image, str(path))

        frame = extract_features(
            image=str(image_path), mask=str(mask_path), roi_name={"1": "roi"},
            image_modality="CT", base_feature_families="morphology",
            ibsi_compliant=True,
        )[0]
        return (float(frame["morph_moran_i"].iloc[0]),
                float(frame["morph_geary_c"].iloc[0]))

    here = run(tmp_path / "location_a")
    there = run(tmp_path / "location_b")
    again = run(tmp_path / "location_a")

    assert here == again, "aynı yolda tekrar farklı sonuç verdi — beklenmiyor"
    assert here != there, (
        "Moran/Geary artık yola bağlı değil — MIRP davranışı değişmiş olabilir. "
        "Öyleyse bu iki feature güvenle raporlanabilir ve test güncellenmeli."
    )
