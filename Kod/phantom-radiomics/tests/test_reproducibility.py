"""Aşama 5 — tekrarlanabilirlik.

Feature değerleri aynı makinede aynı girdi için tekrar üretilebiliyor. Kırılgan
olan taraf zaman içinde ve makineler arasında: sürüm sabitlemesi yok, checkpoint
parametre değişimini yakalamıyor, ve MIRP'in bazı feature'ları dosya yoluna
bağlı.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from conftest import PIPELINE_PATH, make_disc_patch_dir


# --------------------------------------------------------------------------- #
# Yola bağlı feature'lar gerçekten atılıyor mu?
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_path_dependent_features_are_actually_dropped(pipeline, tmp_path):
    """MIRP Moran's I ve Geary's C'yi 1000+ voksellik ROI'lerde Monte-Carlo ile
    kestiriyor ve tohumu dosya yolunu içeren bir hash'ten alıyor.

    Pipeline bu ikisini `_DROPPED_FEATURES` ile atıyor. Bu test, atma mantığının
    MIRP'in GERÇEK sütun adlarına karşı çalıştığını doğruluyor — sütun adları
    sonek alırsa (ör. discretisation etiketi) basit eşitlik kontrolü kaçırırdı.
    """
    from mirp import extract_features

    patch_dir = make_disc_patch_dir(tmp_path / "Insert_01", size=25, n_slices=12)
    hu, mask, spacing = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))
    assert hu.size >= 1000, "Monte-Carlo dalını tetiklemek için 1000+ voksel gerekli"

    image_path, mask_path = tmp_path / "img.nii.gz", tmp_path / "msk.nii.gz"
    pipeline.write_roi_nifti(hu, mask, spacing, image_path, mask_path)

    params = pipeline.get_3d_base_params()
    params["base_feature_families"] = "morphology"
    frame = extract_features(image=str(image_path), mask=str(mask_path),
                            roi_name={"1": "roi"}, image_modality="CT", **params)[0]

    assert any(c.startswith("morph_moran_i") for c in frame.columns), (
        "MIRP artık morph_moran_i üretmiyor — atma listesi güncellenmeli"
    )

    cleaned = pipeline._drop_nonreproducible(frame)
    for dropped in ("morph_moran_i", "morph_geary_c"):
        assert not any(c.startswith(dropped) for c in cleaned.columns)


def test_dropped_feature_list_covers_the_documented_reasons(pipeline):
    """Atılan feature'lar üç gerekçeye ayrılıyor; hepsi listede olmalı."""
    dropped = set(pipeline._DROPPED_FEATURES)
    assert {"morph_moran_i", "morph_geary_c"} <= dropped        # yola bağlı
    assert {"loc_peak_loc", "loc_peak_glob"} <= dropped         # uydurma sınır
    assert {"ivh_v10", "ivh_v90", "ivh_auc"} <= dropped         # bozuk IVH ekseni


def test_drop_is_prefix_aware_not_exact_match_only(pipeline):
    """Sonekli varyantlar da atılmalı (ör. `_fbs_w25.0`)."""
    frame = pd.DataFrame({
        "morph_moran_i": [1.0],
        "morph_geary_c_extra_suffix": [2.0],
        "morph_volume": [3.0],
        "morph_moran_index_lookalike": [4.0],
    })
    cleaned = pipeline._drop_nonreproducible(frame)
    assert "morph_moran_i" not in cleaned.columns
    assert "morph_geary_c_extra_suffix" not in cleaned.columns
    # Yanlışlıkla fazlasını atmamalı:
    assert "morph_volume" in cleaned.columns
    assert "morph_moran_index_lookalike" in cleaned.columns


# --------------------------------------------------------------------------- #
# Determinizm
# --------------------------------------------------------------------------- #

def test_no_random_number_generation_in_the_pipeline():
    """Pipeline'ın kendisi rastgelelik içermemeli."""
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    forbidden = [r"\bnp\.random\b", r"\brandom\.(random|choice|shuffle|sample)\b",
                 r"\bshuffle\("]
    for pattern in forbidden:
        assert not re.search(pattern, source), f"rastgelelik bulundu: {pattern}"


def test_patch_reading_is_bitwise_repeatable(pipeline, tmp_path):
    """Aynı patch klasörü iki kez okunduğunda birebir aynı diziyi vermeli."""
    patch_dir = make_disc_patch_dir(tmp_path / "Insert_01", size=17, n_slices=5)

    first = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))
    second = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[0].tobytes() == second[0].tobytes()


@pytest.mark.slow
def test_feature_values_are_identical_across_repeated_extractions(pipeline, tmp_path):
    """Aynı NIfTI'den iki kez çıkarım birebir aynı sayıları vermeli.

    Yola bağlı feature'lar atıldıktan sonra geriye kalan her şey deterministik
    olmalı. Bu, `--jobs` değerinin feature DEĞERLERİNİ etkilemediğinin de
    dolaylı kanıtı: seriler birbirinden bağımsız işleniyor.
    """
    from mirp import extract_features

    patch_dir = make_disc_patch_dir(tmp_path / "Insert_01", size=21, n_slices=10)
    hu, mask, spacing = pipeline.read_insert_patch_dicom_volume(patch_dir, (0.6, 0.6, 1.0))
    image_path, mask_path = tmp_path / "img.nii.gz", tmp_path / "msk.nii.gz"
    pipeline.write_roi_nifti(hu, mask, spacing, image_path, mask_path)

    params = pipeline.get_3d_base_params()

    def run():
        frame = extract_features(image=str(image_path), mask=str(mask_path),
                                roi_name={"1": "roi"}, image_modality="CT", **params)[0]
        return pipeline._drop_nonreproducible(frame)

    first, second = run(), run()
    numeric = [c for c in first.columns if first[c].dtype.kind == "f"]
    assert numeric, "sayısal feature bulunamadı"

    for column in numeric:
        a, b = float(first[column].iloc[0]), float(second[column].iloc[0])
        assert (np.isnan(a) and np.isnan(b)) or a == b, f"{column} tekrarda değişti"


# --------------------------------------------------------------------------- #
# Checkpoint — BUGÜN BAŞARISIZ OLUR (düzeltmenin şartnamesi)
# --------------------------------------------------------------------------- #

def _checkpoint_rows():
    return [{
        "series_name": "S1", "mode": "3D", "group": "Inserts",
        "label": "Insert_01", "insert_size": "10mm",
        "feature_id": "stat_mean", "family": "Statistics",
        "source": "base", "value": 42.0, "voxel_count": 5070,
    }]


def test_checkpoint_is_reusable_for_an_unchanged_pipeline(pipeline, tmp_path):
    """Kontrol testi: hiçbir şey değişmediyse checkpoint yeniden kullanılabilmeli."""
    path = tmp_path / "S1.xlsx"
    pipeline.save_checkpoint(_checkpoint_rows(), path)
    reusable, reason = pipeline.checkpoint_is_reusable(path, ["3D"])
    assert reusable, reason


def test_checkpoint_is_rejected_when_the_discretisation_anchor_changes(
    pipeline, tmp_path, monkeypatch
):
    """BUGÜN BAŞARISIZ OLUR — düzeltmenin şartnamesi.

    `checkpoint_is_reusable` yalnızca şema sürümünü, atılan feature listesini ve
    mode'ları doğruluyor. Çıkarım PARAMETRELERİ damgalanmıyor.

    Sonuç: `DISCRETISATION_RANGE_HU` veya bin genişliği değiştirilirse eski
    satırlar sessizce yeniden kullanılıyor ve tek bir Excel iki farklı
    parametrelendirmeyi karıştırıyor. Ölçüldü: çapayı -2000'den -1000/400'e
    taşımak 218 feature'ın 191'ini değiştiriyor — yani karışan satırlar
    birbiriyle kıyaslanamaz.

    Beklenen düzeltme: save_checkpoint parametre setinin bir hash'ini yazsın,
    checkpoint_is_reusable onu karşılaştırsın.
    """
    path = tmp_path / "S1.xlsx"
    pipeline.save_checkpoint(_checkpoint_rows(), path)

    monkeypatch.setattr(pipeline, "DISCRETISATION_RANGE_HU", (-1000.0, 400.0))

    reusable, reason = pipeline.checkpoint_is_reusable(path, ["3D"])
    assert not reusable, (
        "Checkpoint, discretisation çapası değiştiği halde geçerli sayıldı. "
        "Parametreler checkpoint'e damgalanmadığı için eski ve yeni satırlar "
        "aynı çalışma kitabında karışabilir."
    )


def test_checkpoint_rejects_a_different_reported_feature_set(pipeline, tmp_path, monkeypatch):
    """Bu koruma VAR: atılan feature listesi değişirse checkpoint reddediliyor."""
    path = tmp_path / "S1.xlsx"
    pipeline.save_checkpoint(_checkpoint_rows(), path)

    monkeypatch.setattr(pipeline, "_DROPPED_FEATURES",
                        pipeline._DROPPED_FEATURES + ("stat_mean",))
    reusable, reason = pipeline.checkpoint_is_reusable(path, ["3D"])
    assert not reusable
    assert "feature set" in reason


def test_checkpoint_rejects_a_missing_mode(pipeline, tmp_path):
    """3D-only bir checkpoint, 3D+2D isteyen bir çalışmayı karşılamamalı."""
    path = tmp_path / "S1.xlsx"
    pipeline.save_checkpoint(_checkpoint_rows(), path)
    reusable, reason = pipeline.checkpoint_is_reusable(path, ["3D", "2D"])
    assert not reusable
    assert "2D" in reason
