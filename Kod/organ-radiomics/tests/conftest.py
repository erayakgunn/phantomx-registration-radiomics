"""phantomxpro-mirp IBSI/tekrarlanabilirlik testleri için altyapı.

Mevcut `test_pipeline.py` proje kökünde duruyor ve keşif/isimlendirme/paralellik
tarafını kapsıyor. Bu dizindeki testler yalnızca IBSI parametreleri ve
tekrarlanabilirlik boşluklarını hedefliyor; oradaki testleri tekrarlamıyor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Gerçek organ segmentasyonu veri seti: ct.nii.gz + seg.nii.gz + label_map.json
# TotalSegmentator zaten çalıştırılmış, dolayısıyla GPU olmadan test edilebilir.
#
# Sıra önemli: sabit bir makine-yoluna bağlıyken bu testler sessizce ATLANIYORDU
# (yol taşınmış/yeniden adlandırılmıştı) — yani gerçek maske doğrulaması hiç
# koşmuyordu ve takım "geçti" diyordu. Artık önce açık override, sonra projenin
# KENDİ çıktısı denenir: pipeline bir kez koştuktan sonra bu testler kendi
# ürettiğimiz maskelere karşı çalışır.
_LEGACY_DATASET = Path.home() / "Desktop" / "20260227_USZ_PCCT"

# Segmentasyon klasörünün adı değişti (segmentations -> segmentation); eski
# çıktılar da bulunabilsin diye ikisi de kabul edilir.
_SEG_DIRS = ("segmentation", "segmentations")


def _seg_dir(root: Path) -> Path | None:
    for name in _SEG_DIRS:
        if (root / name).is_dir():
            return root / name
    return None


def _resolve_seg_dataset() -> Path:
    import os

    env = os.environ.get("PHANTOMX_SEG_DATASET")
    candidates = [Path(env)] if env else []
    # Çıktı artık KAYNAK klasörlerin içinde; gerçek maske testleri oradan besleniyor.
    try:
        from run_and_verify import DEFAULT_SOURCES
        candidates.extend(DEFAULT_SOURCES)
    except Exception:  # noqa: BLE001
        pass
    candidates.append(PROJECT_DIR / "data")     # eski proje-içi çıktı
    candidates.append(_LEGACY_DATASET)
    for candidate in candidates:
        if (candidate / "nifti").is_dir() and _seg_dir(candidate) is not None:
            return candidate
    return candidates[0] if candidates else _LEGACY_DATASET


REAL_SEG_DATASET = _resolve_seg_dataset()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: gerçek veri veya tam MIRP çalışması gerekir")


def resolve_mirp_settings(**kwargs) -> dict:
    """Flatten MIRP's fully-resolved settings into a comparable dict.

    The point is to surface the parameters config.py never states — MIRP fills
    them from its own defaults, and they govern the result just as much as the
    ones that are written down. Kept here rather than imported from the sibling
    phantom pipeline so this folder stands on its own.
    """
    from mirp.settings.import_config_parameters import import_configuration_settings

    s = import_configuration_settings(compute_features=True, **kwargs)[0]

    def spacing(value):
        if value is None:
            return None
        return tuple(tuple(float(x) for x in row) for row in value)

    return {
        "by_slice": s.general.by_slice,
        "ibsi_compliant": s.general.ibsi_compliant,
        "no_approximation": s.general.no_approximation,
        "interpolate": s.img_interpolate.interpolate,
        "new_spacing": spacing(s.img_interpolate.new_spacing),
        "spline_order": s.img_interpolate.spline_order,
        "anti_aliasing": s.img_interpolate.anti_aliasing,
        "smoothing_beta": s.img_interpolate.smoothing_beta,
        "roi_spline_order": s.roi_interpolate.spline_order,
        "resegmentation_method": tuple(s.roi_resegment.resegmentation_method),
        "resegmentation_range": tuple(s.roi_resegment.intensity_range)
        if s.roi_resegment.intensity_range is not None else None,
        "discretisation_method": tuple(s.feature_extr.discretisation_method),
        "discretisation_bin_width": tuple(s.feature_extr.discretisation_bin_width)
        if s.feature_extr.discretisation_bin_width is not None else None,
        "ivh_discretisation_method": s.feature_extr.ivh_discretisation_method,
        "glcm_distance": tuple(s.feature_extr.glcm_distance),
        "glcm_spatial_method": tuple(s.feature_extr.glcm_spatial_method),
        "glrlm_spatial_method": tuple(s.feature_extr.glrlm_spatial_method),
        "glszm_spatial_method": tuple(s.feature_extr.glszm_spatial_method),
        "ngldm_distance": tuple(s.feature_extr.ngldm_dist),
        "ngldm_difference_level": tuple(s.feature_extr.ngldm_diff_lvl),
        "stat_percentile": tuple(s.feature_extr.stat_percentile),
        "spatial_filters": s.img_transform.spatial_filters,
        "intensity_normalisation": s.post_process.intensity_normalisation,
        "denoise_method": s.post_process.image_denoise_method,
    }


@pytest.fixture(scope="session")
def project_config():
    import config
    return config


@pytest.fixture(scope="session")
def seg_dataset():
    if not ((REAL_SEG_DATASET / "nifti").is_dir()
            and _seg_dir(REAL_SEG_DATASET) is not None):
        pytest.skip(f"organ segmentasyonu veri seti yok: {REAL_SEG_DATASET}")
    return REAL_SEG_DATASET


@pytest.fixture(scope="session")
def seg_series(seg_dataset):
    """(seri adı, ct yolu, seg yolu, label_map yolu) üçlülerinin listesi."""
    entries = []
    seg_root = _seg_dir(seg_dataset)
    for series_dir in sorted((seg_dataset / "nifti").iterdir()):
        if not series_dir.is_dir():
            continue
        ct = series_dir / "ct.nii.gz"
        seg = seg_root / series_dir.name / "seg.nii.gz"
        label_map = seg.parent / "label_map.json"
        if ct.is_file() and seg.is_file() and label_map.is_file():
            entries.append((series_dir.name, ct, seg, label_map))
    if not entries:
        pytest.skip("tam bir seri (ct+seg+label_map) bulunamadı")
    return entries
