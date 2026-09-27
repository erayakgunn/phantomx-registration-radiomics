"""Aşama 1 — parametre kilidi.

Bu dosya pipeline'ın MIRP'e verdiği ayarların TAMAMINI sabitler; sadece kodda
yazılı olanları değil, MIRP'ten sessizce miras alınanları da. Bir yayında
"IBSI uyumlu" denebilmesi için beyan edilmesi gereken liste tam olarak budur.

Anahtar ayrım:
  DECLARED  — pipeline'ın get_*_base_params() içinde açıkça verdiği değer.
  INHERITED — pipeline'ın hiç anmadığı, MIRP 2.5.0 varsayılanından gelen değer.
              Bunlar sonucu en az DECLARED olanlar kadar belirliyor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import (PROJECT_DIR, organ_radiomics_project,
                      resolve_mirp_settings)

# --------------------------------------------------------------------------- #
# Altın ayar setleri
# --------------------------------------------------------------------------- #

_COMMON = {
    # --- DECLARED ---
    "ibsi_compliant": True,
    "discretisation_method": ("fixed_bin_size",),
    "discretisation_bin_width": (25.0,),
    # Docstring "no resegmentation" diyor ama aralık VERİLDİĞİ için MIRP bunu
    # 'range' olarak çözümlüyor. Hiçbir voksel atılmıyor (aralık int16
    # kodlamasının tamamını kapsıyor) ama beyan edilmesi gereken mekanizma bu.
    "resegmentation_method": ("range",),
    "resegmentation_range": (-2000.0, 31743.0),
    "spatial_filters": None,          # IBSI-2 response-map feature'ları hesaplanmıyor

    # --- INHERITED (MIRP 2.5.0 varsayılanları) ---
    # İzotropik yeniden örnekleme YOK. Gerçek voksel FOV 300'de
    # 0,586x0,586x1,0 mm, yani 1,71x anizotropik; 3B GLCM yön vektörleri
    # fiziksel olarak eşit uzunlukta değil.
    "interpolate": False,
    "new_spacing": None,
    "spline_order": 3,                # interpolate False olduğu için atıl
    "anti_aliasing": True,            # interpolate False olduğu için atıl
    "smoothing_beta": 0.98,           # atıl
    "roi_spline_order": 1,            # atıl
    # False => Moran's I / Geary's C Monte-Carlo ile kestiriliyor ve tohum dosya
    # yolunu içeriyor. Pipeline bu iki feature'ı _DROPPED_FEATURES ile atıyor.
    "no_approximation": False,
    "ivh_discretisation_method": "none",
    "glcm_distance": (1.0,),
    "ngldm_distance": (1.0,),
    "ngldm_difference_level": (0.0,),
    "stat_percentile": (10.0, 90.0),
    "intensity_normalisation": "none",
    "denoise_method": "none",
}

GOLDEN_3D = {
    **_COMMON,
    "by_slice": False,
    "glcm_spatial_method": ("3d_average", "3d_volume_merge"),
    "glrlm_spatial_method": ("3d_average", "3d_volume_merge"),
    "glszm_spatial_method": ("3d",),
}

GOLDEN_2D = {
    **_COMMON,
    "by_slice": True,
    "glcm_spatial_method": (
        "2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge",
    ),
    "glrlm_spatial_method": (
        "2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge",
    ),
    "glszm_spatial_method": ("2d", "2.5d"),
}

# Pipeline'ın kodunda açıkça geçen anahtarlar. Geri kalan her şey mirastır.
DECLARED_KEYS = {
    "by_slice", "ibsi_compliant", "discretisation_method",
    "discretisation_bin_width", "resegmentation_range",
    "glcm_spatial_method", "glrlm_spatial_method", "glszm_spatial_method",
}


# --------------------------------------------------------------------------- #
# Kilit testleri
# --------------------------------------------------------------------------- #

def test_3d_settings_match_golden(pipeline):
    assert resolve_mirp_settings(**pipeline.get_3d_base_params()) == GOLDEN_3D


def test_2d_settings_match_golden(pipeline):
    assert resolve_mirp_settings(**pipeline.get_2d_base_params()) == GOLDEN_2D


def test_bin_width_is_25_hu(pipeline):
    """Bin genişliği hiçbir çağrı yerinde geçersiz kılınmıyor."""
    for params in (pipeline.get_3d_base_params(), pipeline.get_2d_base_params()):
        assert params["base_discretisation_bin_width"] == 25.0


def test_discretisation_anchor_is_a_fixed_declared_constant(pipeline):
    """Gri seviye orijini -2000 HU; bir kodlama sınırı, veriye uydurulmuş değil.

    Ölçüldü: 80 serinin en düşük gerçek vokseli -1246 HU, yani 754 HU pay var.
    Daha sığ çapalar (-1024, -1100) sırasıyla 80/80 ve 23/80 seride veriyi
    kelepçelerdi.
    """
    assert pipeline._DISCRETISATION_ANCHOR_HU == -2000.0
    assert pipeline.DISCRETISATION_RANGE_HU == (-2000.0, 31743.0)


def test_padding_floor_is_decoupled_from_the_anchor(pipeline):
    """İki sabit ayrı olmalı — birleşiklerken sessiz bir multi-site tehlikesiydi.

    Aynı sabit hem padding ayırıcı hem çapa olduğunda, çapanın altına inen
    gerçek bir voksel 'padding' sayılıyor ve en büyük padding'siz kare onun
    etrafına kaçıyor: ROI GEOMETRİSİ değişiyor, üstelik sessizce. Ölçüldü:
    19x19 diskte -2100 HU'luk bir hava cebi ROI'yi 13x13'ten 7x7'ye düşürüyordu.
    """
    assert pipeline._PADDING_HU_FLOOR != pipeline._DISCRETISATION_ANCHOR_HU
    assert pipeline._PADDING_HU_FLOOR < pipeline._DISCRETISATION_ANCHOR_HU
    # Sentinel'in (HU -33792) üstünde, gerçek CT değerlerinin çok altında.
    assert -33791.0 < pipeline._PADDING_HU_FLOOR < -5000.0


# --------------------------------------------------------------------------- #
# Sapmaları AÇIKÇA kayda geçiren testler
# --------------------------------------------------------------------------- #

def test_resegmentation_is_range_despite_docstring(pipeline):
    """Modül docstring'i 'MIRP is given no intensity range' diyor — yanlış.

    Bu test bugünkü GERÇEK davranışı sabitler. Docstring düzeltilecekse burası
    değişmez; asıl davranış değişirse (aralık kaldırılırsa) test kırılır ve
    çapanın MIRP'in CT varsayılanı -1000'e düştüğü fark edilir.
    """
    resolved = resolve_mirp_settings(**pipeline.get_3d_base_params())
    assert resolved["resegmentation_method"] == ("range",), (
        "Aralık kaldırıldı: FBS çapası artık MIRP'in CT varsayılanı -1000 HU. "
        "Gerçek veride ROI minimumu -1028 HU olduğu için hava popülasyonu bin 1'e "
        "sıkışır."
    )


def test_pcct_pipeline_does_not_resample_to_isotropic(pipeline):
    """En büyük IBSI sapması: izotropik yeniden örnekleme yok.

    Ölçüldü: izotropik yeniden örnekleme açıldığında gerçek bir texture ROI'sinde
    218 feature'ın 202'si değişiyor, en fazla %146. Bu testin amacı sapmayı
    onaylamak değil, GÖRÜNÜR kılmak — değiştirilirse bilinçli olarak değişsin.
    """
    resolved = resolve_mirp_settings(**pipeline.get_3d_base_params())
    assert resolved["interpolate"] is False
    assert resolved["new_spacing"] is None


def test_no_ibsi2_filter_layer(pipeline):
    """Filtre katmanı bilinçli olarak yok: her konvolüsyon patch kenarının
    ötesinde değer uydurmak zorunda kalırdı."""
    for params in (pipeline.get_3d_base_params(), pipeline.get_2d_base_params()):
        assert resolve_mirp_settings(**params)["spatial_filters"] is None


def test_inherited_defaults_are_covered_by_the_lock():
    """Altın sözlük, beyan edilmeyen her parametreyi de kapsamalı.

    Bu test kilidin kendisini korur: MIRP yeni bir varsayılan eklerse ya da
    resolve_mirp_settings genişletilirse, miras alınan bir parametre sessizce
    kilidin dışında kalamaz.
    """
    inherited = set(GOLDEN_3D) - DECLARED_KEYS
    # Sonucu belirleyen ve kodda hiç yazmayan parametreler:
    assert {"interpolate", "spline_order", "anti_aliasing", "no_approximation",
            "ivh_discretisation_method", "glcm_distance",
            "stat_percentile"} <= inherited


# --------------------------------------------------------------------------- #
# Sürüm sabitleme — BUGÜN BAŞARISIZ OLUR (düzeltmenin şartnamesi)
# --------------------------------------------------------------------------- #

def test_requirements_file_exists_and_pins_numerics():
    """MIRP/numpy/scipy sürümleri sabitlenmiş olmalı.

    Feature değerleri bu kütüphanelerin sürümüne bağlı. Kardeş proje
    phantomxpro-mirp mirp==2.5.0'ı sabitliyor; bu proje hiçbir şey sabitlemiyor,
    dolayısıyla başka bir makinede farklı sayılar üretebilir.
    """
    req = PROJECT_DIR / "requirements.txt"
    assert req.is_file(), (
        f"{req} yok. Sürüm sabitlemesi olmadan tekrarlanabilirlik iddia edilemez."
    )
    text = req.read_text(encoding="utf-8")
    for package in ("mirp", "numpy", "scipy"):
        assert f"{package}==" in text, f"{package} sabitlenmemiş"


def test_installed_versions_match_the_pins():
    """Kurulu sürümler sabitlenenlerle aynı olmalı."""
    req = PROJECT_DIR / "requirements.txt"
    if not req.is_file():
        pytest.skip("requirements.txt henüz yok (bkz. yukarıdaki test)")

    import importlib.metadata as md

    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, _, want = line.partition("==")
        assert md.version(name.strip()) == want.strip(), f"{name}: sürüm uyuşmuyor"


# --------------------------------------------------------------------------- #
# İki hattın karşılaştırması
# --------------------------------------------------------------------------- #

def _phantomxpro_settings():
    cfg_path = organ_radiomics_project() / "config.py"
    if not cfg_path.is_file():
        pytest.skip(f"{cfg_path} yok")
    import importlib.util
    spec = importlib.util.spec_from_file_location("pxp_config", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return resolve_mirp_settings(**module.MIRP_SETTINGS)


def test_the_two_pipelines_differ_only_where_intended(pipeline):
    """İki hat aynı kurumda aynı fantomu ölçüyor ama farklı ayarlarla.

    Farklar KASITLI olmalı ve burada belgelenmiş olmalı. Yeni bir fark çıkarsa
    test kırılır ve kararın bilinçli verilmesi gerekir.
    """
    pcct = resolve_mirp_settings(**pipeline.get_3d_base_params())
    pxp = _phantomxpro_settings()

    expected_differences = {
        # phantomxpro 1 mm izotropiğe yeniden örnekliyor, pcct hiç örneklemiyor.
        # pcct'de anti_aliasing atıl (interpolate False), phantomxpro'da açıkça
        # kapatılmış — ikisi de IBSI'nin istediği davranış, farklı yollardan.
        "interpolate", "new_spacing", "anti_aliasing",
        # pcct int16 kodlama tabanını çapa alıyor (hava -1246 HU'ya iniyor);
        # phantomxpro organ maskeleri için -1300..3100 yetiyor.
        "resegmentation_range",
        # pcct hem average hem merge varyantını üretiyor; phantomxpro sadece merge.
        "glcm_spatial_method", "glrlm_spatial_method",
    }
    actual = {k for k in pcct if pcct[k] != pxp[k]}
    assert actual == expected_differences, (
        f"Beklenmeyen fark: {actual ^ expected_differences}"
    )


def test_phantomxpro_anti_aliasing_is_explicitly_disabled():
    """IBSI'nin yeniden örnekleme tanımı Gauss ön-yumuşatması içermez.

    Bu projenin kendi IBSI doğrulaması da (IBSI_test/validate_ibsi.py)
    interpolasyon yapan her konfigürasyonda anti_aliasing=False kullanıyor.
    Üretim ayarı artık onunla hizalı.
    """
    pxp = _phantomxpro_settings()
    assert pxp["interpolate"] is True
    assert pxp["anti_aliasing"] is False, (
        "MIRP varsayılanı True'ya geri dönmüş — bu, kendi IBSI doğrulamanızın "
        "açıkça kapattığı ayarla üretim yapmak demek."
    )
