"""phantomxpro-mirp — IBSI parametre kilidi ve beyan edilmemiş sapmalar.

Bu hat tam CT hacmini TotalSegmentator maskeleriyle işliyor. `config.MIRP_SETTINGS`
sekiz anahtar veriyor; geri kalan her şey MIRP 2.5.0 varsayılanı. Sonucu belirleyen
parametrelerin çoğu o "geri kalan"da.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import PROJECT_DIR, resolve_mirp_settings


def _resolve(**kwargs):
    return resolve_mirp_settings(**kwargs)


GOLDEN = {
    # --- DECLARED (config.MIRP_SETTINGS içinde açıkça yazılı) ---
    "ibsi_compliant": True,
    "interpolate": True,
    "new_spacing": ((1.0, 1.0, 1.0),),
    "spline_order": 3,                       # IBSI Config E ile aynı: tricubic
    "anti_aliasing": False,                  # IBSI yeniden örneklemede ön-filtre yok
    "discretisation_method": ("fixed_bin_size",),
    "discretisation_bin_width": (25.0,),
    "resegmentation_method": ("range",),
    # Hiçbir vokseli elemiyor: maske içi ölçülen gerçek aralık -1047..1625 HU.
    "resegmentation_range": (-1300.0, 3100.0),

    # --- INHERITED (config.MIRP_SETTINGS bunları anmıyor) ---
    # NOT: aşağıdaki by_slice ve *_spatial_method değerleri MIRP'in
    # VARSAYILANLARIDIR ve bu test yalnız MIRP_SETTINGS'i (mod-bağımsız ortak
    # ayarları) çözümlediği için burada görünüyorlar. Üretimde bunların üstüne
    # config.MIRP_MODES yazılıyor: 3D ve 2D ayrı ayrı, açık listelerle
    # (bkz. test_texture_aggregation_is_declared_for_both_modes). Yani artık
    # "hiç yazılmamış" değiller — bu sözlük varsayılanın ne olduğunu kayıt
    # altında tutuyor ki mod ayarları kaldırılırsa neye düşüleceği görünsün.
    "by_slice": False,
    "smoothing_beta": 0.98,                  # anti_aliasing False iken atıl
    "roi_spline_order": 1,
    # no_approximation=False → Moran's I / Geary's C Monte-Carlo ile kestiriliyor.
    # Bu yüzden ikisi de tablodan ATILIYOR (drop_approximated; bkz.
    # test_monte_carlo_features_are_dropped). Yorum eskiden "ATMIYOR" diyordu.
    "no_approximation": False,
    "ivh_discretisation_method": "none",
    "glcm_distance": (1.0,),
    "glcm_spatial_method": ("3d_volume_merge",),
    "glrlm_spatial_method": ("3d_volume_merge",),
    "glszm_spatial_method": ("3d",),
    "ngldm_distance": (1.0,),
    "ngldm_difference_level": (0.0,),
    "stat_percentile": (10.0, 90.0),
    "spatial_filters": None,
    "intensity_normalisation": "none",
    "denoise_method": "none",
}

DECLARED_KEYS = {
    "ibsi_compliant", "interpolate", "new_spacing", "spline_order",
    "anti_aliasing", "discretisation_method", "discretisation_bin_width",
    "resegmentation_method", "resegmentation_range",
}


def test_settings_match_golden(project_config):
    assert _resolve(**project_config.MIRP_SETTINGS) == GOLDEN


def test_the_resampling_parameters_are_declared_not_inherited(project_config):
    """Yeniden örneklemeyi belirleyen üç parametre artık config.py'de yazılı.

    Bunlar MIRP'ten miras alındığında sonucu görünmez biçimde belirliyorlardı;
    özellikle anti_aliasing 177 özelliğin 145'ini değiştiriyordu.
    """
    declared = set(project_config.MIRP_SETTINGS)
    assert {"new_spacing", "spline_order", "anti_aliasing"} <= declared

    source = (PROJECT_DIR / "config.py").read_text(encoding="utf-8")
    for token in ("anti_aliasing", "spline_order"):
        assert token in source


def test_anti_aliasing_is_off_matching_this_project_s_own_ibsi_validation(project_config):
    """IBSI yeniden örnekleme tanımında Gauss ön-filtresi yok.

    Bu projenin kendi doğrulaması (IBSI_test/validate_ibsi.py) interpolasyon
    yapan her konfigürasyonda — B, C, D, E ve IBSI-2 — anti_aliasing=False
    kullanıyor. Üretim ayarı artık onunla aynı.
    """
    resolved = _resolve(**project_config.MIRP_SETTINGS)
    assert resolved["interpolate"] is True
    assert resolved["anti_aliasing"] is False
    # spline_order=3 bir sapma DEĞİL: IBSI Config E tam olarak tricubic kullanıyor.
    assert resolved["spline_order"] == 3


def test_resegmentation_removes_no_voxels(project_config):
    """Aralık artık gerçek veriyi elemiyor, yalnızca çapa görevi görüyor.

    Eski [-1000, 400] yumuşak doku penceresi 117 yapının hepsine uygulanıyordu
    ve kemiği tanımlayan yoğunlukları siliyordu (vertebrae_T10'un %69,5'i).
    Yeni sınırlar maske içi ölçülen -1047..1625 HU aralığını paylı kapsıyor.
    """
    lower, upper = _resolve(**project_config.MIRP_SETTINGS)["resegmentation_range"]
    assert lower <= -1300.0, "alt sınır gözlenen -1047 HU'nun altında kalmalı"
    assert upper >= 3100.0, "üst sınır yoğun kortikal kemiği kapsamalı"


def test_mirp_anti_aliasing_has_no_downsampling_guard():
    """Gauss ön-filtresi KOŞULSUZ çalışıyor — yalnızca küçültmede değil.

    `mirp/_image_processing/anti_aliasing.py` içindeki koruma satırı yorum
    yapılmış durumda:

        # Only apply to down-sampling (map_spacing > 1.0)
        # map_spacing[map_spacing<=1.0] = 0.0

    Sonuç: `interpolate=True` olan her seride filtre devrede — büyütmede de,
    kaynak spacing zaten 1 mm olduğunda da. Bu proje hem recon1mm hem recon3mm
    serileri işlediği için ikisi de etkileniyor.
    """
    import mirp._image_processing.anti_aliasing as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "# map_spacing[map_spacing<=1.0] = 0.0" in source, (
        "MIRP'te küçültme koruması artık etkin olabilir — öyleyse filtre yalnızca "
        "küçültmede çalışıyordur ve bu testin ve raporun güncellenmesi gerekir."
    )


def test_mirp_own_ibsi_benchmarks_disable_anti_aliasing():
    """MIRP'in kendi IBSI referans yolu anti-aliasing'i KAPATIYOR.

    `generic_image.py` içindeki IBSI karşılaştırma yolu `anti_aliasing=False`
    kullanıyor. Yani bu ayarı açık bırakmak, MIRP'in IBSI uyumlu saydığı
    konfigürasyondan sapmak demek.
    """
    import mirp._images.generic_image as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "anti_aliasing=False" in source


@pytest.mark.slow
def test_anti_aliasing_materially_changes_the_features(tmp_path):
    """Sapmanın büyüklüğünü ölçer: kozmetik değil, belirleyici.

    Ölçüldü (0,586x0,586x1,0 mm dokulu küre -> 1 mm izotropik):
    177 feature'ın 145'i değişiyor. `stat_min` 7'den -57'ye iniyor — bulanıklık
    yoğunluk uçlarını siliyor; `ngt_complexity` 12 kat değişiyor.

    Bu test, ayarın açık/kapalı olmasının sonucu gerçekten belirlediğini
    kanıtlıyor; hangisinin doğru olduğuna karar vermiyor.
    """
    import numpy as np
    import SimpleITK as sitk
    from mirp import extract_features

    rng = np.random.default_rng(0)
    size = 48
    volume = rng.normal(60.0, 40.0, size=(size,) * 3).astype(np.float32)
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    mask = (((xx - size / 2) ** 2 + (yy - size / 2) ** 2
             + (zz - size / 2) ** 2) <= 14 ** 2).astype(np.uint8)

    for array, name, dtype in ((volume, "img", np.float32), (mask, "msk", np.uint8)):
        image = sitk.GetImageFromArray(array.astype(dtype))
        image.SetSpacing((0.5859375, 0.5859375, 1.0))
        sitk.WriteImage(image, str(tmp_path / f"{name}.nii.gz"))

    common = dict(
        image=str(tmp_path / "img.nii.gz"), mask=str(tmp_path / "msk.nii.gz"),
        roi_name={"1": "roi"}, image_modality="CT", new_spacing=1.0,
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0,
        resegmentation_intensity_range=[-1000.0, 400.0],
        base_feature_families="all", ibsi_compliant=True,
    )

    def run(**extra):
        frame = extract_features(**common, **extra)[0]
        return {c: float(frame[c].iloc[0]) for c in frame.columns
                if frame[c].dtype.kind == "f" and not c.startswith("image_")}

    with_aa, without_aa = run(), run(anti_aliasing=False)

    shared = [k for k in with_aa if k in without_aa]
    changed = [k for k in shared
               if not (np.isnan(with_aa[k]) and np.isnan(without_aa[k]))
               and not abs(with_aa[k] - without_aa[k]) <= 1e-9 * max(1.0, abs(with_aa[k]))]

    assert len(changed) > 0.5 * len(shared), (
        f"anti-aliasing artık feature'ların yalnızca {len(changed)}/{len(shared)} "
        f"tanesini değiştiriyor — MIRP davranışı değişmiş olabilir."
    )


def test_texture_aggregation_is_declared_for_both_modes(project_config):
    """2B ve 3B toplama yöntemleri AÇIKÇA verilir, MIRP varsayılanına bırakılmaz.

    Bırakıldığında ölçüldü: varsayılan yalnız TEK bir yöntem üretiyor
    (3B'de `3d_volume_merge`, 2B'de `2d_slice_merge`), yani hem 2B tablosu hiç
    oluşmuyor hem de 3B'de IBSI'nin iki yönteminden biri kayboluyordu.

    Listeler kardeş hat `pcct-radiomics-mirp` ile BİREBİR aynı tutuluyor
    (extract_..._v4.py:150-186): aynı kurumda aynı fantom için üretilen
    tablolar ancak böyle karşılaştırılabilir. Bu test o eşitliği kilitler.
    """
    modlar = project_config.MIRP_MODES
    assert set(modlar) == {"3D", "2D"}
    assert project_config.PRIMARY_MODE == "3D"
    assert modlar["3D"]["by_slice"] is False
    assert modlar["2D"]["by_slice"] is True

    iki_matris = ["3d_average", "3d_volume_merge"]
    dort_matris = ["2d_average", "2d_slice_merge",
                   "2.5d_direction_merge", "2.5d_volume_merge"]
    for aile in ("glcm", "glrlm"):
        assert modlar["3D"][f"{aile}_spatial_method"] == iki_matris
        assert modlar["2D"][f"{aile}_spatial_method"] == dort_matris
    for aile in ("glszm", "gldzm", "ngtdm", "ngldm"):
        assert modlar["3D"][f"{aile}_spatial_method"] == ["3d"]
        assert modlar["2D"][f"{aile}_spatial_method"] == ["2d", "2.5d"]

    # Hiçbir doku ailesi varsayılana bırakılmamalı.
    for mod, ayar in modlar.items():
        for aile in ("glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"):
            assert f"{aile}_spatial_method" in ayar, f"{mod}: {aile} beyan edilmemiş"

    # Mod ayarları ORTAK ayarlarla çakışmamalı (üzerine yazma sürprizi olmasın).
    for mod, ayar in modlar.items():
        cakisan = set(ayar) & set(project_config.MIRP_SETTINGS)
        assert not cakisan, f"{mod}: MIRP_SETTINGS ile çakışan anahtar {cakisan}"


def test_pasteable_methods_paragraph_declares_the_live_aggregation(project_config):
    """`methods` sayfasının YAPIŞTIRILABİLİR paragrafı canlı ayarı anlatmalı.

    Bu bir kez oldu: paragrafta "Texture matrices were computed in 3D and
    volume-merged" sabit yazılıydı, oysa config GLCM/GLRLM için hem
    `3d_average` hem `3d_volume_merge` üretiyordu ve ayrıca tam bir 2B modu
    vardı. Sayfanın yapılandırılmış satırları (`_texture_rows`) canlı ayardan
    türediği için doğruyu söylüyor, yayına giden metin yanlışı söylüyordu —
    yani beyanın en çok okunan yarısı sessizce bayatlamıştı.

    Aynı beyanın iki ayrı yerde yazılması bu sapmayı mümkün kıldığı için test,
    yalnız içeriği değil, metnin ayardan TÜRETİLDİĞİNİ de kilitliyor.
    """
    import excel_report

    cumle = excel_report._texture_sentence()

    # Canlı config'de gerçekten kullanılan HER toplama yöntemi metinde geçmeli.
    for mod, ayar in project_config.MIRP_MODES.items():
        assert f"{mod}_base" in cumle, f"{mod} sayfası beyanda yok"
        for anahtar in ("glcm_spatial_method", "glrlm_spatial_method",
                        "glszm_spatial_method", "gldzm_spatial_method",
                        "ngtdm_spatial_method", "ngldm_spatial_method"):
            for yontem in ayar.get(anahtar, []):
                okunur = excel_report._AGGREGATION_LABEL.get(yontem, yontem)
                assert okunur in cumle, (
                    f"{mod}/{anahtar}: '{yontem}' ({okunur}) beyanda geçmiyor — "
                    f"paragraf canlı ayarla uyuşmuyor.")

    # Paragrafın kendisi cümleyi gerçekten TAŞIMALI (biçimlendirme kopmasın).
    paragraf = excel_report._METHODS_PARAGRAPH.format(
        mirp="x", spacing=1.0, order=3, roi_scope="x", low=-1300, high=3100,
        width=25, texture=cumle)
    assert cumle in paragraf

    # Ve sabit metin geri sızmamalı: gövde doku yöntemini kendisi yazmamalı.
    govde = excel_report._METHODS_PARAGRAPH.replace("{texture}", "")
    assert "volume-merged" not in govde and "by_slice" not in govde, (
        "Doku toplama beyanı yeniden paragrafa sabitlenmiş; ayar değişince "
        "sessizce yanlışa döner.")


def test_monte_carlo_features_are_dropped():
    """Moran's I ve Geary's C tablodan ATILIR.

    MIRP bunları 1000 vokselin üstünde Monte-Carlo ile kestiriyor
    (no_approximation=False; tam hesap O(n²) ve bu boyutlarda pratik değil) ve
    tohum dosya YOLUNU içeriyor. Yani ne IBSI'nin tanımladığı kesin değerler
    ne de veri taşındığında tekrarlanabilir. Tabloda tutmak onları
    kullanılabilir sanmaya yol açıyordu.
    """
    import feature_extraction_mirp as fx
    import pandas as pd

    assert set(fx.APPROXIMATED_FEATURES) == {"morph_moran_i", "morph_geary_c"}
    df = pd.DataFrame({"morph_volume": [1.0], "morph_moran_i": [0.1],
                       "morph_geary_c": [0.9]})
    out = fx.drop_unreportable(df)
    assert list(out.columns) == ["morph_volume"]
    # Sütun yoksa patlamamalı (eski/dar tablolar).
    assert list(fx.drop_unreportable(out).columns) == ["morph_volume"]
    # Eski ad hâlâ çalışmalı (dış betikler).
    assert fx.drop_approximated is fx.drop_unreportable


def test_ivh_volume_fractions_are_dropped_and_intensity_fractions_are_kept():
    """IVH hacim kesirleri tablodan ATILIR, yoğunluk kesirleri KALIR.

    Gerekçe ölçüldü (284 satır): resegmentasyon aralığı [-1300, 3100] HU aynı
    zamanda IVH gamma eksenini belirlediği için v-kesimleri yumuşak dokunun
    tamamen dışına düşüyor — ivh_v10 ≡ 1.0, ivh_v75 ≡ ivh_v90 ≡ 0.0,
    ivh_diff_v10_v90 ≡ ivh_v10 ve ivh_diff_v25_v75 ≡ ivh_v25.

    Sütun kümesi referans çalışma kitabıyla (pcct-radiomics-mirp çıktısı)
    aynı olacak şekilde seçildi: yalnız yedi `ivh_i*` sütunu raporlanır.
    """
    import feature_extraction_mirp as fx
    import pandas as pd

    assert set(fx.IVH_VOLUME_FRACTION_FEATURES) == {
        "ivh_v10", "ivh_v25", "ivh_v50", "ivh_v75", "ivh_v90",
        "ivh_diff_v10_v90", "ivh_diff_v25_v75"}

    kalan = ["ivh_i10", "ivh_i25", "ivh_i50", "ivh_i75", "ivh_i90",
             "ivh_diff_i10_i90", "ivh_diff_i25_i75"]
    df = pd.DataFrame({c: [1.0] for c in
                       list(fx.IVH_VOLUME_FRACTION_FEATURES) + kalan})
    out = fx.drop_unreportable(df)
    assert sorted(out.columns) == sorted(kalan)
    assert len([c for c in out.columns if c.startswith("ivh_")]) == 7


def test_ivh_discretisation_is_declared_not_inherited(project_config):
    """IVH ayrımlaştırması config'de YAZILI olmalı.

    14 ivh_* özniteliğinin tamamını bu tek parametre belirliyor. CT için doğru
    davranış 'none' (HU kalibre ve tam sayı; IVH doğrudan yoğunluklar üzerinden)
    ve MIRP'in CT varsayılanı da bu — ama miras alındığında sürüm değişince
    sessizce kayabilirdi.
    """
    assert "ivh_discretisation_method" in project_config.MIRP_SETTINGS
    assert _resolve(**project_config.MIRP_SETTINGS)["ivh_discretisation_method"] == "none"
    source = (PROJECT_DIR / "config.py").read_text(encoding="utf-8")
    assert "ivh_discretisation_method" in source


def test_no_ibsi2_filters(project_config):
    assert _resolve(**project_config.MIRP_SETTINGS)["spatial_filters"] is None


# --------------------------------------------------------------------------- #
# Provenance — BUGÜN BAŞARISIZ OLUR (düzeltmenin şartnamesi)
# --------------------------------------------------------------------------- #

def test_done_sidecar_records_extraction_provenance():
    """BUGÜN BAŞARISIZ OLUR — düzeltmenin şartnamesi.

    `features.done.json` yalnızca organ sayımlarını yazıyor. Bir sonucun hangi
    mirp sürümüyle, hangi ayarlarla ve hangi girdiden üretildiği kayıtlı değil.
    Provenance olmadan "tekrarlanabilir" beyanı doğrulanamaz.

    Beklenen: sidecar'a mirp/numpy/scipy sürümleri, MIRP_SETTINGS hash'i ve
    ct.nii.gz + seg.nii.gz checksum'ları yazılsın.
    """
    source = (PROJECT_DIR / "feature_extraction_mirp.py").read_text(encoding="utf-8")

    missing = [token for token in ("mirp_version", "settings_hash", "input_checksum")
               if token not in source]
    assert not missing, (
        f"features.done.json provenance içermiyor (eksik: {', '.join(missing)}). "
        f"Sonuçların hangi sürüm ve ayarlarla üretildiği kayıt altında değil."
    )
