"""PhantomXPro test paketi.

Çalıştırma:
    C:\\Python314\\python.exe test_pipeline.py           # bağımsız (pytest gerekmez)
    C:\\Python314\\python.exe test_pipeline.py --fast     # yavaş entegrasyon testini atla
    pytest test_pipeline.py                              # pytest ile de çalışır

Birim testleri saftır (DICOM/MIRP gerektirmez). Entegrasyon testi küçük sentetik
veri üretip MIRP'i gerçekten çalıştırır (IBSI çıktı + paralel==sıralı özdeşliği).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

import config
import sort_spatially as ss
import feature_extraction_mirp as fx
import run_totalsegmentator as ts
import batch_120kvp as batch
import series_naming as sn
import apply_naming as an

FAST = "--fast" in sys.argv


# ============================================================ #
# Yardımcı: küçük sahte DICOM üstbilgi nesnesi
# ============================================================ #
class FakeDS:
    """pydicom.Dataset benzeri; getattr ile alan döndürür."""
    def __init__(self, **kw):
        self._d = kw
    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name)


# ============================================================ #
# sort_spatially — saf mantık
# ============================================================ #
def test_sanitize():
    assert ss._sanitize("Scan 1  Br40") == "Scan_1_Br40"
    assert ss._sanitize(None, "x") == "x"
    assert ss._sanitize("", "x") == "x"
    assert ss._sanitize("a/b\\c:d*e") == "a_b_c_d_e"
    assert ss._sanitize("__leading__") == "leading"
    assert ss._sanitize("...") == "NA"          # tümü strip edilince fallback
    assert ss._sanitize(99) == "99"


def test_get_kvp():
    assert ss._get_kvp(FakeDS(KVP="120")) == 120.0
    assert ss._get_kvp(FakeDS(KVP=120)) == 120.0
    assert ss._get_kvp(FakeDS(KVP=119.5)) == 119.5
    assert ss._get_kvp(FakeDS()) is None
    assert ss._get_kvp(FakeDS(KVP="")) is None
    assert ss._get_kvp(FakeDS(KVP="abc")) is None


def test_kvp_bucket():
    assert ss._kvp_bucket(120.0) == ("120kVp", True)
    assert ss._kvp_bucket(119.0) == ("119kVp", True)      # tolerans ±1
    assert ss._kvp_bucket(121.0) == ("121kVp", True)
    assert ss._kvp_bucket(121.5) == ("122kVp", False)     # tolerans dışı, yuvarlama
    assert ss._kvp_bucket(100.0) == ("100kVp", False)
    assert ss._kvp_bucket(None) == ("unknown_kVp", False)


def test_cross_and_spatial_key():
    assert ss._cross([1, 0, 0], [0, 1, 0]) == [0, 0, 1]
    # eksenel: normal = +z, IPP.z sıralamayı verir
    ds_lo = FakeDS(ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, -50])
    ds_hi = FakeDS(ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, 50])
    assert ss._spatial_key(ds_lo) == -50.0
    assert ss._spatial_key(ds_hi) == 50.0
    # IOP/IPP yoksa InstanceNumber'a düşer
    assert ss._spatial_key(FakeDS(InstanceNumber=7)) == 7.0
    # hiçbiri yoksa 0
    assert ss._spatial_key(FakeDS()) == 0.0


def test_spatial_sort_order():
    # karışık sırayı z'ye göre düzeltir
    slices = [
        FakeDS(ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, z])
        for z in (3, -1, 2, 0, -5)
    ]
    ordered = sorted(slices, key=ss._spatial_key)
    zs = [s.ImagePositionPatient[2] for s in ordered]
    assert zs == [-5, -1, 0, 2, 3]


# ============================================================ #
# feature_extraction — saf mantık
# ============================================================ #
def test_split_roi_map():
    rm = {str(i): f"o{i}" for i in range(1, 11)}   # 10 organ
    # 3 öbek → 4,3,3
    chunks = fx._split_roi_map(rm, 3)
    assert [len(c) for c in chunks] == [4, 3, 3]
    # birleşim = tüm organlar, kayıp/çakışma yok
    merged = {}
    for c in chunks:
        merged.update(c)
    assert merged == rm
    # n>len → len öbek, her biri 1
    assert len(fx._split_roi_map(rm, 20)) == 10
    # n=1 → tek öbek
    assert fx._split_roi_map(rm, 1) == [rm]
    # boş
    assert fx._split_roi_map({}, 3) == [{}]


def test_build_roi_map(tmp_seg, monkeypatch):
    seg_path, present = tmp_seg
    monkeypatch.setattr(config, "TARGET_ORGANS", None)   # seçimden bağımsız
    lm = {"1": "spleen", "5": "liver"}
    rm = fx.build_roi_map(seg_path, lm)
    # sadece maskede olanlar + haritada olmayan için fallback
    assert rm["1"] == "spleen"
    assert rm["5"] == "liver"
    assert all(k in [str(p) for p in present] for k in rm)


# ============================================================ #
# Organ (ROI) seçimi — config.TARGET_ORGANS
# ============================================================ #
def test_target_organs_filters_by_name(tmp_seg, monkeypatch):
    # maskede 1=spleen ve 5=liver var; yalnız liver isteniyor
    seg_path, _ = tmp_seg
    monkeypatch.setattr(config, "TARGET_ORGANS", {"liver"})
    rm = fx.build_roi_map(seg_path, {"1": "spleen", "5": "liver"})
    assert rm == {"5": "liver"}


def test_target_organs_none_keeps_everything(tmp_seg, monkeypatch):
    seg_path, _ = tmp_seg
    monkeypatch.setattr(config, "TARGET_ORGANS", None)
    rm = fx.build_roi_map(seg_path, {"1": "spleen", "5": "liver"})
    assert set(rm.values()) == {"spleen", "liver"}


def test_target_organs_unknown_name_raises(monkeypatch):
    # Yazım hatası SESSİZ kalmamalı: filtrelenirse 0 organ çıkar, seri
    # 'tamamlandı' işaretlenir ve resume onu bir daha hiç denemez.
    monkeypatch.setattr(config, "TARGET_ORGANS", {"liver", "livver"})
    try:
        fx.selected_organs()
    except ValueError as exc:
        assert "livver" in str(exc)
    else:
        raise AssertionError("bilinmeyen yapı adı için hata beklenmişti")


def test_target_organs_filter_is_name_based_not_label_based(tmp_seg, monkeypatch):
    # TS etiketleri yeniden numaralandırsa bile AD üzerinden seçildiği için
    # yanlış organa öznitelik yazılamaz: burada liver 5 değil 1 numarada.
    seg_path, _ = tmp_seg
    monkeypatch.setattr(config, "TARGET_ORGANS", {"liver"})
    rm = fx.build_roi_map(seg_path, {"1": "liver", "5": "spleen"})
    assert rm == {"1": "liver"}


def test_target_labels_maps_names_to_label_ids(monkeypatch):
    lm = json.loads((config.PROJECT_ROOT / "totalseg_total_labels.json")
                    .read_text(encoding="utf-8"))
    monkeypatch.setattr(config, "TARGET_ORGANS", {"spleen", "aorta"})
    assert fx.target_labels(lm) == {1, 52}       # spleen=1, aorta=52
    monkeypatch.setattr(config, "TARGET_ORGANS", None)
    assert fx.target_labels(lm) is None          # None → QC hepsini çizer


def test_shipped_config_target_organs_are_valid():
    # config.py'deki varsayılan seçim gerçekten TS 'total' görevinde var mı?
    # Bir harf hatası tüm koşuyu boşa çıkarırdı.
    targets = config.TARGET_ORGANS
    if targets is None:
        return
    lm = json.loads((config.PROJECT_ROOT / "totalseg_total_labels.json")
                    .read_text(encoding="utf-8"))
    unknown = sorted(set(targets) - set(lm.values()))
    assert not unknown, f"config.TARGET_ORGANS geçersiz ad içeriyor: {unknown}"


def test_shipped_selection_is_spleen_and_aorta():
    """KARAR: yalnız dalak ve aorta çıkarılıyor.

    İliopsoas ölçümle elendi (rekonstrüksiyonlar arası %35-40 yayılım, kasın
    yalnız kraniyal ucu kapsanıyor); karaciğer bütün olarak kararlı ama segment
    bölümlemesi fantomda damar olmadığı için anatomik dayanaktan yoksun.
    Bu test, seçimin sessizce genişlemesini önler.
    """
    assert config.TARGET_ORGANS == {"spleen", "aorta"}


def test_combine_applies_current_organ_selection(monkeypatch, tmp_path):
    """Seçim daraltıldığında birleşik tablo GÜNCEL seçimi yansıtmalı.

    Seri tabloları daha geniş bir seçimle yazılmış olabilir; o satırların
    tabloda kalması, artık istenmeyen organların analize sızması demekti.
    Daraltma yeniden çıkarım gerektirmez: MIRP her ROI'yi bağımsız hesaplar.
    """
    monkeypatch.setattr(config, "FEATURES", tmp_path)
    monkeypatch.setattr(config, "COMBINED_FEATURES_CSV", tmp_path / "all.csv")
    monkeypatch.setattr(config, "TARGET_ORGANS", {"spleen", "aorta"})

    eski = pd.DataFrame({
        "series_key": ["S"] * 4,
        "image_mask_label": ["liver", "spleen", "aorta", "iliopsoas_left"],
        "morph_volume": [700000.0, 100000.0, 23000.0, 6000.0],
    })
    (tmp_path / "S").mkdir()
    eski.to_csv(tmp_path / "S" / "features.csv", index=False)

    out = batch.combine_features(["S"])
    assert out is not None
    assert set(out["image_mask_label"]) == {"spleen", "aorta"}
    assert len(out) == 2


def test_repeat_protocol_series_keep_separate_identities(monkeypatch, tmp_path):
    """Aynı protokolün tekrar çekimleri birleşik tabloda AYRI satır olmalı.

    `assign_series_names` çakışmayı `_s<seri_no>` ekiyle çözüp KLASÖR adına
    yazıyor; `_enrich_identity` ise adı DICOM alanlarından yeniden hesaplıyor ve
    o ek orada yok. İki yol ayrıştığı için tekrar çekimleri tabloda tek ada
    çöküyordu — gerçek veride ölçüldü: 32 işlenmiş seri, 28 benzersiz ad,
    4 seri görünmez. Kimlik klasör adından gelmeli.
    """
    monkeypatch.setattr(config, "FEATURES", tmp_path)
    monkeypatch.setattr(config, "COMBINED_FEATURES_CSV", tmp_path / "all.csv")
    monkeypatch.setattr(config, "TARGET_ORGANS", {"spleen"})

    # İki klasör, protokolü AYNI: yeniden hesaplanan ad ikisinde de aynı çıkar.
    ortak = dict(dcm_Manufacturer="Siemens", dcm_ManufacturerModelName="NAEOTOM",
                 dcm_StudyDate="20260205", dcm_KVP=120, dcm_Exposure=100,
                 dcm_ReconstructionDiameter=300, dcm_SliceThickness=1,
                 dcm_ConvolutionKernel="Qr40f")
    for ad in ("A_recon1mm", "A_recon1mm_s49"):
        (tmp_path / ad).mkdir()
        pd.DataFrame([{**ortak, "series_key": ad, "image_mask_label": "spleen",
                       "morph_volume": 100000.0}]).to_csv(
            tmp_path / ad / "features.csv", index=False)

    out = batch.combine_features(["A_recon1mm", "A_recon1mm_s49"])
    assert out is not None
    assert len(out) == 2, "iki seri de tabloda olmalı"
    assert out["series"].nunique() == 2, (
        f"tekrar çekimleri tek ada çöktü: {sorted(out['series'])}")
    assert set(out["series"]) == {"A_recon1mm", "A_recon1mm_s49"}


def test_absent_target_organs_recorded_in_sidecar(monkeypatch, tmp_path):
    # Hedeflenip maskede bulunmayan organ, çıkarım hatasından AYRI raporlanır.
    monkeypatch.setattr(config, "FEATURES", tmp_path)
    monkeypatch.setattr(config, "TARGET_ORGANS", {"liver", "aorta"})
    fx._write_done_sidecar("S", {"liver"}, {"liver"}, [], absent_targets=["aorta"])
    d = json.loads((tmp_path / "S" / fx.DONE_SIDECAR).read_text(encoding="utf-8"))
    assert d["absent_target_organs"] == ["aorta"]
    assert d["missing_organs"] == []          # çıkarım hatası YOK
    assert d["complete"] is True              # istenen ve bulunan her organ çıktı
    assert d["target_organs"] == ["aorta", "liver"]   # provenans


def test_safe_worker_count(monkeypatch):
    # bütçe 20GB, öbek 5GB → 4 işçi (cap ile sınırlı)
    monkeypatch.setattr(fx, "available_memory_bytes", lambda: 20 * 10**9)
    monkeypatch.setattr(fx, "_resampled_bytes", lambda *a: 125 * 10**6)  # ×40 = 5GB
    monkeypatch.setattr(config, "WORKER_MEM_MULTIPLIER", 40.0)
    monkeypatch.setattr(config, "PARALLEL_WORKERS", 8)
    assert fx._safe_worker_count(Path("x")) == 4
    # bütçe çok küçük → 1
    monkeypatch.setattr(fx, "available_memory_bytes", lambda: 1 * 10**9)
    assert fx._safe_worker_count(Path("x")) == 1
    # cap uygulanır
    monkeypatch.setattr(fx, "available_memory_bytes", lambda: 1000 * 10**9)
    monkeypatch.setattr(config, "PARALLEL_WORKERS", 6)
    assert fx._safe_worker_count(Path("x")) == 6


def test_finalize_series_frame():
    df = pd.DataFrame({"image_mask_label": ["liver", "aorta"], "stat_mean": [1.0, 2.0]})
    out = fx._finalize_series_frame(df, "PT__ser", "PT", recon_type="VMI_70keV", kvp=120.0)
    # kimlik + PCCT etiket sütunları başta
    assert list(out.columns[:5]) == ["series_key", "patient_id", "recon_type", "kvp", "image_mask_label"]
    assert out["patient_id"].iloc[0] == "PT"
    assert set(out["recon_type"]) == {"VMI_70keV"} and set(out["kvp"]) == {120.0}
    # image_mask_label'a göre sıralı (aorta < liver)
    assert list(out["image_mask_label"]) == ["aorta", "liver"]
    # varsayılanlar (EID): recon_type=native, kvp=None; patient_id None ise key'den türetir
    out2 = fx._finalize_series_frame(df, "HASTA__ser", None)
    assert out2["patient_id"].iloc[0] == "HASTA" and set(out2["recon_type"]) == {"native"}


def test_execute_backoff_isolation(monkeypatch):
    """OOM simülasyonu: havuz hep çöker → 1 işçiye iner → organları izole eder,
    sadece sorunlu organı atlar."""
    monkeypatch.setattr(fx, "_run_pool", lambda tasks, workers: ([], list(tasks)))

    def fake_iso(task):
        rm = task["roi_map"]
        if len(rm) > 1:
            return None, False                      # çok-organlı öbek 'çöker'
        organ = next(iter(rm.values()))
        if organ == "bad":
            return None, False                      # bu organ hep OOM
        return {"series_key": task["series_key"], "df": pd.DataFrame([{"x": 1}]), "error": None}, True

    monkeypatch.setattr(fx, "_run_isolated", fake_iso)
    tasks = [{"series_key": "S", "roi_map": {"1": "a", "2": "bad", "3": "c"}}]
    results, failed = fx._execute(tasks, initial_workers=4)
    got = [r for r in results if r.get("df") is not None]
    assert len(got) == 2                            # a ve c kurtarıldı
    assert failed == ["S/bad"]                      # sadece bad atlandı


def test_execute_clean_success(monkeypatch):
    """Havuz temiz çalışırsa sonuçlar aynen döner, izolasyona düşmez."""
    def fake_pool(tasks, workers):
        return [{"series_key": "S", "df": pd.DataFrame([{"x": i}]), "error": None}
                for i, _ in enumerate(tasks)], []
    monkeypatch.setattr(fx, "_run_pool", fake_pool)
    tasks = [{"series_key": "S", "roi_map": {"1": "a"}},
             {"series_key": "S", "roi_map": {"2": "b"}}]
    results, failed = fx._execute(tasks, initial_workers=2)
    assert len([r for r in results if r.get("df") is not None]) == 2
    assert failed == []


# ============================================================ #
# batch — dedup + test-retest (take)
# ============================================================ #
def _rec(folder, uid, desc, study_uid="ST1", date="20260101", time="120000", n=100,
         acq="", series_number="1", series_time=""):
    return {
        "folder": Path(folder), "files": [Path(f"{i}.ima") for i in range(n)],
        "series_uid": uid, "study_uid": study_uid, "series_desc": desc,
        "series_number": series_number, "patient_id": "P", "study_date": date,
        "study_time": time, "acq_time": time, "kvp": 120.0,
        "acq_number": acq, "series_time": series_time,
    }


def test_dedup_true_duplicate():
    # aynı UID iki klasörde → biri tutulur (çok slice'lı), diğeri raporlanır
    recs = [_rec("A_copy1", "UID1", "Scan Br40", n=100),
            _rec("A_copy2", "UID1", "Scan Br40", n=50)]
    unique, dups = batch.dedup_and_takes(recs)
    assert len(unique) == 1
    assert unique[0]["folder"].name == "A_copy1"    # daha çok slice tutuldu
    assert len(dups) == 1
    assert dups[0]["duplicate"] == "A_copy2"


def test_no_duplicate_no_take():
    # 70 benzersiz açıklama, tek oturum → kopya yok, take yok (bu veri seti gibi)
    recs = [_rec(f"S{i}", f"UID{i}", f"Scan {i} Br40") for i in range(5)]
    unique, dups = batch.dedup_and_takes(recs)
    assert len(unique) == 5
    assert dups == []
    assert all(r.get("take") is None for r in unique)


def test_test_retest_takes():
    # aynı açıklama, FARKLI çekim oturumu (study date/uid) → take1/take2
    recs = [
        _rec("visit1_Br40", "UIDa", "Scan Br40", study_uid="STa", date="20260101", time="0900"),
        _rec("visit2_Br40", "UIDb", "Scan Br40", study_uid="STb", date="20260115", time="1000"),
    ]
    unique, dups = batch.dedup_and_takes(recs)
    assert dups == []
    takes = {r["folder"].name: r["take"] for r in unique}
    # kronolojik: erken tarih take1
    assert takes["visit1_Br40"] == 1
    assert takes["visit2_Br40"] == 2


def test_series_key_with_take():
    r = _rec("SCAN_1_BR40_0001", "UID", "Scan Br40"); r["take"] = None
    k1 = batch._series_key(r)
    assert k1.startswith("P__SCAN_1_BR40_0001__")     # UID hash eki her zaman var
    r2 = dict(r); r2["take"] = 2
    assert "__take2__" in batch._series_key(r2)
    # FARKLI UID → FARKLI anahtar (aynı klasör adı olsa bile çakışma yok)
    r3 = _rec("SCAN_1_BR40_0001", "BASKA_UID", "Scan Br40"); r3["take"] = None
    assert batch._series_key(r3) != k1


def test_series_key_no_cross_patient_take():
    # Farklı hastalar aynı açıklama → biri diğeri yüzünden take almamalı
    recs = [
        _rec("A_ser", "UIDa", "Thorax BR40", study_uid="STa", date="20260101"),
        _rec("B_ser", "UIDb", "Thorax BR40", study_uid="STb", date="20260201"),
    ]
    recs[0]["patient_id"] = "PA"; recs[1]["patient_id"] = "PB"
    unique, _ = batch.dedup_and_takes(recs)
    assert all(r["take"] is None for r in unique)      # farklı hasta → take yok


def test_empty_desc_no_take():
    recs = [_rec("s1", "U1", ""), _rec("s2", "U2", "")]
    unique, _ = batch.dedup_and_takes(recs)
    assert all(r["take"] is None for r in unique)      # boş açıklama → sahte take yok


def test_take_from_acquisition_number_in_single_session(monkeypatch):
    """Test ve retest AYNI çalışma altında dışa aktarılmışsa take yine atanmalı.

    ÖLÇÜLDÜ (20260312 seti, 202 seri): tek StudyInstanceUID/Date/Time, tekrar
    ekseni yalnız AcquisitionNumber. Oturum karşılaştırması burada hiçbir
    tekrarı göremiyordu ve test-retest çalışmasının ana değişkeni tabloda
    kayboluyordu.
    """
    monkeypatch.setattr(config, "DROP_DUPLICATE_RECONSTRUCTIONS", False, raising=False)
    recs = [_rec("a", "U1", "Scan 1 Br40", acq="11", series_number="85"),
            _rec("b", "U2", "Scan 1 Br40", acq="21", series_number="2085")]
    unique, _ = batch.dedup_and_takes(recs)
    takes = {r["series_uid"]: r["take"] for r in unique}
    assert takes == {"U1": 1, "U2": 2}          # düşük çekim numarası = take1


def test_acquisition_take_order_is_numeric_not_lexicographic(monkeypatch):
    """'9' < '10' sayısal olarak; metin sırasında ters çıkardı."""
    monkeypatch.setattr(config, "DROP_DUPLICATE_RECONSTRUCTIONS", False, raising=False)
    recs = [_rec("a", "U1", "P", acq="10"), _rec("b", "U2", "P", acq="9")]
    unique, _ = batch.dedup_and_takes(recs)
    takes = {r["series_uid"]: r["take"] for r in unique}
    assert takes == {"U2": 1, "U1": 2}


def test_session_takes_win_over_acquisition_number(monkeypatch):
    """Oturum ayrımı VARSA eski davranış korunur (çekim numarasına düşülmez)."""
    monkeypatch.setattr(config, "DROP_DUPLICATE_RECONSTRUCTIONS", False, raising=False)
    recs = [_rec("v1", "Ua", "P", study_uid="STa", date="20260101", acq="99"),
            _rec("v2", "Ub", "P", study_uid="STb", date="20260115", acq="1")]
    unique, _ = batch.dedup_and_takes(recs)
    takes = {r["series_uid"]: r["take"] for r in unique}
    assert takes == {"Ua": 1, "Ub": 2}          # tarih sırası, acq'a rağmen


def test_duplicate_reconstruction_is_dropped_and_reported(monkeypatch):
    """Aynı çekimin ikinci rekonstrüksiyonu işlenmez ama RAPORLANIR.

    ÖLÇÜLDÜ: 62 böyle çiftin tam DICOM başlıkları (dilim düzeyi alanlar hariç)
    birebir aynı — ikinci kopya yeni bilgi taşımıyor ve işlenirse aynı
    öznitelik vektörü tabloya iki kez girip ICC/CV'yi yapay olarak iyileştirir.
    """
    monkeypatch.setattr(config, "DROP_DUPLICATE_RECONSTRUCTIONS", True, raising=False)
    recs = [
        _rec("a", "U1", "Scan 10 3.0 Br36", acq="11", series_number="1092"),
        _rec("b", "U2", "Scan 10 3.0 Br36", acq="11", series_number="92"),
        _rec("c", "U3", "Scan 10 3.0 Br36", acq="21", series_number="2092"),
    ]
    unique, dups = batch.dedup_and_takes(recs)
    kalan = {r["series_uid"] for r in unique}
    assert kalan == {"U2", "U3"}               # en küçük SeriesNumber tutulur
    recon = [d for d in dups if d.get("reason") == "recon"]
    assert len(recon) == 1 and recon[0]["series_uid"] == "U1"
    # Eleme take atamasını BOZMAMALI: iki ayrı çekim, iki take.
    assert {r["series_uid"]: r["take"] for r in unique} == {"U2": 1, "U3": 2}


def test_duplicate_reconstruction_kept_when_disabled(monkeypatch):
    """config.DROP_DUPLICATE_RECONSTRUCTIONS = False → hiçbir seri elenmez."""
    monkeypatch.setattr(config, "DROP_DUPLICATE_RECONSTRUCTIONS", False, raising=False)
    recs = [_rec("a", "U1", "P", acq="11", series_number="1092"),
            _rec("b", "U2", "P", acq="11", series_number="92")]
    unique, dups = batch.dedup_and_takes(recs)
    assert len(unique) == 2
    assert [d for d in dups if d.get("reason") == "recon"] == []


def test_series_without_acquisition_number_is_never_dropped(monkeypatch):
    """Kimlik kurulamıyorsa eleme YAPILMAZ — belirsizlikte veri atılmaz."""
    monkeypatch.setattr(config, "DROP_DUPLICATE_RECONSTRUCTIONS", True, raising=False)
    recs = [_rec("a", "U1", "P", acq=""), _rec("b", "U2", "P", acq="")]
    unique, dups = batch.dedup_and_takes(recs)
    assert len(unique) == 2
    assert [d for d in dups if d.get("reason") == "recon"] == []


# ============================================================ #
# run_totalsegmentator — komut çözümü + etiket haritası yedeği
# ============================================================ #
def test_resolve_totalseg_cmd(monkeypatch, tmp_path):
    fake_exe = tmp_path / "TotalSegmentator.exe"
    fake_exe.write_text("x")
    fake_py = tmp_path / "python.exe"
    fake_py.write_text("x")

    # İKİSİ DE VARSA MODÜL KAZANIR. Konsol exe'sinin içine onu KURAN python'un
    # mutlak yolu gömülüdür; taşınabilir bir kopyada o yol başka makinede yoktur
    # ve sarmalayıcı ya açılmaz ya da sistemdeki (paketteki değil) python'u
    # başlatır. Bu sıra bozulursa paket meslektaşın makinesinde çöker.
    monkeypatch.setattr(config, "TOTALSEG_EXE", fake_exe)
    monkeypatch.setattr(config, "PY311", fake_py)
    cmd = ts._resolve_totalseg_cmd(Path("ct.nii.gz"), Path("seg.nii.gz"))
    assert cmd[0] == str(fake_py) and "-m" in cmd
    assert "-ml" in cmd and "-ta" in cmd and "total" in cmd

    # py311 yok, exe var → yedek olarak exe
    monkeypatch.setattr(config, "PY311", None)
    cmd2 = ts._resolve_totalseg_cmd(Path("ct.nii.gz"), Path("seg.nii.gz"))
    assert cmd2[0] == str(fake_exe)
    monkeypatch.setattr(config, "TOTALSEG_EXE", None)
    monkeypatch.setattr(config, "PY311", fake_py)
    # ikisi de yok → net hata
    monkeypatch.setattr(config, "PY311", None)
    try:
        ts._resolve_totalseg_cmd(Path("ct.nii.gz"), Path("seg.nii.gz"))
        assert False, "RuntimeError bekleniyordu"
    except RuntimeError:
        pass


def test_shipped_label_map_exists():
    shipped = config.PROJECT_ROOT / "totalseg_total_labels.json"
    assert shipped.exists(), "totalseg_total_labels.json repoda olmalı (taşınabilirlik)"
    lm = json.loads(shipped.read_text(encoding="utf-8"))
    assert len(lm) == 117
    assert lm["1"] == "spleen" and lm["5"] == "liver"


# ============================================================ #
# Cihaz-bağımsız sınıflandırma — PCCT (NAEOTOM) + EID (Somatom)
# kVp YALNIZ DICOM metadata'sından (SeriesDescription 0008,103E → KVP tag)
# ============================================================ #
def _pcct(**kw):
    kw.setdefault("ManufacturerModelName", "NAEOTOM Alpha")   # foton-sayan işareti
    return FakeDS(**kw)


def _eid(**kw):
    kw.setdefault("ManufacturerModelName", "SOMATOM Edge Plus")
    return FakeDS(**kw)


def test_is_photon_counting():
    assert ss.is_photon_counting(FakeDS(ManufacturerModelName="NAEOTOM Alpha")) is True
    # ImageType'ta 'COUNT' → foton-sayan (model adı olmasa bile)
    assert ss.is_photon_counting(FakeDS(ImageType=["ORIGINAL", "PRIMARY", "AXIAL", "COUNT"])) is True
    assert ss.is_photon_counting(FakeDS(ManufacturerModelName="SOMATOM Edge Plus")) is False
    assert ss.is_photon_counting(FakeDS()) is False


def test_effective_kvp_pcct_vs_eid():
    # KARAR: kVp HER cihazda KVP etiketinden (0018,0060) okunur; SeriesDescription'a
    # güvenilmez. PCCT'de açıklama "80kVp" dese de KVP etiketi 120 ise → 120 (etiket kazanır).
    assert ss.effective_kvp(_pcct(SeriesDescription="80kVp_100mAs Qr40", KVP="120")) == 120.0
    assert ss.effective_kvp(_pcct(SeriesDescription="120kVp_100mAs", KVP="120")) == 120.0
    # EID (konvansiyonel): KVP etiketi kullanılır
    assert ss.effective_kvp(_eid(SeriesDescription="Scan 10  3.0  Br36", KVP="120")) == 120.0
    # VMI serisinde de etiket (120) esas — "keV" kVp SANILMAMALI
    assert ss.effective_kvp(_pcct(SeriesDescription="120kVp VMI70keV", KVP="120")) == 120.0
    # Opsiyon: KVP_FROM_DESCRIPTION=True eski davranışı geri getirir (açıklamadan oku)
    import config as _cfg
    _old = _cfg.KVP_FROM_DESCRIPTION
    try:
        _cfg.KVP_FROM_DESCRIPTION = True
        assert ss.effective_kvp(_pcct(SeriesDescription="80kVp_100mAs Qr40", KVP="120")) == 80.0
    finally:
        _cfg.KVP_FROM_DESCRIPTION = _old


def test_recon_type():
    native = FakeDS(ImageType=["ORIGINAL", "PRIMARY", "AXIAL", "THRESHOLD", "CT_SOM10 DET3D", "T1", "COUNT"])
    assert ss.recon_type(native) == "native"
    vmi70 = FakeDS(ImageType=["ORIGINAL", "PRIMARY", "AXIAL", "VMI", "CT_SOM10 DEMEP", "ME70KEV", "COUNT"])
    assert ss.recon_type(vmi70) == "VMI_70keV"
    vmi45 = FakeDS(ImageType=["ORIGINAL", "PRIMARY", "AXIAL", "VMI", "ME45KEV"])
    assert ss.recon_type(vmi45) == "VMI_45keV"
    topo = FakeDS(ImageType=["ORIGINAL", "PRIMARY", "LOCALIZER", "CT_SOM5 TOP"])
    assert ss.recon_type(topo) == "localizer"
    assert ss.recon_type(FakeDS()) == "native"   # ImageType yoksa


def test_is_target_series():
    ND = ["ORIGINAL", "PRIMARY", "AXIAL", "THRESHOLD"]   # native
    VMI = ["ORIGINAL", "PRIMARY", "AXIAL", "VMI", "ME70KEV"]
    TOPO = ["ORIGINAL", "PRIMARY", "LOCALIZER"]
    # PCCT 120kVp native → hedef
    ok, kvp, rt = ss.is_target_series(_pcct(Modality="CT", SeriesDescription="120kVp_x", KVP="120", ImageType=ND))
    assert ok and kvp == 120.0 and rt == "native"
    # PCCT 120kVp VMI70 → hedef (TARGET_RECON_TYPES=None → tüm rekonlar)
    ok, _, rt = ss.is_target_series(_pcct(Modality="CT", SeriesDescription="120kVp_x", KVP="120", ImageType=VMI))
    assert ok and rt == "VMI_70keV"
    # PCCT açıklama "80kVp" ama KVP etiketi 120 → HEDEF (kVp etiketten 120 okunur)
    ok, kvp, _ = ss.is_target_series(_pcct(Modality="CT", SeriesDescription="80kVp_x", KVP="120", ImageType=ND))
    assert ok and kvp == 120.0
    # KVP etiketi gerçekten 140 olan PCCT serisi → hedef DEĞİL
    ok, kvp, _ = ss.is_target_series(_pcct(Modality="CT", SeriesDescription="140kVp_x", KVP="140", ImageType=ND))
    assert not ok and kvp == 140.0
    # Topogram → hedef DEĞİL (localizer)
    ok, _, rt = ss.is_target_series(_pcct(Modality="CT", SeriesDescription="120kVp Topogram", KVP="140", ImageType=TOPO))
    assert not ok and rt == "localizer"
    # EID 120kVp (konvansiyonel) → hedef (KVP etiketinden)
    ok, kvp, rt = ss.is_target_series(_eid(Modality="CT", SeriesDescription="Scan 1 Br40", KVP="120", ImageType=ND))
    assert ok and kvp == 120.0 and rt == "native"


def test_recon_type_filter(monkeypatch):
    # TARGET_RECON_TYPES ile kısıtlama: yalnız native
    monkeypatch.setattr(config, "TARGET_RECON_TYPES", {"native"})
    VMI = ["ORIGINAL", "PRIMARY", "AXIAL", "VMI", "ME70KEV"]
    ok, _, _ = ss.is_target_series(_pcct(Modality="CT", SeriesDescription="120kVp", KVP="120", ImageType=VMI))
    assert not ok                                 # VMI izinli değil


# ============================================================ #
# Yeni mekanikler — sidecar / resume / atomik / watchdog / present
# ============================================================ #
def test_sidecar_and_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FEATURES", tmp_path)

    def tablolari_yaz(key: str) -> None:
        """Her modun seri tablosunu oluşturur (resume varlıklarını denetliyor)."""
        for mod in config.MIRP_MODES:
            p = tmp_path / key / fx.mode_csv_name(mod)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("image_mask_label\nspleen\n", encoding="utf-8")

    # eksik organlı sidecar
    fx._write_done_sidecar("K", {"a", "b", "c"}, {"a", "c"}, [])
    tablolari_yaz("K")
    d = json.loads((tmp_path / "K" / fx.DONE_SIDECAR).read_text(encoding="utf-8"))
    assert d["missing_organs"] == ["b"] and d["complete"] is False
    assert d["n_expected_organs"] == 3 and d["n_present_organs"] == 2
    # kısmi sonuç → resume ATLAR (kayıtlı), sonsuz döngü yok
    assert fx.series_is_complete("K") is True
    # tam sonuç → atlar
    fx._write_done_sidecar("K2", {"a"}, {"a"}, [])
    tablolari_yaz("K2")
    assert fx.series_is_complete("K2") is True
    # HİÇ organ (geçici tam başarısızlık) → resume YENİDEN dener
    fx._write_done_sidecar("K3", {"a", "b"}, set(), ["K3/a", "K3/b"])
    tablolari_yaz("K3")
    assert fx.series_is_complete("K3") is False
    # boş maske (expected=0) → atlar
    fx._write_done_sidecar("K4", set(), set(), [])
    tablolari_yaz("K4")
    assert fx.series_is_complete("K4") is True
    # sidecar yok → işle
    assert fx.series_is_complete("YOK") is False

    # BİR MODUN TABLOSU EKSİK → yeniden işle. Sidecar atomik yazılıyor ama
    # modlar sırayla yazıldığı için ikinci mod diske inmeden kesilme mümkün;
    # o durumda seri "tamamlandı" sayılırsa 2B tablosu kalıcı olarak eksik kalır.
    ikincil = [m for m in config.MIRP_MODES if m != config.PRIMARY_MODE]
    if ikincil:
        fx._write_done_sidecar("K5", {"a"}, {"a"}, [])
        tablolari_yaz("K5")
        (tmp_path / "K5" / fx.mode_csv_name(ikincil[0])).unlink()
        assert fx.series_is_complete("K5") is False


def test_resume_detects_changed_input(monkeypatch, tmp_path):
    """Serinin İÇERİĞİ değişirse resume onu yeniden işler.

    Sidecar `input_checksum`'ı yazıyordu ama geri okunmuyordu: bir seriye slice
    eklenince ct.nii.gz değişiyor, öznitelikler eski hacme ait kalıyor ve seri
    'tamamlanmış' sayılıyordu — sessizce yanlış tablo.
    """
    monkeypatch.setattr(config, "FEATURES", tmp_path / "features")
    monkeypatch.setattr(config, "NIFTI", tmp_path / "nifti")
    monkeypatch.setattr(config, "SEG", tmp_path / "seg")
    for alt, ad in ((config.NIFTI, "ct.nii.gz"), (config.SEG, "seg.nii.gz")):
        (alt / "K").mkdir(parents=True, exist_ok=True)
        (alt / "K" / ad).write_bytes(b"ilk-icerik")

    fx._write_done_sidecar("K", {"a"}, {"a"}, [])
    for mod in config.MIRP_MODES:
        (config.FEATURES / "K" / fx.mode_csv_name(mod)).write_text("x\n",
                                                                   encoding="utf-8")
    assert fx.series_is_complete("K") is True, "değişmemiş girdi yeniden işlenmemeli"

    # CT'ye slice eklenmiş gibi: içerik değişti
    (config.NIFTI / "K" / "ct.nii.gz").write_bytes(b"ilk-icerik-plus-yeni-slice")
    assert fx.series_is_complete("K") is False, "değişen CT yakalanmalı"


def test_atomic_write(tmp_path):
    p = tmp_path / "sub" / "x.json"
    fx._atomic_write_text(p, '{"a":1}')
    assert p.read_text(encoding="utf-8") == '{"a":1}'
    assert not list(tmp_path.rglob("*.tmp"))       # kalıntı yok


def test_watchdog_seconds(monkeypatch):
    monkeypatch.setattr(config, "MIRP_WATCHDOG_PER_ORGAN_S", 100)
    tasks = [{"roi_map": {"1": "a", "2": "b"}}, {"roi_map": {"3": "c"}}]  # 3 organ
    assert fx._watchdog_seconds(tasks) == max(1800.0, 100 * 3)
    # taban 1800 alt sınır
    assert fx._watchdog_seconds([{"roi_map": {"1": "a"}}]) == 1800.0


def test_watchdog_isolation_floor_is_tighter(monkeypatch):
    """İzolasyon tabanı havuz tabanından AYRI ve daha sıkı olmalı.

    REGRESYON: tek organ izole edilirken de 1800 sn taban uygulanıyordu (30 dk/organ)
    → 6 organlık öbek 3 saat sürüyordu. İzolasyon artık kendi (küçük) tabanını alır.
    """
    monkeypatch.setattr(config, "MIRP_WATCHDOG_PER_ORGAN_S", 100)
    one = [{"roi_map": {"1": "a"}}]
    assert fx._watchdog_seconds(one) == 1800.0                  # havuz tabanı (varsayılan)
    assert fx._watchdog_seconds(one, floor=600.0) == 600.0      # izolasyon tabanı
    many = [{"roi_map": {str(i): "x" for i in range(10)}}]       # 10 organ x 100 = 1000
    assert fx._watchdog_seconds(many, floor=600.0) == 1000.0     # ölçekleme tabanı aşarsa kazanır


def test_run_one_process_returns_and_does_not_wedge():
    """İzole süreç sonuç döndürmeli ve ARDIŞIK çağrılar kilitlenmemeli.

    REGRESYON (gerçek arıza): izolasyon ProcessPoolExecutor + terminate kullanıyordu.
    Takılan işçi öldürülünce havuzun iç çağrı-kuyruğu/kilitleri bozuk kalıyor ve bir
    SONRAKİ çağrı parent'ı KALICI kilitliyordu (bir koşuda 11 saat askı; watchdog dahil
    hiçbir şey kurtaramadı). Artık her çağrı TAZE Process+Queue kullanır → bozulma
    sızamaz. Burada iki ARDIŞIK çağrının da döndüğünü doğrularız (ikincisi kritik).
    """
    bogus = {"series_key": "t", "ct_path": "yok.nii.gz", "seg_path": "yok.nii.gz",
             "roi_map": {"1": "organ"}, "sitk_threads": 1}
    for i in range(2):
        res, hung = fx._run_one_process(bogus, timeout_s=180)
        assert not hung, f"{i+1}. çağrı zaman aşımına uğradı (kilitlenme regresyonu)"
        assert res is not None, f"{i+1}. çağrı sonuç döndürmedi"
        assert res.get("df") is None and res.get("error"), "bozuk görev hata döndürmeli"


def test_present_organs_nan_excluded():
    df = pd.DataFrame({"image_mask_label": ["liver", "air"],
                       "morph_volume": [1000.0, float("nan")], "stat_mean": [50.0, 10.0]})
    present = fx._present_organs(df)
    assert present == {"liver"}                    # NaN morph_volume → 'air' eksik sayılır


def test_spatial_sort_mixed_orientation():
    # farklı ImageOrientationPatient → fiziksel izdüşüme GÜVENME, InstanceNumber'a düş
    axial = FakeDS(ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, 5], InstanceNumber=2)
    coronal = FakeDS(ImageOrientationPatient=[1, 0, 0, 0, 0, -1], ImagePositionPatient=[0, 0, 0], InstanceNumber=1)
    items = [(Path("a.dcm"), axial), (Path("b.dcm"), coronal)]
    ordered = ss.spatial_sort(items)
    assert [p.name for p, _ in ordered] == ["b.dcm", "a.dcm"]   # InstanceNumber 1,2


def test_discover_series_groups_by_uid(monkeypatch, tmp_path):
    root = tmp_path / "src"; (root / "folderA").mkdir(parents=True)
    spec = {
        "folderA/a1.dcm": ("UIDA", "CT", 120), "folderA/a2.dcm": ("UIDA", "CT", 120),
        "folderA/b1.dcm": ("UIDB", "CT", 120),   # aynı klasör, FARKLI seri
        "folderA/c1.dcm": ("UIDC", "CT", 100),   # 100 kVp → dışla
        "folderA/d1.dcm": ("UIDD", "MR", 120),   # non-CT → dışla
    }
    for rel in spec:
        (root / rel).write_text("x")

    def fake_read(path):
        rel = str(Path(path).relative_to(root)).replace("\\", "/")
        if rel not in spec:
            return None
        uid, mod, kvp = spec[rel]
        return FakeDS(SeriesInstanceUID=uid, Modality=mod, KVP=str(kvp),
                      StudyInstanceUID="ST", SeriesDescription="d", SeriesNumber="1",
                      PatientID="P", StudyDate="20260101", StudyTime="0900", AcquisitionTime="0900")
    monkeypatch.setattr(batch, "_read_header", fake_read)
    recs = batch.discover_series(root)
    assert sorted(r["series_uid"] for r in recs) == ["UIDA", "UIDB"]   # blend YOK
    a = next(r for r in recs if r["series_uid"] == "UIDA")
    assert len(a["files"]) == 2


def _split_export_root(tmp_path, sops_a, sops_b, monkeypatch):
    """Aynı UID'nin iki klasöra bölündüğü sentetik dışa aktarım kurar."""
    root = tmp_path / "src"
    (root / "0000").mkdir(parents=True)
    (root / "0001").mkdir(parents=True)
    spec: dict[str, str] = {}
    for i, sop in enumerate(sops_a):
        rel = f"0000/a{i}.dcm"; spec[rel] = sop; (root / rel).write_text("x")
    for i, sop in enumerate(sops_b):
        rel = f"0001/b{i}.dcm"; spec[rel] = sop; (root / rel).write_text("x")

    def fake_read(path):
        rel = str(Path(path).relative_to(root)).replace("\\", "/")
        if rel not in spec:
            return None
        return FakeDS(SeriesInstanceUID="UID1", Modality="CT", KVP="120",
                      SOPInstanceUID=spec[rel], StudyInstanceUID="ST",
                      SeriesDescription="d", SeriesNumber="1", PatientID="P",
                      StudyDate="20260101", StudyTime="0900",
                      AcquisitionTime="0900", InstanceNumber=1)
    monkeypatch.setattr(batch, "_read_header", fake_read)
    return root


def test_discover_series_merges_series_split_across_folders(monkeypatch, tmp_path):
    # GERÇEK ARIZA: PACS dışa aktarımı 376 slice'lık bir seriyi 109 + 267 olarak
    # iki klasöra bölmüştü. Klasör bazlı gruplama bunları iki kayıt yapıp
    # 'aynı UID' diye birini ATIYORDU → 109 slice sessizce kayboluyordu.
    root = _split_export_root(tmp_path, ["s1", "s2", "s3"], ["s4", "s5"], monkeypatch)
    recs = batch.discover_series(root)
    assert len(recs) == 1, "bölünmüş seri TEK kayıt olmalı"
    r = recs[0]
    assert len(r["files"]) == 5, "her iki klasörün slice'ları da alınmalı"
    assert len(r["folders"]) == 2
    assert sorted(r["folder_counts"].values()) == [2, 3]
    assert r["n_dup_files"] == 0
    # dedup bunu artık kopya SAYMAMALI (yoksa yine parça atılır)
    unique, dups = batch.dedup_and_takes(recs)
    assert len(unique) == 1 and dups == []


def test_discover_series_drops_true_duplicate_slices(monkeypatch, tmp_path):
    # Aynı SOPInstanceUID iki dosyada = gerçek kopya slice: bir kez alınmalı,
    # yoksa hacimde aynı z iki kez görünür ve geometri bozulur.
    root = _split_export_root(tmp_path, ["s1", "s2"], ["s2", "s3"], monkeypatch)
    recs = batch.discover_series(root)
    assert len(recs) == 1
    r = recs[0]
    assert len(r["files"]) == 3, "benzersiz SOP sayısı kadar slice"
    assert r["n_dup_files"] == 1


def test_build_meta_paths_resolve_across_folders(monkeypatch, tmp_path):
    # Birleştirilmiş seride dosyalar iki klasörde; meta TAM yolları taşımalı,
    # yoksa series_dir + ad birleştirmesi var olmayan yol üretir.
    root = _split_export_root(tmp_path, ["s1", "s2"], ["s3"], monkeypatch)
    recs = batch.discover_series(root)
    meta = batch.build_meta(recs[0])
    assert meta["num_slices"] == 3
    assert len(meta["ordered_paths"]) == 3
    assert all(Path(p).exists() for p in meta["ordered_paths"])
    assert meta["n_folders"] == 2
    assert {Path(p).parent.name for p in meta["ordered_paths"]} == {"0000", "0001"}


def test_completeness_report_scoped(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FEATURES", tmp_path)
    fx._write_done_sidecar("run_a", {"x", "y"}, {"x"}, [])       # eksik: y
    fx._write_done_sidecar("orphan", {"z"}, set(), [])           # yabancı (bu run değil)
    n = batch.write_completeness_report(keys=["run_a"])          # sadece bu run
    rep = (tmp_path / "INCOMPLETE_report.txt").read_text(encoding="utf-8")
    assert n == 1 and "run_a" in rep and "orphan" not in rep


# ============================================================ #
# Entegrasyon (yavaş) — sentetik CT + maske, gerçek MIRP
# IBSI çıktı + paralel==sıralı özdeşliği
# ============================================================ #
def test_integration_mirp_parallel_identity(tmp_path, monkeypatch):
    if FAST:
        print("  (atlandı: --fast)")
        return
    # sentetik CT (40^3) + 4 organlı maske
    N = 44
    rng = np.random.default_rng(0)
    vol = rng.normal(0, 150, size=(N, N, N)).astype(np.float32)
    img = sitk.GetImageFromArray(vol); img.SetSpacing((1., 1., 1.))
    ct = tmp_path / "ct.nii.gz"; sitk.WriteImage(img, str(ct))
    m = np.zeros((N, N, N), np.uint8)
    for i, (a, b, c) in enumerate([(3, 3, 3), (3, 25, 3), (25, 3, 25), (25, 25, 25)], 1):
        m[a:a+12, b:b+12, c:c+12] = i
    seg = tmp_path / "seg.nii.gz"; s = sitk.GetImageFromArray(m); s.SetSpacing((1., 1., 1.))
    sitk.WriteImage(s, str(seg))
    lm = {"1": "a", "2": "b", "3": "c", "4": "d"}
    lmp = tmp_path / "label_map.json"; lmp.write_text(json.dumps(lm), encoding="utf-8")
    # Bu test paralel==sıralı IBSI özdeşliğini ölçüyor; organ SEÇİMİNDEN
    # bağımsız olmalı. Aksi halde config.TARGET_ORGANS daraldığında sentetik
    # organlar (a..d) filtrelenip test kendi konusuyla ilgisiz sebeple kırılır.
    monkeypatch.setattr(config, "TARGET_ORGANS", None)
    monkeypatch.setattr(config, "FEATURES", tmp_path / "features")
    monkeypatch.setattr(config, "COMBINED_FEATURES_CSV", tmp_path / "features" / "all.csv")

    meta = {"series_key": "PT__ser", "patient_id": "PT", "ct_path": str(ct),
            "seg_path": str(seg), "label_map_path": str(lmp)}

    # sıralı (1 işçi)
    monkeypatch.setattr(fx, "_plan_workers", lambda *a: (1, 22 * 10**9, 145 * 10**6))
    seq = fx.extract_for_series(dict(meta))
    # paralel (4 işçi → 4 öbek) — IBSI: sonuç bit-birebir aynı olmalı
    monkeypatch.setattr(fx, "_plan_workers", lambda *a: (4, 22 * 10**9, 145 * 10**6))
    par = fx.extract_for_series(dict(meta))

    assert seq is not None and par is not None
    assert len(seq) == len(par) == 4
    fam = [c for c in seq.columns if c.startswith(("morph_", "stat_", "cm_", "rlm_", "szm_"))]
    assert len(fam) > 40, "IBSI aileleri mevcut olmalı"
    # PARALEL == SIRALI (bit-birebir) — IBSI kritik
    a = seq.set_index("image_mask_label").sort_index()
    b = par.set_index("image_mask_label").sort_index()
    fcols = [c for c in a.columns if a[c].dtype.kind in "fi"]
    diff = np.nanmax(np.abs(a[fcols].to_numpy(float) - b[fcols].to_numpy(float)))
    assert diff < 1e-9, f"paralel != sıralı! maks fark {diff:.2e} (IBSI ihlali)"


# ============================================================ #
# pytest fixture olmadan da çalışan basit tmp_seg üreticisi
# ============================================================ #
def _make_tmp_seg(dirpath: Path):
    m = np.zeros((10, 10, 10), np.uint8)
    m[1:4, 1:4, 1:4] = 1
    m[5:8, 5:8, 5:8] = 5
    seg = dirpath / "seg.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(m), str(seg))
    return seg, [1, 5]


# ============================================================ #
# Bağımsız çalıştırıcı (pytest yoksa)
# ============================================================ #
class _MonkeyPatch:
    def __init__(self):
        self._undo = []
    def setattr(self, obj, name, val):
        old = getattr(obj, name)
        self._undo.append((obj, name, old))
        setattr(obj, name, val)
    def undo(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)


# ============================================================ #
# series_naming — DICOM metadata'sına göre isimlendirme
# ============================================================ #
def test_naming_fmt_number():
    # trailing-zero atılır, ondalık korunur, string/None güvenli
    assert sn._fmt_number(3.0) == "3"
    assert sn._fmt_number("3") == "3"
    assert sn._fmt_number(0.6) == "0.6"
    assert sn._fmt_number(1.5) == "1.5"
    assert sn._fmt_number(300.0) == "300"
    assert sn._fmt_number("100.0") == "100"
    assert sn._fmt_number(None) is None
    assert sn._fmt_number("") is None


def test_naming_kernel_token():
    # tek değer aynen; çok-değerli '-' ile birleşir (farklı serileri ayrıştırır)
    assert sn._kernel_token(FakeDS(ConvolutionKernel="Br36f")) == "Br36f"
    assert sn._kernel_token(FakeDS(ConvolutionKernel=["Br36f", "3"])) == "Br36f-3"
    assert sn._kernel_token(FakeDS()) is None


def test_naming_institution_code():
    assert sn._institution_code(FakeDS(InstitutionName="Radiologie USZ")) == "USZ"
    assert sn._institution_code(FakeDS(InstitutionName="CHUV Lausanne")) == "CHUV"
    # büyük-harf kısaltma yoksa tam ad sanitize edilir
    assert sn._institution_code(FakeDS(InstitutionName="clinique")) == "clinique"
    assert sn._institution_code(FakeDS()) is None


def test_naming_qir_level_from_description_only():
    # Q-seviyesi YALNIZ SeriesDescription'dan; 'Qr40' kernel'i yanlışlıkla eşleşmez
    assert sn._qir_level(FakeDS(SeriesDescription="120kVp_100mAs_1mm_300FOV 1,00 Qr40 Q3")) == "Q3"
    assert sn._qir_level(FakeDS(SeriesDescription="80kVp Qr64 Q1")) == "Q1"
    assert sn._qir_level(FakeDS(SeriesDescription="Scan 10  3.0  Br36")) is None
    assert sn._qir_level(FakeDS(SeriesDescription="Qr40 only")) is None   # kernel'deki Q sayılmaz


def test_naming_eid_matches_scheme():
    # EID (konvansiyonel): tarih_kurum_üretici_model_kernel_kvp_mas_fov_kesit; rekon eki YOK
    ds = FakeDS(StudyDate="20260629", InstitutionName="Radiologie USZ",
                Manufacturer="SIEMENS", ManufacturerModelName="SOMATOM Edge Plus",
                ConvolutionKernel="Br36f", KVP="120", Exposure="100",
                ReconstructionDiameter="300", SliceThickness="3",
                SeriesDescription="Scan 10  3.0  Br36", ImageType=["ORIGINAL", "PRIMARY"])
    assert sn.build_name(ds) == "20260629_USZ_SIEMENS_SOMATOM_Edge_Plus_Br36f_120_100_300_3"


def test_naming_pcct_includes_qir_and_recon():
    # PCCT: kVp SeriesDescription'dan; sona Q-seviyesi + rekon (VMI/native) eklenir
    common = dict(StudyDate="20260629", InstitutionName="Radiologie USZ",
                  Manufacturer="Siemens Healthineers", ManufacturerModelName="NAEOTOM Alpha",
                  ConvolutionKernel="Qr40f", KVP="120", Exposure="100",
                  ReconstructionDiameter="300", SliceThickness="1")
    vmi = FakeDS(SeriesDescription="120kVp_100mAs_1mm_300FOV 1,00 Qr40 Q3",
                 ImageType=["DERIVED", "PRIMARY", "COUNT", "VMI", "ME70KEV"], **common)
    nat = FakeDS(SeriesDescription="120kVp_100mAs_1mm_300FOV 1,00 Qr40 Q1",
                 ImageType=["ORIGINAL", "PRIMARY", "COUNT"], **common)
    n_vmi = sn.build_name(vmi)
    n_nat = sn.build_name(nat)
    assert n_vmi.endswith("_Qr40f_120_100_300_1_Q3_VMI70keV"), n_vmi
    assert n_nat.endswith("_Qr40f_120_100_300_1_Q1_native"), n_nat
    assert n_vmi != n_nat            # aynı kVp/mAs/FOV/kesit/kernel ama Q + rekon ayırır


def test_naming_collision_resolution_lossless():
    # Şemanın ayırt edemediği (aynı ad) FARKLI seriler benzersizleşir; hiçbiri kaybolmaz
    def e(name, num, uid, folder):
        return {"series_key": f"k_{uid}", "name": name, "folder": Path(folder),
                "series_uid": uid, "series_number": num, "take": None, "fields": {}}
    ents = [e("A", 11, "u11", "f11"), e("A", 12, "u12", "f12"),
            e("A", 13, "u13", "f13"), e("B", 5, "u5", "f5")]
    an.resolve_collisions(ents)
    names = [x["name"] for x in ents]
    assert len(set(names)) == len(names)           # hepsi benzersiz
    assert "B" in names                            # tekil dokunulmaz
    assert {"A_s11", "A_s12", "A_s13"} <= set(names)   # SeriesNumber ile ayrışır


def test_naming_collision_uid_fallback():
    # SeriesNumber yoksa UID kuyruğuyla ayrışır
    def e(uid, folder):
        return {"series_key": f"k_{uid}", "name": "X", "folder": Path(folder),
                "series_uid": uid, "series_number": "", "take": None, "fields": {}}
    ents = [e("aaa", "fa"), e("bbb", "fb")]
    an.resolve_collisions(ents)
    assert len({x["name"] for x in ents}) == 2
    assert all(x["name"].startswith("X_") for x in ents)


def _run_standalone() -> int:
    import inspect
    import tempfile
    scratch = Path(tempfile.mkdtemp(prefix="phantomx_test_"))
    seg_dir = scratch / "seg"; seg_dir.mkdir()
    tmp_seg_val = _make_tmp_seg(seg_dir)

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        params = inspect.signature(fn).parameters
        kwargs = {}
        mp = None
        if "monkeypatch" in params:
            mp = _MonkeyPatch(); kwargs["monkeypatch"] = mp
        if "tmp_path" in params:
            d = scratch / name; d.mkdir(exist_ok=True); kwargs["tmp_path"] = d
        if "tmp_seg" in params:
            kwargs["tmp_seg"] = tmp_seg_val
        try:
            fn(**kwargs)
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {name}: {exc!r}")
            traceback.print_exc()
            failed += 1
        finally:
            if mp is not None:
                mp.undo()
    print(f"\n{'='*50}\n{passed} geçti, {failed} başarısız ({len(tests)} test)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run_standalone())


def test_ts_device_follows_environment(monkeypatch):
    """Hesap cihazı SABİT olmamalı — makineye göre belirlenir.

    Paketlenmiş torch yalnız belli GPU mimarilerini destekliyor (sm_75+).
    Daha eski bir kartta CUDA "var" görünür ama çekirdek başlatılamaz ve koşum
    ortasında çöker; NVIDIA hiç yoksa da CPU'ya düşmek gerekir. Karar torch'un
    kurulu olduğu ortamda ölçülüp `PHANTOMX_TS_DEVICE` ile geçiriliyor.
    """
    import importlib
    import config as _cfg

    for deger in ("cpu", "gpu"):
        monkeypatch.setenv("PHANTOMX_TS_DEVICE", deger)
        importlib.reload(_cfg)
        assert _cfg.TS_DEVICE == deger

    monkeypatch.delenv("PHANTOMX_TS_DEVICE", raising=False)
    importlib.reload(_cfg)
    assert _cfg.TS_DEVICE == "gpu", "değişken yoksa eski davranış korunmalı"


# ============================================================ #
# run_and_verify — kapsam dışı seri, hatalı seriden ayrılmalı
# ============================================================ #
def _kur_cikti_agaci(root, seriler):
    """Küçük ama GERÇEK bir çıktı ağacı kurar (nii.gz + sidecar + tablolar)."""
    import run_and_verify as rv

    config.set_output_root(root)
    config.ensure_dirs()
    ortak_surumler = {"mirp": "2.5.0"}
    satirlar = []
    for key, organlar, eksik in seriler:
        # ct + seg AYNI ızgarada (geometri denetimi geçsin)
        img = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.int16))
        for hedef in (config.NIFTI / key / "ct.nii.gz",
                      config.SEG / key / "seg.nii.gz"):
            hedef.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img, str(hedef))

        (config.FEATURES / key).mkdir(parents=True, exist_ok=True)
        (config.FEATURES / key / fx.DONE_SIDECAR).write_text(json.dumps({
            "series_key": key,
            "n_expected_organs": len(organlar) + len(eksik),
            "n_present_organs": len(organlar),
            "missing_organs": eksik,
            "absent_target_organs": [] if (organlar or eksik) else ["aorta", "spleen"],
            "settings_hash": "AYNI", "library_versions": ortak_surumler,
        }, ensure_ascii=False), encoding="utf-8")

        if organlar:
            df = pd.DataFrame({"series_key": key, "image_mask_label": organlar,
                               "morph_volume": [100.0] * len(organlar)})
            for mod in config.MIRP_MODES:
                df.to_csv(config.FEATURES / key / fx.mode_csv_name(mod), index=False)
            satirlar.append(df.assign(series=key))

    birlesik = (pd.concat(satirlar, ignore_index=True) if satirlar
                else pd.DataFrame(columns=["series_key", "image_mask_label",
                                           "morph_volume", "series"]))
    import batch_120kvp as _b
    for mod in config.MIRP_MODES:
        birlesik.to_csv(_b.mode_combined_path(mod), index=False)
    return rv


def test_verify_exempts_series_whose_targets_are_out_of_view(tmp_path, monkeypatch):
    """Hedef organı GÖRÜNTÜDE OLMAYAN seri, tabloda eksik sayılmamalı.

    ÖLÇÜLDÜ (20260312 seti): 78 slice'lık bir seri fantomun yalnız kraniyal
    78 mm'sini kapsıyor (eşi 373 mm), dolayısıyla dalak ve aorta görüntüde yok.
    Hat bunu doğru şekilde `absent_target_organs` diye raporluyor, ama doğrulama
    yine de o seriyi tabloda bekleyip ENGELLEYİCİ HATA veriyordu: 140/140
    işlenmiş, sağlam bir çıktı 'HATA VAR' damgası alıyordu.
    """
    monkeypatch.setattr(config, "WRITE_OVERLAYS", False, raising=False)
    rv = _kur_cikti_agaci(tmp_path, [
        ("SERI_A", ["aorta", "spleen"], []),   # normal
        ("SERI_B", [], []),                    # hedefler görüntüde yok
    ])
    errors, warnings, info = rv.verify(["SERI_A", "SERI_B"])
    assert errors == [], f"kapsam dışı seri hata üretmemeli: {errors}"
    assert any("satır vermiyor" in s for s in info)


def test_verify_still_fails_when_extraction_lost_an_organ(tmp_path, monkeypatch):
    """Muafiyet GERÇEK kaybı örtmemeli.

    Organ maskede VARDI ama özniteliği hesaplanamadıysa (missing_organs) bu
    kayıptır ve engelleyici hata olarak kalmalıdır — yukarıdaki muafiyet yalnız
    hiç bulunmayan organlar içindir.
    """
    monkeypatch.setattr(config, "WRITE_OVERLAYS", False, raising=False)
    rv = _kur_cikti_agaci(tmp_path, [
        ("SERI_A", ["aorta", "spleen"], []),
        ("SERI_C", ["aorta"], ["spleen"]),     # spleen çıkarılamadı
    ])
    errors, _w, _i = rv.verify(["SERI_A", "SERI_C"])
    assert any("çıkarım başarısız" in e for e in errors), errors


def test_verify_still_fails_when_a_processed_series_is_absent_from_the_table(
        tmp_path, monkeypatch):
    """Satır vermesi BEKLENEN bir seri tabloda yoksa hata verilmeli.

    Muafiyetin kapsamı dar tutulmalı: yalnız 0 organlı seriler muaf. Organ
    üretmiş bir seri tablodan düşerse bu sessizce geçmemeli.
    """
    monkeypatch.setattr(config, "WRITE_OVERLAYS", False, raising=False)
    rv = _kur_cikti_agaci(tmp_path, [("SERI_A", ["aorta", "spleen"], [])])
    # SERI_A'yı tablodan sil (seri tablosu diskte duruyor ama birleşik tablo boş)
    import batch_120kvp as _b
    for mod in config.MIRP_MODES:
        pd.DataFrame(columns=["series_key", "image_mask_label", "morph_volume",
                              "series"]).to_csv(_b.mode_combined_path(mod), index=False)
    errors, _w, _i = rv.verify(["SERI_A"])
    assert any("eksik" in e for e in errors), errors


def test_take_column_is_recovered_from_the_folder_name():
    """`take` sütunu klasör adından kurtarılmalı — hepsi 1 olmamalı.

    Seri tabloları `take` taşımıyor, bu yüzden `tidy_fields`'ın
    `row.get("take") or 1` varsayılanı devreye girip TÜM satırları take1
    yapıyordu. ÖLÇÜLDÜ (20260312 seti): 139 serinin 69'u take2 olduğu hâlde
    tabloda tek bir take2 yoktu ve test-retest eşleştirmesi kurulamıyordu.
    """
    df = pd.DataFrame({
        "series_key": ["A", "B"],
        "series_dir": ["USZ_X_20260312_Br40f_120_100_300_1_PolyE_take1_recon1mm",
                       "USZ_X_20260312_Br40f_120_100_300_1_PolyE_take2_recon1mm"],
        "image_mask_label": ["aorta", "aorta"],
        "recon_type": ["native", "native"],
        "kvp": [120.0, 120.0],
    })
    out = batch._enrich_identity(df)
    assert list(out["take"]) == [1, 2], f"take kurtarılamadı: {list(out['take'])}"
    # Ad da ayrışmalı (aynı protokolün iki çekimi tek satıra çökmemeli)
    assert out["series"].nunique() == 2
