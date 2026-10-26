"""Merkezi yapılandırma — tüm yollar, sabitler ve MIRP ayarları burada.

Pipeline'ın tamamı Python 3.14 altında (pydicom + MIRP + SimpleITK) koşar.
Yalnızca TotalSegmentator ayrı bir ortamda (Python 3.11 + PyTorch/CUDA) kurulu
olduğu için subprocess ile çağrılır; bkz. PY311 / TOTALSEG_EXE.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# --------------------------------------------------------------------------- #
# Ortamlar — TAŞINABİLİR tespit (her bilgisayarda farklı olabilir)
# --------------------------------------------------------------------------- #
# TotalSegmentator + torch(CUDA) ayrı bir Python'da kuruludur (MIRP orada YOK).
# Sıra: (1) ortam değişkeni override, (2) PATH'te otomatik ara, (3) bu makinedeki
# bilinen yol (yalnızca varsa). Böylece config'i düzenlemeden başka makinede çalışır.
#
# Başka bir makinede farklıysa şu ortam değişkenlerini ayarlayın:
#   PHANTOMX_TOTALSEG_EXE = TotalSegmentator.exe tam yolu
#   PHANTOMX_PY311        = TotalSegmentator'ın kurulu olduğu python.exe tam yolu

_FALLBACK_TS_EXE = Path(
    r"C:\Users\Administrator\AppData\Local\Programs\Python\Python311\Scripts\TotalSegmentator.exe"
)
_FALLBACK_PY311 = Path(
    r"C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
)


def _detect_totalseg_exe() -> Path | None:
    env = os.environ.get("PHANTOMX_TOTALSEG_EXE")
    if env:
        return Path(env)
    found = shutil.which("TotalSegmentator")   # PATH'te ara (taşınabilir)
    if found:
        return Path(found)
    return _FALLBACK_TS_EXE if _FALLBACK_TS_EXE.exists() else None


def _detect_py311() -> Path | None:
    env = os.environ.get("PHANTOMX_PY311")
    if env:
        return Path(env)
    return _FALLBACK_PY311 if _FALLBACK_PY311.exists() else None


TOTALSEG_EXE = _detect_totalseg_exe()   # None olabilir → çalıştırmada net hata verilir
PY311 = _detect_py311()                 # etiket haritası için (yoksa TS'in kendi env'i)

# --------------------------------------------------------------------------- #
# Kök dizinler
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent

# Çıktı kökü. Varsayılan proje içi `data/`, ama koşumda KAYNAK klasöre bağlanır
# (bkz. set_output_root): her veri seti kendi çıktısını yanında taşısın, kod
# klasörü veriyle dolmasın ve bir veri setini taşımak çıktısını da taşısın.
DATA = PROJECT_ROOT / "data"

DICOMS = DATA / "dicoms"              # seçilen serilerin DICOM'ları (seri başına klasör)
NIFTI = DATA / "nifti"                # 120 kVp serilerin ct.nii.gz'leri
SEG = DATA / "segmentation"           # multilabel seg.nii + label_map.json
FEATURES = DATA / "features"          # per-series + birleşik CSV
OVERLAYS = DATA / "overlay"           # segmentasyonun görsel kontrolü (PNG)

# Segmentasyon QC görüntüsü üretilsin mi. Seri başına ~1 sn ve birkaç yüz KB;
# karşılığında maskenin doğru yerde olduğu gözle doğrulanabiliyor. Yanlış bir
# maske aksi halde ancak öznitelikler tuhaf çıktığında fark ediliyor.
WRITE_OVERLAYS = True

# Birleşik öznitelik tablosu
COMBINED_FEATURES_CSV = FEATURES / "all_features_combined.csv"

# Seri adındaki kurum kodu.
#   OVERRIDE : verilirse DICOM'a BAKILMADAN bu kullanılır (elle sabitleme).
#   FALLBACK : yalnız InstitutionName BOŞ olduğunda devreye girer.
# Bu projede NAEOTOM dışa aktarımlarında InstitutionName boş geliyor; aynı
# çalışmanın SOMATOM serileri 'Radiologie USZ' diyor. Yedek olmasa PCCT
# serileri 'NA' koduyla adlanır ve iki cihazın adları birbirini tutmazdı.
INSTITUTION_OVERRIDE: str | None = None
INSTITUTION_FALLBACK: str | None = "USZ"

# --------------------------------------------------------------------------- #
# kVp filtreleme
# --------------------------------------------------------------------------- #
TARGET_KVP = 120.0
KVP_TOLERANCE = 1.0   # bazı cihazlar 119/121 yazabilir → ±1 tolerans

# --- kVp kaynağı: HER CİHAZDA DICOM KVP ETİKETİ (0018,0060) esas alınır ---
# KARAR (kullanıcı): kVp YALNIZCA KVP etiketinden okunur; SeriesDescription'daki
# "80kVp/100kVp/..." metnine GÜVENİLMEZ.
#  NEDEN (ampirik, bu PCCT veri setinde doğrulandı): NAEOTOM Alpha'da tüp voltajı
#  120 kV'de sabit; "80kVp"/"100kVp" etiketli seriler de KVP etiketinde 120 raporlar
#  (bunlar aynı 120 kV çekiminden türetilmiş farklı PolyE rekonstrüksiyonlarıdır,
#  gerçek 80/100 kV ÇEKİMİ DEĞİL). Dolayısıyla gerçek çekim kVp'si = KVP etiketi.
#  Farklı PolyE/rekon türevleri isimlendirmede SeriesNumber ekiyle ayrışır (kayıp yok).
# EID/Somatom'da da KVP etiketi zaten güvenilir. Klasör/dosya adına ASLA bakılmaz.
# (True yapılırsa PCCT'de eski davranış: kVp SeriesDescription'dan okunur.)
KVP_FROM_DESCRIPTION = False

# --- Rekonstrüksiyon tipi filtresi (PCCT: native/PolyE + VMI spektral) ---
# Topogram/scout (LOCALIZER) HER ZAMAN hariç tutulur (hacim değil).
# TARGET_RECON_TYPES: None → tüm rekon tipleri (native + VMI, her biri ayrı işlenir
#   ve çıktıda 'recon_type' sütunuyla etiketlenir). Kısıtlamak için bir küme verin,
#   ör. {"native"} yalnız polienerjik, {"VMI_70keV"} yalnız 70 keV VMI.
# EID verisinde tüm seriler 'native' sınıfına düşer → None sorunsuz çalışır.
TARGET_RECON_TYPES = None

# Topogram/scout (ImageType'ta LOCALIZER) her zaman hariç (hacim değil, tek dilim).
EXCLUDE_LOCALIZER = True

# --- Aynı çekimin ikinci kez rekonstrükte edilmiş kopyaları ---
# Bazı dışa aktarımlarda bir çekim, aynı parametrelerle İKİ KEZ rekonstrükte
# edilip iki ayrı seri olarak yazılıyor: aynı AcquisitionNumber, aynı protokol,
# farklı SeriesInstanceUID ve SeriesTime.
#
# ÖLÇÜLDÜ (20260312 test-retest, 202 seri): 62 böyle çift var ve çiftlerin tam
# DICOM başlıkları — dilim/instans düzeyi alanlar dışında — BİREBİR aynı
# (0 farklı etiket). Yani ikinci kopya yeni bilgi taşımıyor; işlenirse aynı
# öznitelik vektörü tabloya ikinci kez giriyor ve tekrarlanabilirlik
# istatistiklerini (ICC/CV) yapay olarak iyileştiriyor.
#
# True  → her (protokol, çekim) ikilisinden yalnız BİR seri işlenir; atılan
#         seriler DUPLICATES_report.txt'e YAZILIR (sessiz kayıp yok) ve
#         kaynak dosyalara dokunulmaz.
# False → hepsi işlenir (eski davranış); kopyalar ad çakışması çözümüyle
#         `_s<SeriesNumber>` ekini alır.
DROP_DUPLICATE_RECONSTRUCTIONS = True

# --------------------------------------------------------------------------- #
# TotalSegmentator
# --------------------------------------------------------------------------- #
TS_TASK = "total"     # ~117 anatomik yapı

# Hesap cihazı SABİT DEĞİL. Paketlenmiş torch yalnız belli GPU mimarilerini
# destekliyor (bu derlemede sm_75 ve üstü, yani GTX 16 / RTX 20 serisi ve
# sonrası); daha eski bir kartta — örneğin sm_61 olan MX330'da — CUDA "var"
# görünür ama çekirdek başlatılamaz ve koşum ortasında çöker. VRAM de ayrı bir
# eşik: TotalSegmentator'ın `total` görevi 2 GB'lık bir kartta belleğe sığmaz.
#
# Bu yüzden karar, torch'un KURULU OLDUĞU ortamda ölçülüp `PHANTOMX_TS_DEVICE`
# ile buraya geçirilir (bkz. paketleyicinin başlatıcısı). Değişken yoksa
# eskisi gibi "gpu" varsayılır.
TS_DEVICE = os.environ.get("PHANTOMX_TS_DEVICE", "gpu")

# --- Hangi yapılardan öznitelik çıkarılsın (ROI seçimi) ---
# None → maskede bulunan TÜM yapılar (117'ye kadar).
# Bir küme verilirse yalnız o yapılar işlenir. Süre neredeyse tamamen organ
# sayısıyla ölçekleniyor (organ başına ~2-3 sn + öbek başına bir yeniden
# örnekleme), yani seçim en büyük hızlanma kaldıracı.
#
# AD yazılır, etiket NUMARASI değil: numaralar TS sürümleri arasında
# değişebilir, adlar sabittir. Geçerli adlar totalseg_total_labels.json'da.
# Buradaki bir ad 'total' görevinde yoksa çıkarım en başta net bir hatayla
# durur (bir yazım hatası sessizce 0 organlık "tamamlanmış" seri üretmesin).
#
# iliopsoas TS'te sol/sağ iki ayrı yapıdır; "iliopsoas kası" ikisini kapsar.
TARGET_ORGANS: set[str] | None = {
    "spleen",
    "aorta",
}
# KARAR (ölçümle): iliopsoas bu taramalarda yalnız kraniyal ucundan kapsanıyor
# (~6 mL, gerçek kas 200-400 mL) ve rekonstrüksiyonlar arası %35-40 oynuyor —
# aynı fiziksel nesnede bu kadar sapma, ölçümün temsili olmadığını gösteriyor.
# Karaciğer bütün olarak çok kararlı (CV %0,28) ama segment bölümlemesi fantomda
# damar olmadığı için anatomik dayanaktan yoksun. Kalan iki organ en
# tekrarlanabilir olanlar: spleen CV %0,34, aorta CV %0,73.
#
# Daraltma geri alınabilir: seri başına features.csv'ler 5 organı da içeriyor,
# bu listeyi genişletip birleşik tabloyu yeniden derlemek yeterli (MIRP her
# ROI'yi bağımsız hesaplar, yeniden çıkarım gerekmez).
# TS subprocess üst zaman sınırı (saniye): bir seri takılırsa tüm batch'i
# dondurmasın.
#
# 7200 (2 saat) idi ve ölçümlere göre aşırıydı: TS bir seride GPU'da ~1 dk,
# CPU'da ~5 dk sürüyor (sahada 72 serilik bir koşumda 6 seri üst üste ölçüldü).
# 2 saat, gözlenenin ~24 katı. Sahada bir seri takıldığında koşum boşuna
# 2 saat bekledi. 3600 hâlâ CPU'daki normalin ~12 katı, yani yavaş bir
# makinede meşru bir seriyi kesme riski düşük; ama kurtarma süresi yarıya iner.
#
# Paketle gelen model ağırlıkları hazır olduğu için "ilk koşumda indirme"
# gerekçesi artık geçerli değil (bkz. models\nnunet\results).
#
# Çok yavaş bir makinede yetmezse ortam değişkeniyle yükseltilebilir.
TS_TIMEOUT_S = int(os.environ.get("PHANTOMX_TS_TIMEOUT_S", "3600"))

# --------------------------------------------------------------------------- #
# MIRP — IBSI uyumlu CT radiomics ayarları
# --------------------------------------------------------------------------- #
# mirp.extract_features(**MIRP_SETTINGS, image=..., mask=..., roi_name=...) biçiminde
# açılarak kullanılır. Parametre adları mirp 2.5.0 kaynağından doğrulandı.
MIRP_SETTINGS = {
    "image_modality": "CT",
    # 1x1x1 mm izotropik yeniden örnekleme
    "new_spacing": 1.0,
    # Tricubic — IBSI Config E'nin kullandığı interpolasyon.
    "spline_order": 3,
    # KAPALI OLMAK ZORUNDA. MIRP'in varsayılanı True ve yeniden örnekleme
    # öncesi Gauss ön-yumuşatması uyguluyor; aynı bayrak ikili maskeye de
    # geçiyor (standardWorkflow, mask.register çağrısı). IBSI'nin yeniden
    # örnekleme tanımında böyle bir ön-filtre yok: bu projenin kendi IBSI
    # doğrulaması (CalibraMics Thesis/2 - Radiomics Feature Extraction/
    # IBSI_test/validate_ibsi.py) interpolasyon yapan HER konfigürasyonda
    # (B, C, D, E) ve IBSI-2'de anti_aliasing=False kullanıyor.
    #
    # Ayrıca mirp/_image_processing/anti_aliasing.py içindeki "yalnızca
    # küçültmede uygula" koruması yorum satırı yapılmış durumda, yani filtre
    # büyütmede ve spacing zaten 1 mm olduğunda bile çalışıyor.
    #
    # Ölçüldü (0,586x0,586x1,0 mm dokulu küre -> 1 mm izotropik): açık/kapalı
    # farkı 177 özelliğin 145'ini değiştiriyor; stat_min 7 -> -57.
    "anti_aliasing": False,
    # Sabit bin genişliği 25 HU (IBSI CT için önerilen)
    "base_discretisation_method": "fixed_bin_size",
    "base_discretisation_bin_width": 25.0,
    # Hiçbir vokseli ELEMEYEN aralık; yalnızca fixed-bin-size gri seviye
    # orijinini sabitlemek için var (MIRP'te ayrı bir çapa parametresi yok).
    #
    # Eski değer [-1000, 400] yumuşak doku aralığıydı ve 117 yapının HEPSİNE
    # uygulanıyordu, yani kemiği tanımlayan yoğunlukları siliyordu: ölçüldü,
    # vertebrae_T10 voksellerinin %69,5'i, alt kaburgaların %38-58'i eleniyordu
    # (karaciğer/dalak gibi yumuşak dokularda kayıp %0). Geriye kalan kemik
    # özellikleri kortikal kemiği değil iliği tanımlıyordu.
    #
    # Yeni sınırlar 20260227_USZ_PCCT'de maske içi ölçülen gerçek aralığa
    # (-1047 .. 1625 HU) pay bırakacak şekilde seçildi; Ng <= 177, pratik.
    "resegmentation_intensity_range": [-1300.0, 3100.0],
    # --- Yoğunluk-hacim histogramı (IVH) ayrımlaştırması ---
    # BEYAN: "none" = yoğunluklar yeniden ayrımlaştırılmadan kullanılır.
    # CT için IBSI'nin öngördüğü davranış budur: HU kalibre ve tam sayı
    # değerlidir, dolayısıyla IVH doğrudan yoğunluklar üzerinden hesaplanır.
    #
    # DÜZELTME: burada eskiden "bin sayısı maskede bulunan farklı gri seviye
    # sayısına eşit olur" yazıyordu. ÖLÇÜLDÜ, yanlış: "none" iken MIRP yoğunluk
    # aralığı olarak YUKARIDAKİ RESEGMENTASYON ARALIĞINI kullanıyor
    # (mirp/_images/base_mask.py:428) ve histogramı o aralığın tamamı boyunca
    # kuruyor (ivh_features.py:80) → 3100-(-1300)+1 = 4401 seviye, maskede
    # bulunmayanlar sıfır sayımla dolduruluyor (284 satırda ölçüldü: 4401
    # seviyenin yalnız 296'sı dolu).
    #
    # SONUCU: aynı aralık IVH'nin gamma eksenini de belirliyor. Kesim noktaları
    # v10=-860, v25=-200, v50=+900, v75=+2000, v90=+2660 HU; yumuşak dokuda
    # gözlenen span (stat_min -729..-29, stat_max 136..1100) yalnız gamma
    # 0,13-0,55 aralığını dolduruyor. Bu yüzden ivh_v10/v50/v75/v90 ve
    # ivh_diff_v10_v90 SABİT (0 ya da 1) çıkıyor, ivh_diff_v25_v75 ise
    # ivh_v25'in birebir kopyası. Bu yedi hacim kesri tablodan ÇIKARILIYOR
    # (feature_extraction_mirp.IVH_VOLUME_FRACTION_FEATURES) — kardeş hattın
    # referans çalışma kitabı da yalnız yedi ivh_i* sütunu raporluyor.
    # Yoğunluk kesirleri (ivh_i10..i90, ivh_diff_i*) etkilenmiyor ve kalıyor.
    # MIRP'te CT'nin varsayılanı da tam olarak bu (ct_image.py'deki
    # get_default_ivh_discretisation_method "none" döndürür; "fixed_bin_number"a
    # düşme yalnız generic/transformed görüntülerde olur).
    # Buraya YAZILMASININ nedeni değeri değiştirmek değil, MIRP varsayılanı
    # kayarsa sessizce başka bir yöntemin devreye girmemesi — 14 ivh_*
    # özniteliğinin tamamını bu parametre belirliyor.
    "ivh_discretisation_method": "none",
    # Tüm IBSI öznitelik aileleri
    "base_feature_families": "all",
    "ibsi_compliant": True,
}

# --------------------------------------------------------------------------- #
# Doku toplama (texture aggregation) — 3B ve 2B mod ayarları
# --------------------------------------------------------------------------- #
# IBSI doku matrislerini (GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM) tek bir
# sayıya indirmek için birden çok TOPLAMA YÖNTEMİ tanımlar ve hangisinin
# kullanıldığının BEYAN EDİLMESİNİ ister. Yöntem seçimi serbesttir, beyan
# zorunludur.
#
# Bu sözlükler verilmediğinde MIRP kendi varsayılanına düşüyordu ve o varsayılan
# yalnız TEK bir 3B yöntem üretiyor (ölçüldü, mirp 2.5.0):
#     by_slice=False -> glcm/glrlm = ['3d_volume_merge'], diğerleri = ['3d']
#     by_slice=True  -> glcm/glrlm = ['2d_slice_merge'],  diğerleri = ['2d']
# Yani hem 2B tamamen hesaplanmıyordu, hem de 3B'de iki IBSI yönteminden yalnız
# biri tabloya giriyordu. Kardeş hat (pcct-radiomics-mirp,
# extract_..._v4.py:150-186) ikisini de açıkça veriyor; aynı kurumda aynı
# fantom için üretilen tabloların karşılaştırılabilir olması adına buradaki
# listeler onunla BİREBİR aynı tutuldu.
#
# Ekleme KATKISALDIR: `3d_volume_merge` sütunlarının değerleri değişmez, yanına
# `3d_average` sütunları gelir. Yine de ayar hash'i değiştiği için eski
# çıkarımlar bayatlar ve yeniden hesaplanır (bkz. _stale_reason).
MIRP_SETTINGS_3D = {
    "by_slice": False,
    "glcm_spatial_method": ["3d_average", "3d_volume_merge"],
    "glrlm_spatial_method": ["3d_average", "3d_volume_merge"],
    "glszm_spatial_method": ["3d"],
    "gldzm_spatial_method": ["3d"],
    "ngtdm_spatial_method": ["3d"],
    "ngldm_spatial_method": ["3d"],
}

# `by_slice=True` yalnız doku matrislerini değil TÜM iş akışını 2B yapar:
# yeniden örnekleme düzlem içinde kalır (dilim kalınlığı korunur) ve morfoloji
# dilim dilim hesaplanır. IBSI'nin 2B tanımı budur; bu yüzden 2B satırları
# 3B satırlarıyla AYNI TABLODA birleştirilemez — ayrı sayfada durur.
MIRP_SETTINGS_2D = {
    "by_slice": True,
    "glcm_spatial_method": ["2d_average", "2d_slice_merge",
                            "2.5d_direction_merge", "2.5d_volume_merge"],
    "glrlm_spatial_method": ["2d_average", "2d_slice_merge",
                             "2.5d_direction_merge", "2.5d_volume_merge"],
    "glszm_spatial_method": ["2d", "2.5d"],
    "gldzm_spatial_method": ["2d", "2.5d"],
    "ngtdm_spatial_method": ["2d", "2.5d"],
    "ngldm_spatial_method": ["2d", "2.5d"],
}

# Çıkarılan modlar. Anahtar = tablo/sayfa adı, değer = MIRP_SETTINGS'in üstüne
# yazılan ek ayarlar. Sıra önemlidir: ilk mod "ana" tablodur (features.csv ve
# all_features_combined.csv onun içindir; geriye uyumluluk).
MIRP_MODES: dict[str, dict[str, Any]] = {
    "3D": MIRP_SETTINGS_3D,
    "2D": MIRP_SETTINGS_2D,
}
PRIMARY_MODE = "3D"

# --- MIRP tek çağrı içi backend ---
# MIRP bir seriyi TEK "workflow" olarak işler ve ROI'ları (organları) sıra sıra
# hesaplar; yani MIRP'in kendi num_cpus'u ROI'ları DEĞİL, yalnızca farklı
# serileri paralelleştirir. Ölçüm: süre ~%99 ROI döngüsünde (organ başına ~2-3 sn),
# ortak yeniden örnekleme maliyeti ihmal edilebilir. Bu yüzden asıl hızlanma
# organları süreçlere bölerek elde edilir (aşağıya bakın).
# Her işçi süreç MIRP'i tek çekirdek/sıralı çalıştırır:
MIRP_NUM_CPUS = 1
MIRP_PARALLEL_BACKEND = "none"    # "none" | "joblib" | "ray"

# --- Süreç-düzeyi paralellik (öbek bazlı) + RAM-farkında işçi sayısı ---
# ÖNEMLİ: crop_around_roi öznitelikleri ~%0.3 değiştirdiği için IBSI uyumu adına
# KULLANILMAZ. MIRP hesabına hiç dokunmayız; belleği YALNIZCA eşzamanlı işçi
# sayısını kısarak yönetiriz (çıktı tek-çağrı referansıyla bit-birebir kalır).
#
# ÖLÇÜM (gerçek CT, resample ~145 MB): MIRP çağrısı başına tepe bellek organ
# boyutundan BAĞIMSIZ ~sabit ~5.8 GB (küçük rib de, büyük colon da ~5.8 GB).
# Çünkü maliyet organ değil, tam hacmin resample'ı + MIRP görüntü işleme. Bir
# öbekteki organlar bu resample'ı PAYLAŞIR ve sıra sıra işlenir → öbek tepe
# belleği ~sabit ~5.8 GB.
#
# ÖNEMLİ: Tepe bellek SLICE SAYISIYLA DEĞİL, FİZİKSEL KAPSAMLA ölçeklenir.
# 135-slice (3mm) ve 404-slice (1mm) aynı ~40 cm bölgeyi kaplar; 1mm'ye ikisi de
# AYNI ~145 MB'a örneklenir → ikisi de aynı işçi sayısını (~3) alır. İşçi sayısı
# _resampled_bytes() ile (size×spacing) header'dan hesaplanır; slice sayısına
# göre elle ayar YAPMAYIN. Eşzamanlı öbek(işçi) sayısı = RAM bütçesi / öbek belleği.
_cpu = os.cpu_count() or 4
PARALLEL_WORKERS = max(1, _cpu - 2)   # ÜST sınır süreç; asıl sınır RAM bütçesi

# --- SERİ-DÜZEYİ paralellik (kaç seri aynı anda işlensin) ---
# ÖLÇÜLDÜ: organ seçimi daraldığında (2 organ) seri içi paralellik
# `n_chunks = min(workers, len(roi_map))` yüzünden 2'ye kilitleniyor. 20
# çekirdekli bir makinede koşum sırasında CPU %9'da, 20 çekirdeğin 1'i meşgul
# ölçüldü — RAM 5-6 işçiye izin verse bile organ sayısı buna izin vermiyor.
# Boşta kalan çekirdekleri kullanmanın tek yolu BİRDEN ÇOK SERİYİ aynı anda
# işlemek.
#
# 1 → eski davranış (seriler sırayla).
#
# GÜVENLİK: TotalSegmentator adımı seri kalır (tek GPU; iki TS aynı anda
# VRAM'i taşırır) — bkz. batch_120kvp._TS_LOCK. Paralelleşen yalnız MIRP,
# NIfTI dönüşümü ve QC adımlarıdır.
#
# RAM: `available_memory_bytes()` bütçeyi bu sayıya BÖLER, yani eşzamanlı
# seriler birbirinin payını yiyemez. Organ sayısı zaten işçiyi 2'ye
# kilitlediği için bölme pratikte işçi sayısını düşürmüyor, yalnız
# aşırı-abonelik güvenliği sağlıyor.
SERIES_PARALLELISM = int(os.environ.get("PHANTOMX_SERIES_PARALLELISM", "3"))

# RAM bütçesi. psutil varsa: min(TOPLAM×MEM_TOTAL_FRACTION, BOŞ×MEM_AVAIL_FRACTION)
# — hem OS'e pay bırakır hem anlık doluluğa uyum sağlar. psutil yoksa MAX_MEMORY_GB.
MAX_MEMORY_GB = 24.0
MEM_TOTAL_FRACTION = 0.72
MEM_AVAIL_FRACTION = 0.90

# Öbek(işçi) tepe belleği ≈ WORKER_MEM_MULTIPLIER × (1 mm resample edilmiş CT boyutu).
# Kalibrasyon: R=145MB, izole ölçülen tepe ~5.8 GB. Ancak tepeler ANLIK ve kayan
# (gerçek çalışmada 4 işçide toplam ~10 GB gözlendi), yani pay var. 28 → seri başına
# ~5-6 işçi (daha hızlı). Nadir hizalı tepede OOM backoff otomatik toparlar.
# OOM tekrarlarsa 40'a geri alın (32 GB için güvenli değer).
#
# YENİDEN KALİBRE (seri-düzeyi paralellik için): 28 idi ve İZOLE ölçülen tepeye
# (5.8 GB) yakın bir pay tutuyordu. Aynı yorumun kendi ölçümü, gerçek koşumda
# 4 işçinin toplam ~10 GB kullandığını söylüyor — yani işçi başına ~2.5 GB,
# izole tepenin yarısından az; tepeler anlık ve kayan olduğu için hiç
# hizalanmıyorlar.
#
# 28 ile işçi başına bütçe 4.06 GB düşüyordu ve SERIES_PARALLELISM ile
# bölündüğünde seri başına yalnız 1 işçi kalıyordu (ölçüldü: 4.6/4.06). O
# durumda paralellik hiçbir şey kazandırmıyor: 3 seri × 1 işçi, 1 seri × 2
# işçiyle aynı organ akışı demek.
#
# 18 → işçi başına ~2.6 GB, yani gözlenen gerçek kullanıma denk. 3 paralel
# seri × 2 işçi = 6 eşzamanlı MIRP süreci ≈ 16 GB bütçe. OOM olursa işçi
# sayısı yarıya inip yeniden deneniyor (geri çekilme zinciri bozulmadı), o
# yüzden yanılma payı sessiz bir kayba değil, yavaşlamaya dönüşür.
# OOM tekrarlarsa 28'e, sonra 40'a geri alın.
WORKER_MEM_MULTIPLIER = 18.0

# OOM (işçi çökmesi VEYA MemoryError) olursa: işçi sayısı yarıya iner, kalan
# yeniden denenir; 1 işçide de çökerse organlar tek tek izole edilir, yalnız
# sorunlu organ atlanır ve RAPORLANIR (asla sessizce kaybolmaz).

# MIRP takılma koruması (watchdog): bir öbek, organ_sayısı × bu saniyeyi aşarsa
# askıda sayılıp iptal edilir; kalan organlar izolasyonla yeniden denenir.
# Normal organ ~50 sn; bu değer çok cömert (yalnızca gerçek kilitlenmeyi yakalar).
MIRP_WATCHDOG_PER_ORGAN_S = 300

# İZOLASYON zaman aşımı tabanı (saniye). Havuz koşusu 1800 sn taban kullanır (tüm
# öbekler paylaşımlı çalışır); ama tek organ izole edilirken 1800 çok cömertti
# (30 dk/organ → 6 organlık öbek 3 saat). Normal organ ~50 sn, en ağırı birkaç dk;
# 600 sn meşru organa fazlasıyla yeter, patolojik organı 5x daha erken atlar.
# Atlanan organ features.done.json + INCOMPLETE_report.txt'de RAPORLANIR (kayıp yok).
MIRP_ISOLATED_FLOOR_S = 600.0


def set_output_root(root: Path | str) -> Path:
    """Tüm çıktı yollarını `root` altına bağlar ve kökü döndürür.

    Modüller yolları `config.NIFTI` gibi ÇAĞRI ANINDA okuduğu için buradaki
    yeniden bağlama hepsine geçer. Böylece bir koşumda birden çok kaynak
    sırayla işlenip her biri kendi çıktısını kendi klasörüne yazabiliyor.
    """
    global DATA, DICOMS, NIFTI, SEG, FEATURES, OVERLAYS, COMBINED_FEATURES_CSV
    DATA = Path(root)
    DICOMS = DATA / "dicoms"
    NIFTI = DATA / "nifti"
    SEG = DATA / "segmentation"
    FEATURES = DATA / "features"
    OVERLAYS = DATA / "overlay"
    COMBINED_FEATURES_CSV = FEATURES / "all_features_combined.csv"
    return DATA


def ensure_dirs() -> None:
    """Çıktı dizinlerini oluşturur (idempotent)."""
    for d in (DICOMS, NIFTI, SEG, FEATURES, OVERLAYS):
        d.mkdir(parents=True, exist_ok=True)
