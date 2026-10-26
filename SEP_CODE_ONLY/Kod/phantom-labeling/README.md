# CT Phantom Annotation Pipeline

Otomatik CT fantom analizi — DICOM taramalarından ROI (ilgi bölgesi) çıkarma, sınıflandırma ve doğrulama sistemi. **Philips Spectral CT** (klasik), **Siemens NAEOTOM Alpha / Photon-Counting CT** ve **Siemens SOMATOM Force** (çift-tüplü, enerji-integre eden dedektör) taramalarında çalışır.

## Genel Bakış

Bir kalibrasyon fantomunu içeren CT taramalarından otomatik olarak **texture, star ve insert ROI'lerini** tespit edip, sınıflandırıp, doğrulayıp **DICOM (`.dcm`) formatında** dışa aktarır. Her ROI, HU değerlerini koruyan minimal bir CT-Image DICOM dosyası olarak yazılır; herhangi bir DICOM görüntüleyici ile açılabilir.

Kullanım alanları:
- Spektral / Photon-counting CT kalibrasyon doğrulama
- Kantitatif görüntü kalite kontrolü (IQ QC)
- Doz ve Hounsfield Unit (HU) tutarlılık analizi
- Rekonstrüksiyon algoritması / kernel / Q-level karşılaştırması

## Dizin Yapısı

```
Annotations/
├── README.md                       # Bu dosya
└── codes/                          # Kaynak kod snapshot'ı
    ├── .gitignore
    ├── requirements.txt            # Python bağımlılıkları
    ├── config.py                   # Fantom geometrisi + algoritma sabitleri
    │
    ├── label_series.py             # Tek-komut giriş: etiketle + annotate (ana CLI)
    ├── main.py                     # Düşük seviye pipeline CLI (sadece annotate)
    ├── organize_siemens_pcct.py    # Düz Siemens dump'ını serilere ayıran yardımcı
    ├── batch_run_siemens.py        # Organize edilmiş seriler üstünde batch runner
    │
    ├── labeling/                   # DICOM otomatik etiketleme modülü
    │   ├── __init__.py
    │   ├── header_extractor.py     # DICOM header metadata okuyucu (+PCCT VMI kVp, Q-level)
    │   ├── name_builder.py         # Windows-safe isim üretici (+PCCT/Force isim şemaları)
    │   ├── phantom_instances.py    # Çoklu-fantom tek tarama bölücü (SOMATOM Force)
    │   ├── labeler.py              # Batch etiketleme orkestratörü (+flat dump / DICOMDIR staging, take-index)
    │   └── annotator.py            # Etiketlenen seri üzerinde pipeline sürücüsü
    │
    ├── core/                       # Ana pipeline fazları
    │   ├── dicom_loader.py         # Phase 0: DICOM yükle, HU'ya çevir
    │   ├── phantom_detector.py     # Phase 1: İnce-kabuk kontur tespiti (+Hough ve özellik-kümesi fallback)
    │   ├── section_classifier.py   # Phase 2: Bölüm sınıflandırma
    │   ├── orientation_corrector.py# Phase 2b: Ters (inserts-first) fantomu kanonik forma çevir
    │   ├── section_trim.py         # Phase 2c: Bölüm başına iç fiziksel-z penceresi trim'i
    │   ├── rotation_detector.py    # Phase 3: Global rotasyon
    │   ├── roi_extractor.py        # Phase 4: ROI çıkarma (+adaptif PVE guard)
    │   └── validator.py            # Phase 5: Kalite doğrulama
    │
    ├── analysis/                   # Yardımcı analiz modülleri
    │   ├── fft_utils.py            # FFT + polar dönüşüm
    │   ├── insert_detector.py      # Insert tespit ve halka oturtma
    │   ├── star_classifier.py      # STAR8 vs STAR16 ayrımı (açısal-FFT simetri gücü)
    │   └── texture_validator.py    # Radon + sinüs fit doğrulama
    │
    ├── output/                     # Sonuç dışa aktarım
    │   ├── exporter.py             # DICOM (.dcm) ROI + metadata.json yazıcı
    │   ├── visualizer.py           # Overlay PNG üretici
    │   ├── insert_painter.py       # Boyalı insert-ROI PNG'leri (piksel-piksel maske doğrulama)
    │   └── coronal_overlay.py      # Koronal görünümde section z-pencereleri PNG'si
    │
    ├── diagnostic/                 # Bağımsız tanı / doğrulama scriptleri
    │   ├── duplicate_excel_report.py  # Duplicate-slice denetimi (label_series sonrası otomatik)
    │   └── ...                     # probe / debug / ölçüm araçları (pipeline'a dahil değil)
    │
    └── utils/                      # Genel yardımcılar
        ├── file_utils.py           # DICOM dosya tarama (uzantılı + uzantısız)
        └── logger.py               # Logging yapılandırması
```

## Pipeline Fazları

`main.py` ve `label_series.py` her DICOM serisini sıra ile şu fazlardan geçirir:

| Faz | Modül | Görev |
|-----|-------|-------|
| **Phase 0** | `core/dicom_loader.py` | DICOM serisini 3D hacme yükle, `RescaleSlope/Intercept` ile HU birimine çevir |
| **Phase 1** | `core/phantom_detector.py` | Fantom sınırını **dilim başına** bul: birincil ince-kabuk kontur tespiti; başarısızsa Hough dairesi, sonra iç-özellik kümesi fallback'i. Ham tespitler Theil-Sen ile yumuşatılmış (cx, cy, r) iz'e dönüştürülür |
| **Phase 2** | `core/section_classifier.py` | Her dilimi STAR16, STAR8, INSERT1 veya INSERT2 olarak etiketle |
| **Phase 2b** | `core/orientation_corrector.py` | Ters yerleştirilmiş (inserts-first) fantomu algıla; hacmi Z+X ekseninde çevirerek kanonik "stars-first" forma getir |
| **Phase 2c** | `core/section_trim.py` | Her bölümü sinyal-ağırlıklı z-merkezine ortalanmış iç fiziksel pencereye (varsayılan ±15 mm) kırp — protokolden bağımsız karşılaştırılabilirlik için |
| **Phase 3** | `core/rotation_detector.py` | Texture landmark'larından global rotasyon açısını hesapla |
| **Phase 4** | `core/roi_extractor.py` | Texture (8 tip), insert (40 × 3 alt-ROI) ve star ROI'lerini çıkar; STAR bölümlerinde adaptif PVE guard uygula |
| **Phase 5** | `core/validator.py` | FFT, Radon ve 3D tutarlılık kontrolleri ile ROI'leri doğrula |
| **Phase 6** | `output/` modülleri | DICOM (`.dcm`) dışa aktarım + overlay PNG + boyalı insert PNG'leri + koronal genel bakış PNG'si |

## Fantom Geometrisi

`config.py` içinde tanımlanmış sabitler:

| Parametre | Değer |
|-----------|-------|
| Fantom çapı | 19.0 cm |
| Dış insert halkası mesafesi | 7.0 cm (merkezden) |
| İç insert halkası mesafesi | 4.5 cm (merkezden) |
| Texture ROI kare kenarı | 3.9 cm |
| Star çapı | 4.0 cm |
| Insert ROI çapı | 1.5 cm (görsel parlak çekirdek) |
| Insert alt-ROI çapları | **5 / 10 mm** (her insert için 2 eşmerkezli ROI; varsayılan 15.09.2026'da değişti — 15 mm çıkarımda hiç kullanılmıyordu). Eski üçlü düzen için `--insert-sizes 5,10,15` |
| Bölüm sayısı | 4 (STAR16, STAR8, INSERT1, INSERT2) |
| Insert sayısı | 40 (bölüm başına 20 = 10 dış + 10 iç) |
| Texture sayısı | 8 (bölüm başına 4) |

### Texture İsimleri

- **STAR16 bölümünde**: Diagonal, Vertical, Square, Horizontal
- **STAR8 bölümünde**: Honeylarge, Honeymedium, Sinusoidal, Honeysmall

## Kurulum

```bash
cd codes
pip install -r requirements.txt
```

**Bağımlılıklar** (kodun gerçekten kullandıkları):
- `numpy>=1.24.0` — Dizi hesaplama
- `scipy>=1.10.0` — FFT, optimizasyon, Theil-Sen regresyonu, sinyal işleme
- `scikit-image>=0.20.0` — Görüntü işleme, Radon transform
- `opencv-python>=4.8.0` — Kontur/Hough tespiti, morfolojik operasyonlar, PNG çıktıları
- `pydicom>=2.4.0` — DICOM okuma **ve** ROI `.dcm` yazma
- `openpyxl` — `duplicate_audit.xlsx` denetim raporu için (requirements.txt'te yok; kurulu değilse denetim adımı uyarı verip atlanır, etiketleme etkilenmez)

> **Not:** `requirements.txt` içinde hâlâ listelenen `matplotlib` ve `tqdm` kod tarafından **kullanılmıyor** (tüm görselleştirme OpenCV ile yapılır, ilerleme raporları log satırlarıdır). Kurulmaları zorunlu değildir.

Python 3.12+ gerekir.

## Otomatik DICOM Etiketleme + Annotation (Ana Kullanım)

`label_series.py` tek komutla iki iş yapar:

1. Girdi klasöründeki tüm CT-recon serilerini DICOM header'dan üretilen **standart bir isimle** ayrı klasöre kopyalar.
2. Her seri üzerinde **annotation pipeline'ını** (Phase 0-6) çalıştırarak `overlays/`, `painted/`, `coronal/`, `Textures/`, `Stars/`, `Inserts/` ve `metadata.json` üretir.

### Hızlı Kullanım

```bash
cd codes

# Tipik kullanım — her seriyi etiketler + pipeline'ı çalıştırır
python label_series.py --input "C:/path/to/PCCT_folder"

# Sadece etiketle, annotation'ı atla (hızlı)
python label_series.py --input "<dicom_root>" --no-annotate

# Dosya kopyalamadan sadece rapor üret (güvenli ön kontrol)
python label_series.py --input "<dicom_root>" --dry-run

# Sadece ilk N seriyi işle (smoke test için)
python label_series.py --input "<dicom_root>" --max-series 3

# Anonim InstitutionName'i (ör. "Ser.Nr.246663") gerçek merkez adıyla değiştir
python label_series.py --input "<force_export>" --institution AcibademKent

# Özel çıktı dizini + detaylı log
python label_series.py -i "<dicom_root>" -o "D:/labeled" --verbose --log-file run.log
```

Varsayılan çıktı: `Desktop/<giriş_klasör_adı>_LABELED/`. Orijinal girdi klasörüne asla yazılmaz — kod her zaman yeni bir çıktı dizinine kopya üretir. Aynı seri yeniden annotate edilirse eski `overlays/`, `Textures/`, `Stars/`, `Inserts/`, `coronal/`, `painted/` ve `metadata.json` çıktıları önce silinir (bayat ROI dosyası kalmaz); `dicoms/` hiç dokunulmaz.

### CLI Seçenekleri

| Seçenek | Açıklama |
|---|---|
| `--input, -i` | Girdi DICOM kök dizini (zorunlu) |
| `--output-dir, -o` | Çıktı dizini (varsayılan: `Desktop/<name>_LABELED`) |
| `--dry-run` | Dosya kopyalamadan sadece rapor üret (annotation da atlanır) |
| `--no-annotate` | Annotation pipeline'ını atla, sadece etiketle ve `dicoms/` altına kopyala |
| `--max-series N` | En çok N adet CT-recon serisini işle (büyük dataset'lerde smoke test için) |
| `--institution AD` | DICOM `InstitutionName` tag'ini isimlendirmede bu değerle değiştir — tag anonimleştirilmişse (Siemens servis export'larındaki `Ser.Nr.NNNN`) gerekli |
| `--verbose, -v` | Debug log seviyesine geç |
| `--log-file PATH` | Stdout'a ek olarak log dosyasına yaz |

### Desteklenen Girdi Klasör Yapıları

Araç dört farklı layout'u **otomatik algılar**:

**1. İki seviyeli (Philips test case export'u):**
```
CALIBRAMICS_TEST/
├── TestCaseA/
│   ├── CT_Seq._201_IMR_BR_1/
│   └── SR_Seq._5000_Dose/      ← atlanır (SR rapor)
└── TestCaseB/ ...
```

**2. Tek-seviye seri:**
```
MyScans/
├── CT_Seq._201_IMR_BR_1/
└── CT_Seq._202_iDose_BR_2/
```

**3. Düz DICOM dump (klasik Siemens PCCT):**
```
27032026-PCCT/
├── Calibramics.CT.PhantomX_Calibr.10.1.2026....dcm
├── Calibramics.CT.PhantomX_Calibr.10.2.2026....dcm
└── ... (tek klasörde binlerce .dcm dosyası)
```

**4. DICOMDIR iç içe yapı (Siemens PCCT / Force DICOMDIR export):**
```
10042026-PCCT/
├── DICOMDIR
└── DICOM/
    └── <studyID>/
        ├── <seriesID_1>/    (uzantısız binary DICOM dosyaları)
        ├── <seriesID_2>/
        └── ...
```

Düz ve DICOMDIR yapılarında dosyalar `SeriesInstanceUID`'ye göre otomatik gruplanır ve geçici bir `_staging/` dizinine hardlink edilir. Uzantısız DICOM dosyaları "DICM" magic byte kontrolüyle tanınır. Aynı `SeriesNumber` + `SeriesDescription`'ı paylaşan ama farklı UID'li test-retest çiftleri ayrı seriler olarak korunur.

### İsim Formatı

Her seri için DICOM header'dan okunan alanlardan tek bir isim üretilir. Taban yapı:

```
{Merkez}_{Marka}_{Cihaz}_{Tarih}_{ReconKernel}_{ConvKernel}_{KVP}_{mAs}_{FOV}_{SliceThickness}
```

Cihaz ailesine göre son isim şu ekleri alır:

**Philips (ve diğer PCCT/Force olmayan) seriler** — tabanın sonuna rekonstrüksiyon aralığı token'ı eklenir:
```
Memorial_Hospital_Philips_Spectral_CT_7500_20260409_IMR_BR_120_225_250_0p67_recon0p67mm
```

**Siemens NAEOTOM Alpha (PCCT)** — `<taban>_<ReconType>_take<N>_recon<X>mm`:
```
USZ_Siemens_Healthineers_NAEOTOM_Alpha_20260410_Q3_Qr40f_120_100_300_1_MonoE70keV_take1_recon1mm
```
- `build_pcct_base` taban ismin başındaki `Radiologie_` önekini kırpar (USZ export'larında `InstitutionName = "Radiologie USZ"` olduğundan isim `USZ_...` ile başlar).
- **ReconType**, `ImageType` + `ImageComments`'tan sınıflandırılır: `MonoE<N>keV` (VMI recon, keV değeri `ME NNkeV` deseninden), `PolyE` (THRESHOLD/polikromatik), `VMI_unknown` veya `OTHER_<tip>`.
- **take<N>**: aynı `(taban, ReconType)` grubundaki tekrar çekimler `AcquisitionDateTime` + `SeriesNumber` sırasına göre `take1..takeN` numarası alır (test-retest çalışmaları için).
- **recon<X>mm**: dilimler arası rekonstrüksiyon aralığı — `SpacingBetweenSlices` tag'i, yoksa ~20 dosyanın `SliceLocation` farklarının medyanından kestirilir (`recon1mm`, `recon0p5mm`, ...).

**Siemens SOMATOM Force (Dual-Source EID)** — PCCT ile aynı ek yapısı, ReconType her zaman `PolyE` (bu cihazda spektral/VMI recon yoktur):
```
AcibademKent_SIEMENS_SOMATOM_Force_20251023_Br40d_120_285_350_3_PolyE_take1_recon2mm
```
- Ayrıştırıcı alanlar `ConvolutionKernel` (Br40d/Br59d), gerçek tüp `KVP`'si (80/100/120 — PCCT'nin aksine fizikseldir), `Exposure`, FOV ve dilim kalınlığıdır. ADMIRE gücü yalnızca özel (CSA) tag'lerde olduğundan isimde Q-level eşleniği yoktur.
- `InstitutionName` anonimse (`Ser.Nr.NNNN`) ve `--institution` verilmemişse merkez token'ı isimden tamamen düşülür.
- Tekrarlanan aynı-protokol çekimler (aynı SeriesDescription, farklı `IrradiationEventUID`/`AcquisitionDateTime`) PCCT'deki gibi `take1/take2` olur.

**Çoklu-fantom bölme (SOMATOM Force):** Bazı merkezler tek uzun spiral çekimde birden fazla fiziksel CalibraMics fantomunu art arda tarar (Acibadem Kent'te 3 fantom + boş masa gözlendi). `labeling/phantom_instances.py` örneklenmiş dilimlerde "fantom boyutunda içi boş daire var mı" testiyle (Hough) fantom örneklerinin z-aralıklarını bulur; hava boşluğu ve masa dilimleri hiçbir örneğe ait olmadığından kopyalanmaz. N > 1 örnek bulunursa seri `<isim>_P1 .. _PN` şeklinde N ayrı klasöre bölünür ve her biri bağımsız annotate edilir — tek-fantom varsayan section classifier her hacimde tam bir STAR16/STAR8/INSERT1/INSERT2 seti görür.

İki seri yine de aynı ismi üretirse ikincisi `_01`, üçüncüsü `_02` son ekini alır.

### DICOM Alan Eşleşmesi

| Kullanıcı Adı | DICOM Tag | Yedek (fallback) |
|---|---|---|
| Merkez | `InstitutionName` | `--institution` CLI override |
| Marka | `Manufacturer` | — |
| Cihaz | `ManufacturerModelName` | `StationName` |
| Tarih | `StudyDate` | `AcquisitionDate` → `ContentDate` |
| ReconKernel | `ReconstructionAlgorithm` | SeriesDescription içinden `IMR`/`iDose`/`FBP` veya **PCCT için `Q1..Q4`** |
| ConvKernel | `ConvolutionKernel` | — |
| KVP | `KVP` | **PCCT için SeriesDescription'dan VMI kVp** (fiziksel KVP sabit olduğu için) |
| mAs | `Exposure` | `XRayTubeCurrent × ExposureTime / 1000` |
| FOV | `ReconstructionDiameter` (mm) | `DataCollectionDiameter` |
| SliceThickness | `SliceThickness` (mm) | — |
| ReconSpacing | `SpacingBetweenSlices` (mm) | `SliceLocation` farklarının medyanı (~20 header'dan kestirim) |

Eksik alanlar `UNKNOWN` ile doldurulur. Her seri için eksik alanların listesi `labels.json` çıktısında tutulur.

**Siemens NAEOTOM Alpha özelliği:** Photon-counting CT'lerde `ReconstructionAlgorithm` tag'i boştur ve `KVP` fiziksel tüp voltajını (sabit 120/140) tutar. İsimlendirme SeriesDescription'daki `"120kVp_100mAs_1mm_300FOV 1,00 Qr40 Q3"` örüntüsünü parse ederek VMI kVp'yi ve Q-level'ı çıkarır — böylece farklı Q-level'lı rekonstrüksiyonlar ayrı klasöre gider.

### Dosya Adı Güvenliği

- Ondalık noktalar `p` ile değiştirilir (`0.67` → `0p67`) — Windows uzantı algılamasıyla karışmaması için
- Eksi işareti `neg` olur (`-5` → `neg5`)
- Windows yasaklı karakterler (`<>:"/\\|?*`) ve kontrol karakterleri temizlenir
- Boşluklar tek bir `_` olur
- Rezerve cihaz isimleri (`NUL`, `CON`, `COM1`…) başına `_` eklenerek güvenli hale getirilir
- İki seri aynı ismi üretirse ikincisi `_01`, üçüncüsü `_02` şeklinde son eklenir

### Filtreleme

Sadece **CT rekonstrüksiyon serileri** etiketlenir. Otomatik olarak atlanır:

- `SR_*` yapısal raporlar (doz bilgisi vb.)
- `Surview` / `Scout` / `Localizer` / `Topogram` tarama resimleri
- SOP Class UID'si CT Image Storage (`1.2.840.10008.5.1.4.1.1.2`) olmayan her şey

### Çıktı Yapısı

Her etiketlenen seri kendi klasörüne aşağıdaki gibi yerleştirilir:

```
Desktop/<girdi>_LABELED/
├── labels.csv                                     # Excel uyumlu mapping + annotation özeti
├── labels.json                                    # Tüm metadata + eksik alan uyarıları
├── duplicate_audit.xlsx                           # Duplicate-slice denetim raporu (openpyxl gerektirir)
├── _staging/                                      # (düz/DICOMDIR dump'larda) geçici SeriesUID gruplaması
└── <StandartIsim>/
    ├── dicoms/                    # Orijinal DICOM dosyaları (kopya/hardlink)
    ├── overlays/                  # Phase 6: slice görselleştirmeleri (PNG, 512×512)
    │   ├── slice_0000.png
    │   └── ...
    ├── painted/                   # Phase 6: insert ROI'leri yarı saydam boyalı PNG'ler
    │   └── slice_NNNN.png         #   (boyanan alan = diske yazılan ROI maskesiyle birebir aynı)
    ├── coronal/
    │   └── coronal.png            # Phase 6: koronal görünümde section z-pencereleri (trim doğrulama)
    ├── Textures/                  # Phase 6: tekstür ROI'leri (.dcm, nearest-neighbor)
    │   ├── Diagonal/ Vertical/ Square/ Horizontal/      ← STAR16 bölümünde
    │   └── Honeylarge/ Honeymedium/ Sinusoidal/ Honeysmall/  ← STAR8 bölümünde
    │       └── slice_XXXX_<TextureName>.dcm
    ├── Stars/                     # Phase 6: yıldız ROI'leri
    │   ├── Star16/  slice_XXXX_Star16.dcm
    │   └── Star8/   slice_XXXX_Star8.dcm
    ├── Inserts/                   # Phase 6: 40 silindir insert × 3 alt-ROI — SIZE-FIRST düzen
    │   ├── 5mm/   Insert_01/ ... Insert_40/   ← slice_XXXX_Insert_NN_5mm.dcm
    │   ├── 10mm/  Insert_01/ ... Insert_40/
    │   └── (15mm/ yalnız --insert-sizes 5,10,15 verilirse)
    └── metadata.json              # ROI metadata + doğrulama raporu (errors, warnings)
```

> **Not:** Kullanıcı terminolojisindeki "cylinders" = `Inserts/` klasörüdür. Pipeline 40 adet silindir insert tespit eder; her insert için 5/10 mm çaplı 2 eşmerkezli alt-ROI (`config.INSERT_SUBROI_DIAMETERS_MM`; `--insert-sizes` ile değiştirilebilir), **boyut-öncelikli** (size-first) düzende `.dcm` olarak saklanır — böylece belirli bir çaptaki 40 insert'in tamamı 40 ayrı klasöre girmeden birlikte incelenebilir.

**ROI `.dcm` dosya formatı:** her ROI, `RescaleSlope=1` / `RescaleIntercept=-1024` ile HU değerlerini int16 olarak saklayan minimal bir CT-Image DICOM'udur. Dairesel maske dışındaki (NaN) pikseller `PixelPaddingValue=-32768` sentinel değeriyle işaretlenir; `ImageComments` alanı ROI kimliğini (label, slice, alt-piksel merkez, açı, accepted) taşır. Her ROI dosyası kendi `SeriesInstanceUID`'sini alır (seriye göre gruplayan görüntüleyicilerde her insert/texture/star ayrı seri olarak görünür); `SeriesDescription` `<seri>|<bölüm>|<label>|slice<N>` formatındadır.

**`labels.csv` sütunları**: `original_path, new_name, new_path, institution, manufacturer, device_model, study_date, recon_kernel, conv_kernel, kvp, mas, fov, slice_thickness, series_description, skipped, skip_reason, annotated, accepted_rois, rejected_rois, annotation_error`

## Algoritma Notları

### ROI Çıkarım Politikası (nearest-neighbor, zero interpolation)

ROI çıkarımında **nearest-neighbor** sampling kullanılır — orijinal HU değerleri hiçbir filtreleme / yumuşatma / interpolasyon olmadan korunur ve `.dcm` dosyasına aynen yazılır. Bu, kantitatif CT analizlerinin güvenilirliği için kritiktir.

### PVE (Partial Volume Effect) Güvenlik Katmanları

Her section'ın z-sınırlarında fiziksel insert/star patternleri komşu modülden "sızar" ve HU değerleri yanıltıcı olur. Pipeline bunu üç katmanda filtreler:

1. **Classifier PVE trim** ([core/section_classifier.py](codes/core/section_classifier.py)) — boundary slice'ları `TRANSITION` olarak etiketler (ROI çıkarılmaz)
2. **İç pencere trim'i** ([core/section_trim.py](codes/core/section_trim.py)) — her bölümü sinyal-ağırlıklı z-merkezine ortalanmış **sabit fiziksel pencereye** kırpar (`SECTION_INNER_WINDOW_HALF_MM = 15` → ±15 mm: 1 mm taramada 30, 3 mm taramada 10 dilim). Bu, tutulan hacmi dilim kalınlığından bağımsız yapar; sabit-mm kapalıysa `SECTION_INNER_WINDOW_FRACTION = 0.7` oransal fallback'i devreye girer
3. **Adaptif PVE guard** ([core/roi_extractor.py](codes/core/roi_extractor.py)) — STAR bölümlerinde eski sabit `pve_lead`/`pve_tail` trim'inin yerini alan sinyal-tabanlı koruma. Her aday dilimin star diski üzerindeki gradyan sinyali bölüm medyanıyla karşılaştırılır; bir kenar dilimi ancak (a) sinyali gerçekten çökmüşse (< 0.75 × medyan — modül-kenarı geçiş malzemesi) veya (b) böyle bir çöküşe `STAR_GUARD_CLIFF_BUFFER_MM` (2 mm) fiziksel tampondan daha yakınsa soyulur. Soyma bütçesi de fizikseldir (`STAR_ADAPTIVE_GUARD_MAX_MM` = 3 mm, dilim kalınlığına göre dilim sayısına çevrilir) ve pencere **kaydırılır**: bir uçtan soyulan her dilim için karşı uçtan güvenli bir aday geri kazanılmaya çalışılır, böylece hedef dilim sayısı korunur

### Section Classification

1. **Blob sayımı** → STAR (az blob) vs INSERT (çok blob) ayrımı
2. **STAR8 vs STAR16** ([analysis/star_classifier.py](codes/analysis/star_classifier.py)) → birincil ayırıcı, açısal profil FFT'sinde **8. ve 16. simetri derecelerindeki güç oranıdır** (p16 ≫ p8 ⇒ STAR16, p8 ≥ p16 ⇒ STAR8). Keskin PCCT kernellerinde (Qr40) düz peak sayımı her iki bölümde de ~27-32 pik verdiği için ayırt edici değildir; eski peak-count aralıkları yalnızca spektrum belirsiz kaldığında fallback olarak kullanılır
3. **Gap filling** → Eksik bölüm dilimlerini pozisyonel çıkarımla tamamla
4. **Yönelim tespiti** → `stars-first` (kanonik) vs `inserts-first` (ters yerleştirilmiş fantom) otomatik algılanır; inserts-first durumda Phase 2b ([core/orientation_corrector.py](codes/core/orientation_corrector.py)) hacmi Z ve X ekseninde çevirerek kanonik forma getirir — sonraki tüm fazlar ve overlay'ler kanonik yönelimde çalışır (diskteki DICOM'lara dokunulmaz)

### Rotation Detection

- 4 texture blob'un konumundan (0°, 90°, 180°, 270° beklenen) sapmalar ölçülür
- **Circular mean** ile agregasyon — açısal wraparound (±180°) sorunu çözülür
- Her dilim için güven skoru hesaplanır

### Validation Katmanları

1. **Stage 1**: FFT verify (star sınıflandırma teyidi)
2. **Stage 2**: Texture patterns (Radon + sinüs fit, R² > 0.7 eşiği)
3. **Stage 3**: 3D tutarlılık (dilimler arası merkez drift ≤ 10 piksel)

## Alternatif Giriş Noktaları

### `main.py` — Sadece Annotate

Etiketleme yapmadan bir veri kökü altındaki serileri annotate etmek için (label_series ile aynı Phase 0-6 zincirini çalıştırır):

```bash
# Tüm serileri işle (config.py'de tanımlı kök üzerinden)
python main.py

# Belirli bir seriyi işle
python main.py --series "<seri_klasör_adı>"

# Belirli bir tarihteki serileri filtrele
python main.py --date 20260114

# İşlenecek maksimum seri sayısını sınırla
python main.py --max-series 5

# Özel veri ve çıktı dizinleri
python main.py --data-dir "D:/CT_Data" --output-dir "D:/Results"
```

Ortam değişkenleri: `CALIBRAMICS_DATA_ROOT`, `CALIBRAMICS_OUTPUT_ROOT`.

### `organize_siemens_pcct.py` — Siemens Dump'ını Organize Et

Düz `.dcm` dump'ını `SeriesInstanceUID` bazında klasörlere ayırır ve her klasörü açıklayıcı bir isimle adlandırır (annotation çalıştırmaz):

```bash
python organize_siemens_pcct.py --input "C:/path/to/27032026-PCCT" --copy
```

### `batch_run_siemens.py` — Organize Edilmiş Serilerde Batch

`organize_siemens_pcct.py` çıktısı üzerinde her recon serisine annotation pipeline'ı koşturur, seri başına logları `_logs/` altına ve özeti `_batch_summary.csv`'ye yazar:

```bash
python batch_run_siemens.py --data-dir "C:/path/to/organized" --output-dir "D:/results"
```

Çoğu kullanıcı için `label_series.py` yeterlidir — organize+annotate iki adımı tek komuta birleştirir.

### `diagnostic/` — Tanı Scriptleri

Pipeline'a dahil olmayan bağımsız probe/debug/ölçüm araçları (merkezleme ölçümü, sınır sinyal probu, sentetik adversarial testler vb.). İçlerinden yalnızca `duplicate_excel_report.py` otomatik çalışır: her başarılı `label_series.py` koşusunun sonunda girdi serilerindeki yinelenen dilimleri denetleyip `duplicate_audit.xlsx` üretir.

## Teknoloji Stack

- **Python 3.12+**
- **numpy** — Temel dizi hesaplama
- **scipy** — FFT, `optimize.curve_fit`, Theil-Sen regresyonu, signal processing
- **scikit-image** — Morfoloji, etiketleme, Radon transform
- **opencv-python** — Kontur/`HoughCircles` tespiti, connected components, PNG çıktıları
- **pydicom** — DICOM okuma + ROI `.dcm` yazma
- **openpyxl** — Duplicate-slice denetim Excel'i (opsiyonel)

## Notlar

- Bu klasör (`Annotations/codes/`) **kaynak kod snapshot**'ıdır. Pipeline çıktıları burada saklanmaz — varsayılan olarak `Desktop/<girdi>_LABELED/` altına yazılır.
- `__pycache__`, log dosyaları ve IDE konfigürasyonları `.gitignore` ile hariç tutulur.
- `config.py` varsayılan path'leri kullanıcının `Desktop`'ını işaret eder (env var ile override edilebilir). Doğrudan yolu değiştirmeniz gerekmez; `label_series.py --input/--output-dir` yeterlidir.

## Lisans

Bu proje dahili kullanım içindir.
