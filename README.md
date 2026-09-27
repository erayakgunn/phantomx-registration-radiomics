# PCCT Veri Paketi — bitirme projeleri (baseline protokol)

Hazırlanma tarihi: 18 Eylül 2026. Kaynak: `E:\PCCT` (USZ Zürich, Siemens NAEOTOM Alpha, seri no 127032, syngo CT VB20A). Bu klasör kaynağın **kopyasıdır**; burada yapacağınız hiçbir değişiklik asıl veriyi etkilemez.

> **İlk iş:** paketi kısa bir köke kopyalayın (ör. `D:\PCCT_Veri_Paketi`). Seri klasör adları uzundur; derin bir klasörün altına koyarsanız Windows'un 260 karakterlik yol sınırına takılırsınız (buradaki en uzun yol ≈ 230 karakter).

## 1. Pakette ne var?

İki fantom **aynı helikal çekimlerde** birlikte tarandı: **CalibraMics** (3D-baskı geometrik fantom: 40 yoğunluk inserti, 8 texture hedefi, 2 yıldız) ve **PhantomX** (antropomorfik abdomen; aorta ve dalak). Yani her DICOM serisinde iki fantom da görünür.

Pakete **yalnız tek protokol** alındı:

| QIR | Kernel | Tüp voltajı | Tüp akımı-zamanı | Rekon FOV | Kesit | Rekon tipi | Pitch |
|---|---|---|---|---|---|---|---|
| Q3 | Qr40f | 120 kV | 100 mAs | 300 mm | 1.0 mm | PolyE (T3D) | 1.0 |

- **92 seri / 23 oturum** (11 Şub 2026 – 5 Ağu 2026). 11 Şubat 2026'dan önce tarayıcı T3D/PolyE üretmediği için daha eski oturum yoktur.
- Oturum başına seri: 17 oturumda 3 (aynı oturumda 3 bağımsız ışınlama), 20260526 / 20260622 / 20260730'da 6, **20260731 ve 20260805'te 10** (adanmış tekrarlanabilirlik oturumları).
- PhantomX tarafı **89 seri / 22 oturum**: 20260219 oturumunda PhantomX işlenmemiştir (3 seri yalnız CalibraMics).
- 20260312 oturumunun pitch 0.95 ile çekilen 3 serisi protokol dışı olduğu için **alınmadı** (`MANIFEST.csv` içinde `selected=0` olarak listelenir).
- Diğer kernel'ler (Qr36f/Qr64f), mAs, FOV, kalınlık, QIR ve VMI (45/70 keV) rekonları bu pakette **yoktur**.

Boyut: ≈ 34.6 GB, ≈ 318 bin dosya.

## 2. Klasör düzeni

```
PCCT_Veri_Paketi\
├── README.md, MANIFEST.csv / .xlsx, CHECKSUMS.md5
├── DICOM\<oturum>\<seri>\                 ham CT (TEK KOPYA; iki fantom için ortak). 353–387 dilim/seri, 512×512
├── CalibraMics\
│   ├── LABELED\<oturum>_USZ_PCCT\<seri>\  Inserts\{5mm,10mm}\Insert_01..40, Stars\{Star16,Star8}, Textures\<8 ad>,
│   │                                      overlays\, painted\, coronal\, metadata.json   (+ oturumda labels_baseline.csv/.json)
│   └── RADIOMICS\                         <oturum>_PCCT_radiomics_baseline.xlsx + TUM_OTURUMLAR_3D / _2D (.csv, .xlsx)
├── PhantomX\
│   ├── LABELED\<oturum>_PCCT_Aorta_Spleen\{nifti, segmentation, overlay, features}\<seri>\
│   └── RADIOMICS\                         <oturum>_PCCT_Aorta_Spleen_baseline.xlsx + TUM_OTURUMLAR_3D / _2D (.csv, .xlsx)
├── Kod\                                   phantom-labeling, phantom-radiomics, organ-radiomics (+ README'ler, IBSI_UYUM_RAPORU.md)
└── Dokuman\                               PCCT_BILGI_TABANI.md, Data Analysis\, konum/koordinat tabloları
```

- `nifti\<seri>\ct.nii.gz` = serinin NIfTI'si, `dicom_metadata.json` = ilk dilimin bütün DICOM etiketleri.
- `segmentation\<seri>\seg.nii.gz` = TotalSegmentator 2.12.0 (task `total`) **çok etiketli** maske (117 yapı), `label_map.json` = etiket numarası ↔ organ adı. Aorta ve dalak dışındaki organlar da maskede vardır.
- `features\<seri>\features.csv` (3D), `features_2d.csv` (2D), `features.done.json` (sürüm/parametre kaydı).
- `overlays\`, `painted\`, `coronal\`, `overlay\` yalnız görsel kontrol (QC) içindir.

## 3. MANIFEST — iki fantomu birbirine bağlayan tablo

Her satır bir seri. En önemli sütunlar:

| Sütun | Anlamı |
|---|---|
| `cal_series` | CalibraMics klasör adı = `DICOM\` altındaki klasör adı = CalibraMics tablolarındaki `series` |
| `px_series` | PhantomX klasör adı = PhantomX tablolarındaki `series` (**farklı ad şeması!**) |
| `SeriesInstanceUID` | İki fantomu eşleyen **tek güvenilir anahtar** (tablolarda `dcm_SeriesInstanceUID`) |
| `IrradiationEventUID`, `acquisition_id` | Bağımsız ışınlama kimliği (tekrar birimi) |
| `take`, `AcquisitionDateTime` | Oturum içi sıra (CalibraMics adından) ve çekim zamanı |
| `flip`, `yerlesim` | Fantomun yerleşim sınıfı A/B (bkz. §5.3) |
| `rc_x_mm` | Rekon merkezi x; ≤ −20 mm ise dalak FOV tarafından kesilir (bkz. §5.6) |
| `n_dcm`, `n_dcm_no_ext` | Dilim sayısı; `.dcm` uzantısı **olmayan** dosya sayısı |

**Dikkat:** PhantomX klasör adları CalibraMics adlarıyla birebir aynı değildir (`_recon1mm` eki yok; 20260617'de aynı protokolün üç tekrarı `_120_…`, `_80_…`, `_100_…` adını taşır). Eşlemeyi **addan değil UID'den** yapın; en kolayı `MANIFEST.csv`'yi kullanmak.

## 4. Feature tabloları

| | CalibraMics | PhantomX |
|---|---|---|
| Satır | ROI (seri başına **90**: 40 insert × {5 mm, 10 mm} + 8 texture + 2 star) | organ (seri başına 2: aorta, spleen) |
| Sayfalar | `methods`, `insert_ranking`, `2D_base`, `3D_base` (+`corrections`), `PAKET_NOTU` | `methods`, `3D_base`, `2D_base`, `PAKET_NOTU` |
| Insert satırları | yalnız 18 `stat_*` sütunu dolu | — |
| Yeniden örnekleme | yok (0.586 × 0.586 × 1.0 mm) | 1 mm izotropik, tricubic, anti-aliasing kapalı |
| Ayrıklaştırma | bin 25 HU, orijin −2000 HU | bin 25 HU, orijin −1300 HU |
| Yazılım | MIRP 2.5.0, IBSI-1 | MIRP 2.5.0, IBSI-1, TotalSegmentator 2.12.0 |

- `*_baseline.xlsx` dosyaları, asıl oturum workbook'larının **yalnız bu paketteki serilere süzülmüş** hâlidir; feature değerleri değiştirilmedi. `methods` ve `insert_ranking` aynen alındı, dolayısıyla oradaki seri/ROI sayıları oturumun bütün protokollerini beyan eder.
- `TUM_OTURUMLAR_*.csv/.xlsx` bütün oturumların alt alta eklenmiş hâlidir (baştaki `session`, `source_workbook` sütunları eklenmiştir).
- `insert_ranking`: **rank = insert numarası = etiket numarası** (Insert_01 en yoğun, +535 HU → Insert_40, −466 HU). 01–11 taş dolgulu filament, 12–40 PLA (spektral bloklar 12–20, 21–30, 31–40).
- csv okurken: `pd.read_csv(..., dtype=str)` ile başlayın ya da `dcm_*` sütunlarını metin tutun (UID'ler ve `0000` gibi değerler sayıya çevrilince bozulur). csv'ler UTF-8 (BOM'lu), virgül ayraçlı, ondalık nokta; Türkçe Excel'de çift tıklayıp açmak yerine yanındaki `.xlsx`'i kullanın.

## 5. Bilinmesi gereken tuzaklar

1. **kVp etiketleri yanlış olabilir.** Bütün seriler **120 kV**'dir. `kvp`, `kvp_nominal` sütunları, PhantomX klasör adlarındaki kVp, DICOM `SeriesDescription` ("80kVp_…", "100kVp_…") ve `labels_baseline.csv` içindeki `kvp` tarayıcıdaki yanlış protokol kartı adından gelir. Doğru olan DICOM `KVP` etiketi / `dcm_KVP` sütunudur. "80/100 kVp" etiketli seriler aynı protokolün **ayrı ışınlamalarıdır** (tekrar olarak kullanılır).
2. **`PatientName` ile gruplamayın** (tek fantom için 7 farklı değer) ve `SeriesDescription`'ı ayrıştırmayın.
3. **Etiketleme çerçevesi ≠ DICOM çerçevesi.** `metadata.json` içindeki `center_x` ve `slice_idx`, fantomun yerleşimine göre normalize edilmiştir:
   - `yerlesim = A` (`flip=True`, 68 seri): `x_dicom = (Columns − 1) − center_x`, `InstanceNumber = N − slice_idx`
   - `yerlesim = B` (`flip=False`, 24 seri; fantom 180° çevrilmiş): `x_dicom = center_x`, `InstanceNumber = slice_idx + 1`
   - `N` = benzersiz InstanceNumber sayısı. Bunu yeniden yazmak yerine **`Dokuman\cal_coords_corrected_baseline.csv`**'yi kullanın: her ROI için DICOM çerçevesinde piksel (`cx_px_dicom`, `cy_px_dicom`, `inst`) ve hasta koordinatı (`x_mm`, `y_mm`, `z_mm`) hazırdır (doğruluk ≈ 30 µm; o tabloda artık diskte olmayan 15 mm insert satırları da vardır).
4. **ROI patch DICOM'larında geometri etiketi yoktur** (PixelSpacing / SliceThickness / ImagePositionPatient yok). Piksel boyutunu her zaman `DICOM\<oturum>\<seri>\` altındaki asıl seriden alın (FOV 300 → 0.5859375 mm). Patch'lerde RescaleIntercept −1024, daire dışı dolgu `PixelPaddingValue = −32768`. Asıl DICOM'larda RescaleIntercept **−8192**, Slope 1.
5. **ROI şekilleri:** insert = 5 / 10 mm çaplı daire (çubuğun kendisi 20 mm); star = **28 mm** çaplı daire; texture = **eksene hizalı 27.3 mm kare** (döndürülmez). `metadata.json`'daki `size_cm` star/texture için nominal 4.0 / 3.9 yazar (1.43× büyük) ve `angle_deg` şeklin dönme açısı **değil**, ROI'nin fantom merkezine göre konum açısıdır. Insert başına 30 dilim (±15 mm pencere).
6. **Dalağı FOV tarafından kesilen seriler (PhantomX).** Rekon merkezi `rc_x_mm ≤ −20` olan 4 seri var: 20260707 take2 ve take3 (dalak satırı birleşik tablolardan **silinmiştir**; seri klasöründeki `features.csv` hâlâ içerir — klasöre uyarı dosyası kondu), 20260629 take3 ve 20260730 take5 (dalak satırı duruyor; hacim/şekil analizlerinde dışlayın veya duyarlılık testi yapın). Aorta etkilenmez.
7. **Yıldız/texture için oturum etkisi.** 7 oturumda (20260219, 20260227, 20260312, 20260410, 20260512, 20260611, 20260617) fantom z ekseninde farklı yerleşmiştir (yıldız kesiti dilim ≈ 70, diğerlerinde ≈ 200); Star8 crop ortalaması +16…+18 HU, Star16 −10 HU kayar. Zaman serisi/kontrol kartı çalışmalarında bu oturumları ayrı katman sayın. Ayrıca fantom her oturumda hafif eğik durur (−1.7° … +0.7°) ve oturumlar arasında konumu 27 mm (x) / 10 mm (y), dönmesi 4.3° değişir (`Dokuman\KONUM_ANALIZI.md`).
8. **DICOM dosyalarının ~%32'sinde `.dcm` uzantısı yoktur** (ör. `78129984`). Dosyaları `*.dcm` ile süzmeyin; klasördeki her dosyayı okuyun (`pydicom.dcmread`).
9. **Havuzlama kuralları:** `2D_base` ile `3D_base` birleştirilmez (73 sütun iki sayfada özdeştir; texture sütunları farklı hesaplanır). İki fantomun feature değerleri tek tabloda karıştırılmaz (orijin 700 HU farklı, biri yeniden örneklenmiş). CalibraMics'te morfoloji ailesi + `ngl_dc_perc` (küboid maske → dejenere, 25 sütun) ve 12 cebirsel tekrar (`cm_dissimilarity`, `cm_sum_var`, `cm_sum_avg` × 4 agregasyon) kullanılmaz.
10. **20260526** oturumunda PhantomX ağacında DICOM klasörü hiç yoktu; görüntüler CalibraMics ağacından alındı (aynı seriler). Diğer 83 ortak seride iki ağaçtaki DICOM'ların birebir aynı olduğu dosya adı + boyut + MD5 ile doğrulandı.
11. Segmentasyon ortamı oturuma göre değişir (CPU/GPU, iki PyTorch derlemesi; ölçülen maske farkı %0.6–0.7). Cihazı her PhantomX workbook'unun `methods` sayfasından okuyun.

## 6. Hangi proje nereden başlar?

| Proje | Başlangıç noktası |
|---|---|
| **PWLS — HU ölçeği harmonizasyonu** | Insert HU'ları: `CalibraMics\RADIOMICS\TUM_OTURUMLAR_3D` → `stat_mean`, `insert_size = 10mm` (5 mm duyarlılık için). Organ vokselleri: `DICOM\` + `PhantomX\LABELED\…\segmentation\` (saklı çok etiketli maskeler; yeniden segmentasyon gerekmez). Referans D0: 20260731 / 20260805 (10 çekim). Paket yalnız P0 protokolünü içerdiği için doğrudan **Kol A (zamansal)** ve **Kol E (sentetik)** çalışılabilir. |
| **GGS/NPS — kernel harmonizasyonu** | Homojen bölgelerden NPS: `DICOM\` + PhantomX maskeleri (karaciğer dahil, `label_map.json`) + insert 01/12/13 konumları (`cal_coords_corrected_baseline.csv`). **Bu pakette yalnız Qr40f (referans kernel) vardır**; Qr64f/Qr36f kaynak serileri ayrıca istenmelidir. Pakette yapılabilecekler: NPS ölçüm hattı, D0'da çekimler-arası taban, oturumlar-arası kararlılık, sentetik filtre (Kol E). |
| **Phantom segmentasyon / varyans ayrıştırma** | `DICOM\` (geometri etiketleri dokunulmamış: IPP, IOP, PixelSpacing) + `PhantomX\LABELED\…\{nifti, segmentation}`. Pakette **L1** (yazılım tabanı — seçtiğiniz görüntülerde TotalSegmentator'ı yeniden koşarak), **L3** (aynı oturumda 3–10 tekrar çekim) ve **L4** (23 oturum, aynı protokol) basamakları için veri vardır; **L2** (aynı çekimin farklı rekonları) için yoktur. Maskeleri dizin uzayında değil **dünya koordinatında** karşılaştırın. |
| **Yıldız MTF izleme** | `DICOM\` (tam kesitler) + `CalibraMics\LABELED\…\Stars\{Star16,Star8}` + yıldız merkezleri ve kesit aralığı `cal_coords_corrected_baseline.csv`. Sürüm günlüğü için `MANIFEST.csv` → `software` (bu pakette hepsi `syngo CT VB20A`). Yıldız bulma/sınıflama kodu: `Kod\phantom-labeling\codes\analysis\star_classifier.py`, `Kod\phantom-radiomics\extract_star_dicoms.py`. §5.7'deki oturum etkisini hesaba katın. |

## 7. Kod ve doküman

- `Kod\phantom-labeling\` — CalibraMics ROI tespiti ve patch export (`codes\label_series.py`; ROI maskeleri `codes\core\roi_extractor.py`).
- `Kod\phantom-radiomics\` — patch'lerden MIRP feature çıkarımı (`extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py`).
- `Kod\organ-radiomics\` — PhantomX uçtan uca hat (`batch_120kvp.py`, `run_and_verify.py`, `config.py`: MIRP ayarları burada).
- Her ağacın kendi `README.md`'si ve `requirements.txt`'si vardır (mirp 2.5.0, pydicom 3.0.1, SimpleITK 2.5.3, numpy 2.4.1 …). Büyük PyInstaller `dist\` klasörü alınmadı.
- `Dokuman\PCCT_BILGI_TABANI.md` — veri setinin ayrıntılı referansı (klasörler, ad şeması, şemalar, kurallar, bilinen sorunlar). Yollar orada `D:\PCCT` olarak geçer; bu paketteki karşılıkları §2'dedir.
- `Dokuman\Data Analysis\` — 6 Eylül 2026 tarihli tam analiz (md + `PCCT_Veri_Analizi_Raporu.docx/.pdf` + `tables\` + `figures\` + `scripts\`; istatistik yardımcıları `scripts\lib\stats.py`). **Bütün protokolleri** anlatır ve 8 Eylül 2026'daki veri tamamlamalarından önce üretilmiştir; sayıları bağlam için okuyun, kendi sonuçlarınızla karıştırmayın.
- `Dokuman\image_geometry_INSERT1_baseline.csv` (seri başına yerleşim bayrağı ve fantom konumu), `rename_map_baseline.csv` (6 Eyl 2026 klasör adı düzeltmesi: eski ad → yeni ad), `KONUM_ANALIZI.md`, `DENETIM.md` (disk ↔ Excel tutarlılık denetimi).

## 8. Bütünlük

`CHECKSUMS.md5` DICOM, NIfTI ve tablo dosyalarının MD5 özetlerini içerir (biçim: `md5 *göreli\yol`). Paketi başka bir diske kopyaladıktan sonra doğrulamak için:

```
certutil -hashfile "DICOM\20260211\<seri>\<dosya>" MD5
```

Paketin üretim betikleri ve logları asıl diskte `E:\PCCT\work\bitirme_paketi_20260918\` altındadır.
