# KSW PhantomXPro — Radiomics Extraction

KSW Philips Spectral CT 7500 (ve aynı etiketleme çıktısını üreten diğer
tarayıcılar, ör. Siemens SOMATOM) ile çekilmiş **CalibraMics PhantomXPro**
veri setinden **MIRP / IBSI uyumlu** radiomics özellikleri çıkaran
self-contained Python script. Çıktı; filtrelenebilir, renklendirilmiş ve tüm
DICOM metadata'sını içeren tek bir Excel workbook.

**Kapsam:** Insert, Star ve Texture ROI'lerinin **tamamı** işlenir; her biri için
**tüm MIRP base feature aileleri** çıkarılır. IBSI-2 filtre katmanı
**hesaplanmaz** — gerekçesi aşağıda. (Eski sürüm yalnızca insert'leri, sadece first-order
`statistics` ile çıkarıyordu.)

Hedef veri seti kök klasörü (örnek):
`C:\Users\Administrator\Desktop\KSW - labeled`

## Klasör içeriği

| Dosya | Amaç |
|-------|------|
| `extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py` | Tüm pipeline (discovery, NIfTI inşası, MIRP, Excel formatlama). Tek dosya. |
| `PhantomXPro_Radiomics.spec` | PyInstaller reçetesi — dağıtılabilir .exe üretir |
| `KULLANIM.txt` | Son kullanıcı talimatı; exe klasörünün içine gömülür |
| `test_series.txt` | `--series-file` argümanı için örnek seri listesi |
| `requirements.txt` | Sabitlenmiş sürümler — feature değerleri bunlara bağlı |
| `tests/` | pytest paketi (aşağıya bakın) |
| `README.md` | Bu dosya |

### Testler

```bash
"C:/Python314/python.exe" -m pytest tests -q                 # hepsi
"C:/Python314/python.exe" -m pytest tests -q -m "not slow"    # veri seti gerekmeyenler
```

`slow` işaretli testler gerçek veri setlerini veya bir MIRP çalıştırmasını
gerektirir ve veri yoksa atlanır. Paket yedi aşamaya ayrılmıştır:

| Dosya | Neyi kilitler |
|-------|---------------|
| `test_ibsi_params.py` | MIRP'e verilen TÜM ayarlar — beyan edilenler ve MIRP'ten sessizce miras alınanlar |
| `test_patch_reader.py` | Padding/sentinel sözleşmesi; kırpma determinizmi; gerçek veride voksel korunumu |
| `test_geometry.py` | Voxel spacing'in kaynağı, NIfTI geometrisi, dilim sıralaması |
| `test_discovery.py` | Keşif kapısı okuyucudan geçirgen olamaz; kaybedilen hiçbir şey exit 0 ile geçemez |
| `test_reproducibility.py` | Yola bağlı feature'ların atılması; checkpoint'in parametre değişimini yakalaması |
| `test_workbook_layout.py` | Kolon düzeni, blok renkleri, `series` kolonunun kırpılmaması, beyan sayfası |
| `test_multidevice.py` | Üç cihazın DICOM farklılıkları: çekirdek/güç kaynakları, HU tabanı, test-retest, kopya olmama |
| `test_output_workbook.py` | ÜRETİLMİŞ kitap: satır sayısı, dolu feature blokları, ayırt edilebilirlik |

> Not: Dosya adı tarihsel nedenle `..._inserts_only_v4.py` olarak kalmıştır;
> içerik artık insert + star + texture'ın tamamını işler.

## Gereksinimler

Python 3.14 ortamında:

```
mirp
SimpleITK
pydicom
pandas
numpy
openpyxl
itk          # MIRP'in dahili olarak kullandığı, 5.4+
```

Mevcut kurulumda `C:\Python314\python.exe` bu paketleri içeriyor.

## Veri seti yapısı

Script, dataset root'u içinde recursive olarak tarar — root altındaki bir
seriyi de, root altındaki tarih/grup klasörlerinin altındaki serileri de bulur.
ROI'ler **kırpılmış DICOM patch** olarak okunur; `.npy` ve `metadata.json`
artık **gerekmez**. Her seri klasöründe şunlar olmalı:

```
<series_name>/
├── dicoms/                          # *.dcm — voxel spacing + tüm DICOM metadata kaynağı
├── Inserts/<5mm|10mm>/Insert_NN/*.dcm        # kırpılmış insert patch'leri (15mm varsa yok sayılır)
├── Stars/<Star16|Star8>/*.dcm                # merkez yıldız patch'leri (daire)
└── Textures/<label>/*.dcm                    # doku karesi patch'leri (Diagonal, Horizontal,
                                              #   Vertical, Square, Honeylarge, Honeymedium,
                                              #   Honeysmall, Sinusoidal)
```

Bir serinin işlenmeye alınması için `Inserts/` altında DICOM patch'i bulunması
yeterlidir; `Stars/` ve `Textures/` varsa otomatik dahil edilir.

## Çalıştırma

### 1) Doğrudan Python ile

Yol argümanı zorunlu (sabit default yok) ve positional da verilebilir.

```bash
# 1. Hızlı keşif: kaç seri bulundu, grup başına ROI sayıları — hiçbir şey hesaplamaz
"C:/Python314/python.exe" extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py \
    "C:/Users/Administrator/Desktop/20260506_KSW" --dry-run

# 2. Tek seri test (~2 dakika)
"C:/Python314/python.exe" extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py \
    "C:/Users/Administrator/Desktop/20260506_KSW" --max-series 1

# 3. TÜM seriler, 3D + 2D (varsayılan)
"C:/Python314/python.exe" extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py \
    "C:/Users/Administrator/Desktop/20260506_KSW"
# Çıktı: .../20260506_KSW/radiomics_output/radiomics_combined.xlsx

# 4. Üç veri setini sırayla
for d in 20260402_USZ_PCCT 20260506_KSW 29062026-EID_LABELED; do
  "C:/Python314/python.exe" extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py \
      "C:/Users/Administrator/Desktop/$d"
done
```

> **Not**: Flag verilmezse **3D + 2D base** feature'lar üç ROI grubu için de
> çıkarılır (seri başına ~2 dakika). Kesilirse aynı komutu tekrar çalıştırın —
> `_per_series/` altındaki checkpoint'ler sayesinde tamamlanmış seriler atlanır.
> Checkpoint'ler şema damgası taşır: farklı bir pipeline sürümüyle yazılmış bir
> checkpoint yeniden kullanılmaz, yeniden çıkarılır.

> **Paralel çalıştırma (hız)**: Seriler bağımsızdır ve varsayılan olarak
> `min(12, çekirdek-2)` seri aynı anda ayrı process'lerde işlenir — çok çekirdekli
> makinelerde neredeyse doğrusal hızlanma sağlar (MIRP tek ROI'yi kendisi
> paralelleştirmez). `--jobs N` ile ayarlanır: örn. eski sıralı davranış için
> `--jobs 1`, RAM sınırlıysa `--jobs 4`. Örnek ölçüm: 16 çekirdek / 15 GB RAM'de
> 100 seri (3D+2D, insert=statistics) `--jobs 6` ile ~20 dakikada biter;
> sıralı (`--jobs 1`) ise günler sürerdi. Not: her worker ~0.5-1 GB RAM kullanır,
> RAM sıkışırsa `--jobs`'ı düşürün.

### 2) Belirli serilerin listesi (txt) ile

`test_series.txt` içine her satıra bir TAM seri ismi yaz, sonra:

```bash
"C:/Python314/python.exe" extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py \
    --dataset-root "C:/Users/Administrator/Desktop/KSW - labeled" \
    --series-file test_series.txt
```

### 3) Derlenmiş .exe — dağıtılabilir uygulama

```bash
"C:/Python314/python.exe" -m PyInstaller PhantomXPro_Radiomics.spec --noconfirm
```

Çıktı: `dist/PhantomXPro_Radiomics/` — **klasörün tamamı** dağıtılır (tek dosya
değil; one-file build itk/SimpleITK/MIRP yükünü her açılışta temp'e açar,
onlarca saniye kaybettirir ve bazı makinelerde multiprocessing worker'larını
bozar). Klasörü zip'leyip gönderin; `KULLANIM.txt` içine gömülüdür.

Kullanıcıdan istenen tek şey **klasör yolu**:

- Veri seti klasörünü `PhantomXPro_Radiomics.exe` üzerine sürükle-bırak, **veya**
- exe'ye çift tıkla → native klasör seçme penceresi açılır (`tkinter`; yoksa
  yol yazmaya düşer).

Sonra: bulunan seri sayısını gösterir, onay ister, çalışır, bitince sonucu
açmayı önerir. Kurulum, Python, flag gerekmez. Aynı binary üç tarayıcının
(NAEOTOM / Philips 7500 / SOMATOM Edge Plus) çıktısını da işler.

Dağıtımda dikkat edilenler:

- **Sabit veri seti yolu yok.** Eskiden modülde geliştiricinin kendi klasörü
  yazılıydı; bir meslektaş exe'ye çift tıkladığında var olmayan bir klasörü
  tarayıp "0 seri" alıyor ve sebebini gösteren hiçbir şey olmuyordu. Yol artık
  komut satırından, sürükle-bıraktan veya klasör seçiciden gelir.
- **`--jobs` RAM'e göre sınırlanır** (`_default_jobs`). Sadece çekirdek sayısına
  bakmak 20 çekirdek/31 GB geliştirme makinesinde doğru, 8 GB dizüstünde takla
  attırıyordu — exe'yi alan kişilerin makinesi ikincisi.
- **UPX kapalı.** numpy/itk/SimpleITK native DLL'lerini sıkıştırmak sessiz yükleme
  hatalarının ve antivirüs yanlış pozitiflerinin bilinen kaynağı; bu binary bir
  hastane ağında dolaşacak.
- **`matplotlib` ve `skimage` dışlanamaz** — MIRP ikisini de import ediyor
  (`generic_image.py`, `morph_3d_data.py`, `dzm_matrix.py`). Boyut için atmak
  build'i çalışma anında `ImportError` ile bozar. Script `MPLBACKEND=Agg`
  ayarlar, böylece GUI backend'i aranmaz.
- Boş sonuç artık açıklanır (`_explain_empty_dataset`): beklenen klasör yapısı,
  en sık üç hata ve o klasörde gerçekte ne olduğu yazdırılır.

## CLI argümanları

| Argüman | Default | Açıklama |
|---------|---------|----------|
| `dataset_root_pos` (positional) | — | Veri seti kök klasörü; .exe üstüne sürükle-bırak bunu doldurur |
| `--dataset-root` | **(yok)** | Positional ile aynı işi yapar. İkisi de verilmezse .exe klasör seçme penceresi açar; sabit default kasıtlı olarak kaldırıldı |
| `--output-dir` | `{dataset}/radiomics_output` | Çıktı klasörü |
| `--series` | (hepsi) | Sadece bu substring'i içeren seriler |
| `--series-file` | (yok) | Tam isimleri içeren txt — `--series`'ten önce gelir |
| `--max-series N` | (hepsi) | İlk N seri (filtreleme sonrası) |
| `--no-3d` | False | 3D extraction'ı atla |
| `--no-2d` | False | 2D extraction'ı atla |
| `--jobs` / `-j` | çekirdek **ve RAM** ile sınırlı | Aynı anda PARALEL işlenecek seri sayısı (her seri ayrı process). Seriler bağımsız olduğu için çekirdekle neredeyse doğrusal hızlanır. Default `min(12, çekirdek-2)` iken ayrıca boş RAM'e göre kısılır (worker başına ~1,2 GB, 2 GB işletim sistemine bırakılır) — bkz. `_default_jobs`. `1` = sıralı. |
| `--no-skip` | False | Checkpoint olsa bile yeniden çıkar |
| `--dry-run` | False | Seri listesini + grup başına ROI sayılarını yaz, çık |
| `--yes` / `-y` | False | Frozen .exe'de onay istemini atla |

## Pipeline (her seri için)

1. `dicoms/` DICOM header'larından voxel spacing + tüm metadata alınır.
2. `Inserts/`, `Stars/`, `Textures/` altındaki kırpılmış DICOM patch'leri
   keşfedilir (40 Insert × {5,10}mm + 2 Star + 8 Texture = 90 ROI).
3. Her ROI'nin patch'leri Z ekseninde yığılıp HU'ya çevrilir. Etiketleme
   pipeline'ı ROI dışını int16 -32768 (HU ≈ -33792) olarak yazdığı için
   dairesel ROI'ler kare bir patch'in içinde gelir. ROI şekline göre:
   - **Kare ROI'ler (8 texture deseni)** patch'i tamamen doldurur; hiç
     dokunulmadan geçirilir, değerleri birebir korunur.
   - **Dairesel ROI'ler (40 insert × 3 boyut, Star16, Star8)** yalnızca gerçek
     voxel içeren **en büyük kareye kırpılır**. Maskeleme değil, kırpma:
     çıkan patch'te sentinel hiç bulunmaz.
   - **Maske** artık patch'in tamamıdır (hepsi 1), yani `roi_mean_hu` /
     `roi_std_hu` / `voxel_count` MIRP'in gördüğü voxel'lerin aynısını anlatır.

> **Kırpma neden maskelemekten daha önemli.** Maske filtre katmanını korumaz:
> MIRP her filtreyi **görüntünün tamamı** üzerinde konvolve eder, maskeyi
> ancak sonra uygular. Patch'in herhangi bir yerindeki -33792 voxel'i, ROI
> sınırına yakın **maske içi** yanıt değerlerine sızar ve resegmentasyon bunu
> geri alamaz (o baz HU'yu keser, yanıt değerlerini değil). Gerçek bir Star16
> ROI'sinde ölçüldü: sentinel her filtre yanıt feature'ını değiştiriyordu,
> Daubechies-3 L1'de ~%4100'e varan farkla. Padding'siz patch'te çekirdek
> patch kenarında beyan edilmiş `*_boundary_condition` ile karşılaşır — bu
> IBSI-2'nin tanımladığı davranıştır.

Kare kırpmanın voxel bütçesi (0.586 mm in-plane, 30 dilim):

| ROI | patch | disk | kare | kalan |
|-----|-------|------|------|-------|
| Insert 5mm | 11×11 | 1830 | 7×7 = 1470 | %80 |
| Insert 10mm | 19×19 | 6750 | 13×13 = 5070 | %75 |
| Star16 / Star8 | 49×49 | 53670 | 33×33 = 32670 | %61 |
| Texture (8 desen) | 49×49 | 72030 | 72030 | %100 |

**Insert boyutları.** Yalnızca **5mm ve 10mm** çıkarılır (`INSERT_SIZES` sabiti).
Diskteki bir `15mm/` klasörü görmezden gelinir: üç boyut da aynı silindiri
örneklediği için 15mm, insert başına üçüncü bir korelasyonlu satır ve ek süre
getiriyordu, bilgi getirmiyordu. Seri başına insert ROI sayısı 120 → **80**.

**Çıkarılan feature'lar.** Aşağıdakiler hiçbir tabloya yazılmaz, çünkü ölçüm
değil, hesap yönteminin yan ürünüler:

| Feature | Neden |
|---------|-------|
| `morph_moran_i`, `morph_geary_c` | MIRP bunları 1000+ voxel'de Monte Carlo ile tahmin ediyor ve örnekleyicinin seed'i **dosya konumunu** içeriyor; aynı voxel'ler farklı klasörde farklı değer veriyor. Tekrarlanabilir değiller. |
| `ivh_v10/v25/v50/v75/v90`, `ivh_diff_v10_v90`, `ivh_diff_v25_v75` | MIRP, IVH yoğunluk eksenini resegmentasyon aralığından alıyor (`ivh_features.py:50`). Bu pipeline o aralığı kodlama sınırlarına (−2000…31743) sabitlediği için %10 seviyesi +1374 HU'ya düşüyor — verinin çok üstünde — ve **tüm hacim oranları tam 0,0 çıkıyor**. Ölçüm değil. |
| `loc_peak_loc`, `loc_peak_glob` | 300 voxel üstü ROI'lerde MIRP 1 cm³'lük küreyi görüntü üzerinde konvolve ederken dışarısını `cval=-1000` ile **uyduruyor** (`local_intensity_features.py:96-105`). Çekirdek yarıçapı 6,2 mm = 10,6 piksel; 33×33 patch'te neredeyse her voxel sınırın erişiminde. Ölçüldü: uydurma dolgu yerine kenar tekrarı konulunca 10 ROI'nin **9'unda** değer değişiyor, %159'a varan farkla. |

`ivh_i10..i90` ve `ivh_diff_i*` **korunuyor** — bunlar gerçek HU eşikleri ve
eksen seçiminden etkilenmiyorlar.

> IVH hacim oranlarını geri kazanmanın tek yolu `ivh_discretisation_method`'ı
> `fixed_bin_number` yapmak; ama o zaman eksen her ROI'nin kendi aralığından
> türüyor ve ROI'ler arası karşılaştırılabilirlik gidiyor — sabit çapanın
> korumak için var olduğu şey tam olarak bu. Ayrıca `ivh_i*` HU olmaktan çıkıp
> kutu indeksine dönüşüyor.

**Çapanın beyan zorunluluğu.** Fixed-bin-size'da gri seviye indeksine ağırlık
veren her feature (`*_lgre`, `*_srlge`, `*_lrlge`, `*_lglze`, `*_lzlge`,
`*_hdlge`, `*_lgce` ve GLCM'in `inv_diff_norm` türevleri) çapanın **nerede**
olduğuna bağlıdır — çünkü ağırlık 1/i² biçimindedir ve i mutlak kutu indeksidir.
Ölçüldü: çapayı −2000'den −1100'e almak (ikisi de verinin altında, yani veri
bölümlemesi birebir aynı) 218 feature'ın **43'ünü** değiştiriyor, bazılarını
%9000'e varan oranda.

Bu bir hata değil, FBS'in doğası. Önemli olan üç şey:

1. **İç tutarlılık korunuyor.** Çapa modül sabiti, yani her ROI ve her seri aynı
   çapayı kullanıyor. Protokoller/rekonstrüksiyonlar arası karşılaştırmalarınız
   geçerli.
2. **Dış karşılaştırılabilirlik yok.** Başka bir çapa kullanan bir çalışmanın
   sayılarıyla bu sayılar kıyaslanamaz. IBSI'nin çapayı beyan etmeyi şart
   koşmasının sebebi tam olarak budur.
3. **Alternatifin bedeli var.** Daha dar bir çapa bu feature'ları daha ayırt
   edici yapar ama çapanın altındaki gerçek voxel'leri 1. kutuya sıkıştırır.
   Aynı anda (a) cihazdan bağımsız, (b) hiç sıkıştırmayan ve (c) indeks-ağırlıklı
   feature'ları bozmayan bir çapa yoktur; ikisini seçebilirsiniz. Burada (a) ve
   (b) seçilmiştir.

Metodlar bölümünde beyan edilecek tam liste (İngilizce, olduğu gibi
kullanılabilir):

> Radiomics features were extracted with MIRP 2.5.0 using IBSI-1 compliant
> feature definitions. Discretisation: fixed bin size, bin width 25 HU, grey
> level origin −2000 HU (the int16 encoding floor, chosen so that it lies below
> the lowest value any reconstruction produces — the deepest observed was
> −1246 HU). A re-segmentation range of (−2000, 31743) HU was supplied because
> MIRP derives the fixed-bin-size origin from it; **it excludes no voxel**, and
> the pipeline raises rather than proceeding if one ever falls outside.
> **No interpolation was performed**: voxels remained anisotropic, from
> 0.586 × 0.586 × 1.0 mm at FOV 300 to 0.977 × 0.977 × 1.0 mm at FOV 500.
> Circular ROIs were cropped to their largest padding-free square, retaining
> 43–80 % of the segmented voxels depending on ROI size and FOV; **the analysis
> mask is consequently a rectangular cuboid**, so morphological features
> describe the crop rather than the object and are not reported for inserts.
> Texture aggregation: 3D averaged and 3D volume-merged, with 2D/2.5D variants
> reported separately. No IBSI-2 filter-based features were computed. Twelve
> features were excluded as non-reproducible (Moran's I, Geary's C, the IVH
> volume fractions and the local intensity peaks).

Çapa neden −2000: gerekçe ve ölçümler `_DISCRETISATION_ANCHOR_HU`'nun yanındaki
yorumda. Kısaca, daha sığ bir çapa (−1000/−1024/−1100) gerçek hava vokselini
tek bir gri seviyeye çökertiyor, daha derini (−4000/−8000) yoğunluk-ağırlıklı
özelliklerin ayırt etme gücünü her katta yarıya indiriyor.

**Kutu şeklinin yan etkisi.** Kare kırpma sonrası ROI bir dikdörtgen prizma
olduğu için `morph_vol_dens_conv_hull` ve `ngl_dc_perc` her ROI'de tam 1,0
çıkıyor. Bunlar yanlış değil (bir kutunun dışbükey zarfı kendisidir), sadece
artık bilgi taşımıyorlar.

## Excel çıktısı

`{output_dir}/radiomics_combined.xlsx` — filtrelenebilir workbook:

| # | Sheet | İçerik | Filtre |
|---|-------|--------|--------|
| 1 | `methods` | IBSI beyan listesi — canlı sabitlerden üretilir, sayılarla asla ayrışmaz | — |
| 2 | `series` | 1 satır/seri: DICOM tag'lerinden okunan akiz + rekonstrüksiyon parametreleri | Excel Table |
| 3 | `insert_ranking` | 40 insert, **etiketleme sırasında**: rank = insert numarası (Insert_01 → 1), böylece overlay/`label` numarasıyla fark olmaz; seri- ve boyut-arası ortalama mean_HU + data bar | AutoFilter |
| 4 | `3D_base` / `2D_base` | 1 satır/ROI/seri × IBSI base feature'lar + **tüm DICOM tag'leri** `dcm_*` kolonları olarak | AutoFilter |

Ayrı bir `dicom_metadata` sayfası **yok**: etiketler feature satırlarıyla birlikte
geziyor, böylece bir satırı okumak ikinci bir sayfaya bakmayı gerektirmiyor ve bir
tag'in okunabileceği tek bir yer kalıyor. `TOC` de kaldırıldı — gezinme değeri
yoktu, yerini `methods` aldı.

### Feature sayfalarının kolon düzeni (sabit)

Düzen `_COMPACT_META_ORDER` ile sabitlenmiştir ve soldan sağa "hangi ROI →
hangi kayıt → hangi çekim koşulu" sırasını izler:

```
A label   B insert_size            (yeşil: hangi ROI)
C series  D n_vox                  (sarı: kayıt/izlenebilirlik)
E recon_k E..L k_strength kVp mAs FOV SliceThickness me_keV repeat
                                  (mavi: çekim koşulları)
M...      dcm_* tag'leri           (mavi: yükseltilenler, sonra sarı: kalanlar)
...       feature kolonları
son       section (gizli, renklendirme yardımcısı)
```

Dondurulan sadece satırı **tanımlayan** dört kolon (A–D). Çekim koşulları da
solda ve renkli ama dondurulmuyor, yoksa ekranın yarısını yer.

> Kolon düzeni testlerle kilitli (`tests/test_workbook_layout.py`). Konumsal
> okuyan bir downstream analiz varsa başlık adına göre okumaya geçirilmeli:
> düzen sabittir ama sabit *bu* düzendir, daha eski bir sürümdeki değil.

Düzeltilen kolonlar:

- **`conv_k` → `k_strength`.** Eski `conv_k`, `recon_k` ile *aynı* DICOM
  tag'inden okunuyordu, yani kernel adını tekrar eden bilgisiz bir kolondu.
  Yerine gerçek rekonstrüksiyon gücü geldi — kaynağı **yalnızca DICOM tag'leri**,
  klasör adı değil (bkz. aşağıdaki cihaz tablosu).
- **`SliceThickness` eklendi.** Daha önce `_COMPACT_DROP` içinde olduğu için
  feature sayfalarına hiç yazılmıyordu.
- **`repeat` eklendi.** Bu fantom test-retest çekiliyor; aynı protokol birden
  çok kez görünüyor. Sütun yazım anında türetilir (`_assign_repeat_index`):
  aynı (recon_k, k_strength, kVp, mAs, FOV, kalınlık, me_keV) tuple'ına sahip
  seriler `AcquisitionTime` sırasına göre 1..n numaralanır. Önceden bu kolon
  vardı ama her zaman boştu ve çıktıdan düşürülüyordu.

> `insert_ranking`, `roi_mean_hu` özet sütununu kullanır; insert feature politikası
> (`statistics`/`all`) fark etmeksizin çalışır. Her insert için sıralama, o
> insert'in **tüm seriler ve çıkarılan tüm boyutlar (`INSERT_SIZES` = 5/10 mm)**
> üzerinden alınan ortalama HU'suna göre hesaplanır.

Görsel öğeler:
- Dondurulmuş başlık satırı + satırı tanımlayan kolonlar
- ROI kategori başına satır rengi (40 Insert = mavi; STAR16 ve STAR8 alt-grupları farklı pastel renkler)
- Meta blok renkleri: yeşil = hangi ROI, sarı = kayıt, mavi = çekim koşulu
- `series` kolonu 60 karakter genişlikte sabitlenir (`_WIDE_DISPLAY_COLUMNS`);
  bazı satırları ayıran tek şey uzun seri adı olduğu için kırpılmaz
- kVp / mAs / me_keV 3-renkli scale conditional formatting
- **Sayı gösterimi kolon başına seçilir.** Sabit `0.000` biçimi ölçülerek
  sorunlu bulundu: bir doku ROI'sinde 207 öznitelik kolonunun 102'si ekranda
  taşıdığından az sayıda değer gösteriyordu ve `rlm_lrlge_3d_v_mrg` 80 farklı
  protokol değerini iki string olarak yazıyordu — üstelik o kolonun yayılımı
  ortalamasının %71'i. Saklanan değerler her zaman tam hassasiyette (formüller
  ve pandas etkilenmiyordu), ama bu kitap Excel'de okunmak için var: ekranda
  sabit görünen bir öznitelik incelenmez. Biçim artık kolonun **en dar ROI-içi
  yayılımını** ~100 basamağa bölecek şekilde seçiliyor (bağlayıcı kısıt bu,
  çünkü karşılaştırma tek bir ROI'yi protokoller boyunca okur), uç değerlerin
  biçimi kabalaştırmaması için 5–95 yüzdelik aralığı da hesaba katılıyor ve
  genişlik `####` çıkmayacak şekilde sınırlanıyor. `tests/test_output_workbook.py`
  bunu 50 ROI × 207 kolon üzerinde doğruluyor.
- Tab renkleri: methods kırmızı / series mavi / insert_ranking yeşil / 2D turuncu / 3D mor

### Cihazlar arası: rekonstrüksiyon gücü nereden okunur

Rekonstrüksiyon gücü için tek bir standart DICOM tag'i yok ve üç cihaz üçünü de
başka yere yazıyor. `_reconstruction_identity` sırayla şunlara bakar; hiçbir
adımda klasör adına başvurulmaz:

| Cihaz | Kaynak | Değerler |
|-------|--------|----------|
| NAEOTOM Alpha | `(0029,1043)` private | IQ level 1–4 |
| SOMATOM Edge Plus | `ConvolutionKernel`'in 2. elemanı (`['Br40f','3']`) | ADMIRE 1–5 |
| SOMATOM Edge Plus | tek değerli çekirdek (`Br40f`) | boş = iteratif değil |
| Philips 7500 | `(01F7,109B)` "iDose Level" | 4 |
| Philips 7500 | `(00E1,1040)` "Image Label" (`IMR_BR_3`) | IMR 1–3 |

Çok değerli `ConvolutionKernel`'e `str()` uygulamak Python liste yazımı
(`['Br40f', '3']`) üretiyordu; aynı çekirdek 100 SOMATOM serisinin 70'inde bu
biçimde, 30'unda `Br40f` olarak yazılıyor ve kolon Excel'de gruplanamıyordu.
Artık `recon_k` yalnızca çekirdek adını, `k_strength` gücü taşır.

**Ayırıcı tag'ler mavi blokta.** Çözümlenmiş meta kolonları bazen bir seriyi
tekil olarak tanımlamıyor (ör. NAEOTOM setinde 24 seri aynı tuple'ı paylaşıyor —
gerçek test-retest tekrarları). Bu yüzden seriyi komşusundan ayıran tag'ler mavi
akiz bloğuna yükseltildi: `dcm_SeriesDescription`, `dcm_SeriesNumber`,
`dcm_AcquisitionNumber`, `dcm_AcquisitionTime`, `dcm_CTDIvol`, `dcm_(0029,1043)`,
`dcm_(00E1,1040)_Image_Label`, `dcm_(01F7,109B)_iDose_Level`.

**Filtreleme örneği**: `series` sayfasında `kVp` başlığındaki ▾ oka tıkla → `120` seç → sadece 120 kVp serileri kalır.

Yan ürünler:
- `{output_dir}/_per_series/{series_name}.xlsx` — checkpoint (yarıda kalan run resume edilebilir)
- `{output_dir}/_nifti_cache/{series_name}/{ROI}_image.nii.gz` ve `..._mask.nii.gz` — intermediate NIfTI'lar

## Renk paleti

- **Inserts (40 ROI)** — light blue `#D6EAF8` (hepsi tek renk)
- **STAR16** Star16=yellow, Diagonal=peach, Horizontal=coral, Square=purple, Vertical=blue
- **STAR8** Star8=gold, Honeylarge=salmon, Honeymedium=mid salmon, Honeysmall=red, Sinusoidal=violet

## Çalıştırma sonunda ne raporlanır

Sessiz kayıp bırakmamak için üç blok basılır ve **eksik bir çalıştırma exit
kodu 2 döner** (temiz çalıştırma 0), böylece otomatik bir çağıran kısmi sonucu
tam sanamaz:

| Blok | Ne zaman | Ne anlatır |
|------|----------|------------|
| `REJECTED FOLDERS` | keşiften hemen sonra | Seri gibi görünen ama işlenmeyen klasörler (CT yok, insert yok). 80 klasörden 77 seri çıkarsa sebebi burada yazar. |
| `METADATA WARNING` | keşiften hemen sonra | Klasör adının DICOM tag'iyle çeliştiği yerler. Değer daima tag'den alınır. |
| `FAILED SERIES` | çalıştırma sonunda | Hiç satır üretmeyen seriler (çöken ya da boş). |
| `SKIPPED ROIs` | çalıştırma sonunda | Diskte olup feature üretmeyen ROI'ler, sebebiyle. |

Checkpoint'ler kendilerini doğrular: damgada şema sürümü, **raporlanan feature
setinin kendisi** ve yazıldığı sırada atlanan ROI sayısı bulunur. Feature seti
değişirse eski checkpoint'ler otomatik geçersiz olur — kimsenin bir sürüm
numarasını yükseltmeyi hatırlaması gerekmez. Resume edilen bir çalıştırma,
checkpoint'e kaydedilmiş atlanan ROI'leri kendi hesabına geri alır, yani kısmi
bir çıkarım resume ile temiz görünmez.

## Notlar

- **IBSI-2 filtre katmanı hiç yok** — tek bir filtre bile hesaplanmıyor.
  `spatial_filters` MIRP'e verilmiyor ve bu `tests/test_ibsi_params.py::
  test_no_ibsi2_filter_layer` ile kilitli. Gerekçe: her konvolüsyon görüntü
  kenarının ötesinde değer **uydurmak** zorunda ve MIRP yalnızca
  reflect / constant / nearest / mirror / wrap sunuyor — hepsi veri sentezler,
  "yalnızca geçerli bölge" kipi yok. Bu kırpılmış patch'lerde uydurulan kenar
  baskın: FOV 300'de LoG çekirdeği 21×21 piksel, 33×33'lük bir yıldız
  patch'inde yalnızca 13×13'lük çekirdek bölge kenardan bağımsız kalır; 7×7'lik
  bir 5 mm insert'te patch'in tamamı kenardır. Böyle bir feature fantomu değil
  sınır koşulunu ölçer. Ayrıntılı gerekçe modül docstring'inde.

  Filtreler, etiketleme çıktısı her ROI'nin etrafına **en büyük çekirdek
  yarıçapı kadar GERÇEK doku payı** koyduğu anda geri gelebilir (0.586 mm'de
  21×21 LoG için ~6 mm). O zamana kadar hesaplanmaması bir eksiklik değil,
  ölçüm olmayan bir sayıyı yayımlamama kararıdır.

  İkinci bir engel: izotropik yeniden örnekleme yapılmıyor. MIRP 3D Mean
  filtresini non-isotropic grid'de (0.586×0.586×1.0 mm) uygulamayı reddedip
  `NotImplementedError` fırlatıyor, ve IBSI-2 filtre tanımları izotropik voksel
  varsayar. Bunu interpolasyonla çözmek IBSI zincirine bir interpolasyon adımı
  ekler ve **diğer tüm feature'ları** değiştirir (ölçüldü: 218 feature'ın
  202'si, en fazla %146) — yani sessizce yapılacak bir şey değil.
- **Reseg (−2000, 31743)** hiçbir vokseli elemiyor; tek işi fixed-bin-size gri
  seviye orijinini sabitlemek (MIRP'te ayrı bir çapa parametresi yok). Soft
  tissue varsayılanı [-1000, 400] hem yoğun insert'leri (HU > 400) hem de
  doku/yıldız içindeki hava boşluklarını (ölçülen en düşük −1246 HU) elerdi.
  Aralığın dışına çıkan bir voksel görülürse çalışma sessizce devam etmiyor,
  `_assert_within_discretisation_range` hata veriyor.
- **Resume**: kesilen bir run'ı sürdürmek için aynı komutu tekrar çalıştır —
  `_per_series/*.xlsx` checkpoint'leri varsa o seriler atlanır. Force etmek için `--no-skip`.
- **Recursive discovery**: dataset root altında doğrudan seri yoksa, root'un alt
  klasörleri (tarih/grup) taranır.
