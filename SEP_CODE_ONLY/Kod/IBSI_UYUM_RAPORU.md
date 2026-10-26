# IBSI Uyum ve Tekrarlanabilirlik Denetimi

Tarih: 2026-07-29 · MIRP 2.5.0, numpy 2.4.1, scipy 1.17.0, Python 3.14.2

Bu rapordaki her sayı, kod okunarak değil **çalıştırılarak** ölçüldü. Kullanılan
veri: `Desktop\20260402_USZ_PCCT` (fantom patch'leri) ve
`Desktop\20260227_USZ_PCCT` (72 seri, CT + TotalSegmentator organ maskeleri).

---

## Uygulanan düzeltmeler (2026-07-29)

Aşağıdaki bulguların tamamı düzeltildi. Ölçümler yerinde bırakıldı, çünkü
düzeltmelerin gerekçesi onlar.

| # | Düzeltme | Dosya | Öznitelik değerleri |
|---|---|---|---|
| 1 | Padding ayırıcı çapadan ayrıldı (`-20000` / `-2000`) + aralık dışı voksel artık hata veriyor | `extract_...v4.py` | Değişmedi (mevcut veride aralık dışı voksel yok) |
| 2 | `anti_aliasing: False` açıkça beyan edildi | `phantomxpro/config.py` | **Değişti** |
| 3 | Resegmentation `[-1000,400]` → `[-1300,3100]`, sıfır voksel eliyor | `phantomxpro/config.py` | **Değişti** |
| 4 | Docstring + README çelişkileri giderildi, tam metod beyanı yazıldı | `extract_...v4.py`, `README.md` | Değişmedi |
| 5 | `requirements.txt`, checkpoint parametre hash'i, provenance kaydı | üç dosya | Değişmedi |
| + | Sınır dışı kırpma artık uyarı veriyor (üç etiketleme ağacına da uygulandı) | `roi_extractor.py` | Değişmedi |

Ayrıca `spline_order: 3` phantomxpro'da artık açıkça yazılı — sapma değil, IBSI
Config E'nin kullandığı tricubic; miras alınmak yerine beyan edilmiş olması için.

**Test durumu: 131 test, hepsi geçiyor.** Daha önce kasıtlı olarak başarısız
bırakılan dört test artık geçiyor; kod değiştiği için eski davranışı doğrulayan
testler yeni doğru davranışı doğrulayacak şekilde güncellendi.

**Uçtan uca doğrulandı:** bir gerçek seri üzerinde 90/90 ROI, 3510 satır, temiz
çıkış. Checkpoint damgası test edildi — çapa değiştirilince eski checkpoint
doğru şekilde reddediliyor.

> **Yeniden çıkarım gerekiyor.** 2 ve 3 numaralı düzeltmeler organ hattının
> öznitelik değerlerini değiştiriyor. Fantom hattının değerleri değişmedi, ama
> checkpoint şeması v3'ten v4'e çıktığı için eski checkpoint'ler yine de
> yeniden hesaplanacak.

---

## Üç soruya cevap

### 1. Kod padding için yeni piksel verisi uyduruyor mu?

**Etiketleme exporter'ı uyduruyor; çıkarım hattı bunu sonuçlara sızdırmıyor.**

`output/exporter.py:83` daire dışındaki her pikseli stored `-32768` yazıyor;
`RescaleSlope=1 / Intercept=-1024` altında bu **HU −33792** olarak geri geliyor.
Gerçek patch'lerde ölçülen oranlar:

| ROI | patch | uydurma padding |
|---|---|---|
| Insert 5 mm | 11×11 | **%49,6** |
| Insert 10 mm | 19×19 | **%37,7** |
| Star16 | 49×49 | **%25,5** |
| Texture (8 desen) | 49×49 | %0 |

Çıkarım hattı bunu **maskelemek yerine kırpıyor**: en büyük padding'siz kareyi
kesip alıyor, böylece sentinel MIRP'e hiç girmiyor. Doğrulandı — kırpılmış
hacimlerde `-33792` yok, gerçek texture patch'lerinde de sentinel yok.

> Karar doğru, ama koruma **tek bir büyüklük eşiğine** (`hu > -2000`) dayanıyor
> ve bu eşik iki ayrı projede yaşayan iki sabit arasındaki yazısız bir sözleşme.
> Artık test ediliyor.

### 2. IBSI ile %100 uyumlu mu?

Soruyu iki katmana ayırmak gerekiyor, çünkü cevapları farklı.

**Katman 1 — MIRP'in kendisi: ✅ DOĞRULANMIŞ.**
Kullanıcı MIRP'i IBSI referans verisine karşı bağımsız olarak test etmiş
(`CalibraMics Thesis\2 - Radiomics Feature Extraction\IBSI_test\`). Sonuç:
IBSI-1'de **1804/1804 feature, %100** — dijital fantom (482) + Config A (346),
B (346), C (210), D (210), E (210). IBSI-2 filtre konfigürasyonları da geçmiş.
Bu katman kapalı; feature tanımları ve MIRP'in matematiği doğru.

**Katman 2 — pipeline'ın MIRP'i hangi ayarlarla çağırdığı: ⚠️ AÇIK.**
Referans doğrulama, MIRP'i **IBSI'nin kendi konfigürasyonlarıyla** çalıştırıyor.
Üretimdeki iki hat ise farklı ayarlar kullanıyor:

- **`pcct-radiomics-mirp`** — yapı olarak **IBSI Config A'ya çok yakın**:
  interpolasyon yok, fixed bin size 25 HU. Farkı sadece resegmentation aralığı
  ve 3B/2B tercihi. Meşru ve beyan edilebilir bir konfigürasyon; feature
  değerlerini değiştirecek bir düzeltme gerekmiyor.

- **`phantomxpro-mirp`** — **doğrulanmış konfigürasyonlardan biri değil.**
  Kanıt kullanıcının kendi betiğinde: `validate_ibsi.py` içinde interpolasyon
  yapan **her** konfigürasyon (B, C, D, E) ve IBSI-2 için satır şu —

  ```python
  anti_aliasing=False,               # IBSI: no anti-aliasing
  ```

  Üretimdeki `config.py` ise bu parametreyi hiç anmıyor, dolayısıyla MIRP
  varsayılanı `True` devrede. Yani hat, kendi doğrulamasının açıkça kapattığı
  bir ayarla çalışıyor.

  Not: `spline_order=3` (tricubic) bir sapma **değil** — Config E tam olarak
  onu kullanıyor. Tek sorun anti-aliasing.

### 3. Tekrarlanabilir mi?

Aynı makinede aynı girdi için **evet**. Makineler arası ve zaman içinde **hayır**:
sürüm sabitlemesi yok, checkpoint parametre değişimini yakalamıyor,
phantomxpro'da iki feature dosya yoluna bağlı.

---

## Bulgular (önem sırasına göre)

### B1 · phantomxpro: anti-aliasing IBSI'de yok, ama açık — **145/177 feature etkileniyor**

`config.py` `new_spacing=1.0` veriyor ama `anti_aliasing`'i hiç anmıyor. MIRP
varsayılanı `True` (β=0,98), dolayısıyla yeniden örnekleme öncesi Gauss
ön-yumuşatması uygulanıyor. Aynı bayrak **ikili maskeye de** geçiyor.

İki ağırlaştırıcı gerçek:

- `mirp/_image_processing/anti_aliasing.py:26-27` — küçültme koruması **yorum
  satırı yapılmış**. Filtre `interpolate=True` olan her seride çalışıyor;
  büyütmede de, spacing zaten 1 mm olduğunda da.
- MIRP'in kendi IBSI karşılaştırma yolu `anti_aliasing=False` kullanıyor
  (`generic_image.py:203`). Yani açık bırakmak, MIRP'in IBSI uyumlu saydığı
  konfigürasyondan sapmak demek.

**Ölçüldü:** kapatıldığında 177 feature'ın **145'i** değişiyor. `stat_min`
7 → −57, `ngt_complexity` 12 kat, `cm_clust_prom` 16 kat.

*Düzeltme:* `MIRP_SETTINGS`'e `"anti_aliasing": False` ekleyin — ve önceki tüm
çıktıları yeniden üretin.

### B2 · pcct: izotropik yeniden örnekleme yok — **202/218 feature etkileniyor**

`new_spacing` hiç verilmiyor → `interpolate=False`. Gerçek voksel FOV 300'de
0,586×0,586×1,0 mm, yani **1,71× anizotropik**. `by_slice=False` ve
`3d_average`/`3d_volume_merge` ile 3B GLCM/GLRLM yön vektörleri fiziksel olarak
eşit uzunlukta değil.

Daha kötüsü: anizotropi **FOV'a göre değişiyor** (FOV 500'de 0,977×0,977×1,0,
neredeyse izotropik). Çalışma FOV'ları karşılaştırdığı için FOV etkisi ile
geometri artefaktı birbirine karışmış durumda.

**Ölçüldü:** izotropik yeniden örnekleme açıldığında 218 feature'ın **202'si**
değişiyor, en fazla %146.

Ayrıca iki hat birbiriyle çelişiyor: aynı kurum, aynı fantom — biri yeniden
örnekliyor, diğeri örneklemiyor.

### B3 · pcct: maske ROI değil, kutu — morfoloji ailesi kırpmayı ölçüyor

Kırpma sonrası `binary = np.ones(...)`. Doğrulandı: `np.unique(mask) == [1]`,
ve `morph_vol_dens_conv_hull` tam **1,0** çıkıyor (bir kutunun konveks örtüsü
kendisidir). Tüm morfoloji ailesi fantomu değil kırpma algoritmasını tanımlıyor.

Kırpmanın bedeli ölçüldü:

| ROI | FOV 300 | FOV 500 |
|---|---|---|
| Insert 5 mm | 1830 → 1470 (%80,3) | 630 → **270 (%42,9)** |
| Insert 10 mm | 6750 → 5070 (%75,1) | 2670 → 1470 (%55,1) |
| Star16 | 53670 → 32670 (%60,9) | 19710 → 13230 (%67,1) |
| Texture | %100 | %100 |

FOV 500'de 5 mm insert **3×3×30'luk bir kutuya** iniyor.

### B4 · phantomxpro: yumuşak doku aralığı kemiğe uygulanıyor

`resegmentation_intensity_range = [-1000, 400]` **117 organın hepsine**
uygulanıyor. Gerçek veride ölçülen kayıp:

| Organ | atılan voksel |
|---|---|
| `vertebrae_T10` | **%69,5** |
| `rib_left_12` | %57,5 |
| `rib_left_11` | %51,9 |
| `rib_right_11` | %49,8 |
| `vertebrae_T11` | %38,8 |

Karşılaştırma: `spleen`, `liver`, `autochthon_*` gibi yumuşak doku organlarında
kayıp **%0**. Yani kemik feature'ları kortikal kemiği değil, ilik ve kısmi hacim
kenarını tanımlıyor. IBSI bunu yasaklamıyor ama `config.py` sadece "CT yumuşak
doku aralığı" diyor — kemiğe ve akciğere de uygulandığı hiçbir yerde yazmıyor.

### B5 · Dokümantasyon üç farklı şey söylüyor

| Kaynak | Resegmentation iddiası |
|---|---|
| Modül docstring `:23-24` | "MIRP is given no intensity range" |
| README | `[-1100, 1100]` |
| **Kod (ölçülen)** | **`['range']`, (−2000, 31743)** |

Bir IBSI raporlama listesi bu üç kaynaktan doldurulamaz. Etki olarak hiçbir
voksel atılmıyor (aralık int16 kodlamasının tamamını kapsıyor), ama mekanizma
beyanı yanlış.

**Not:** −2000 çapası **savunulabilir**. Gerçek ROI minimumu −1028 HU, yani
MIRP'in CT varsayılanı −1000'in altında; fixed-bin-size çapanın altındaki her
şeyi bin 1'e sıkıştırdığı için varsayılan çapa gerçek veriyi bozardı. Sorun
tercihte değil, beyanda.

### B6 · Tekrarlanabilirlik boşlukları

- **Sürüm sabitlemesi yok** — `pcct-radiomics-mirp/` içinde `requirements.txt` yok.
- **Checkpoint parametreleri damgalamıyor** — `bin_width` veya çapa değişirse eski
  satırlar sessizce yeniden kullanılıyor. Çapa değişimi 191/218 feature'ı
  değiştirdiği için karışan satırlar kıyaslanamaz.
- **Moran/Geary dosya yoluna bağlı** — MIRP 1000+ vokselde Monte-Carlo kullanıyor
  ve tohumu yolu içeren bir hash'ten alıyor. Doğrulandı: aynı voksellerden farklı
  dizinde farklı değer çıkıyor. pcct hattı bu ikisini atıyor; **phantomxpro
  atmıyor** ve `apply_naming.py` klasörleri yeniden adlandırıyor.
- **Provenance yok** — çıktıda mirp sürümü, ayar hash'i, girdi checksum'ı yok.
- **`warnings.filterwarnings("ignore")`** MIRP'in discretisation uyarılarını da susturuyor.
- **Exporter başlıkları** — her dilim ayrı rastgele Study/Series UID alıyor,
  `datetime.now()` damgalanıyor. Yan etki: standart bir DICOM okuyucu patch'leri
  3B hacim olarak toplayamaz.
- **Patch'lerde geometri yok** — `PixelSpacing`, `SliceThickness`,
  `ImagePositionPatient` hiçbiri yok; `SliceLocation` dizi indeksi. Spacing her
  zaman fallback'ten geliyor. Bugün doğru sonuç veriyor, ama kodun ilan ettiği
  "asla varsayılan kullanma" garantisi bu yolda fiilen çalışmıyor.
- **Kırpma kutusu sınırlanmıyor** — görüntü dışına taşan pikseller sessizce
  sentinel oluyor; uyarı yok, `accepted=False` yok. Bu veri setinde tetiklenmiyor.

---

## Test paketi

123 test geçiyor, **4 tanesi kasıtlı olarak başarısız** — bunlar düzeltilmesi
gerekenin şartnamesi.

| Konum | Geçen | Kasıtlı başarısız |
|---|---|---|
| `feature-extraction/pcct-radiomics-mirp/tests/` | 50 (+1 skip) | 2 |
| `feature-extraction/phantomxpro-mirp/` | 57 (mevcut 42 dahil) | 1 |
| `labeling/.../pcct_sep_error_corrected/tests/` | 16 | 1 |

### Kasıtlı başarısızlıklar

1. `test_requirements_file_exists_and_pins_numerics` — sürüm sabitlemesi yok.
2. `test_checkpoint_is_rejected_when_the_discretisation_anchor_changes` —
   checkpoint parametreleri damgalamıyor.
3. `test_done_sidecar_records_extraction_provenance` — provenance yazılmıyor.
4. `test_out_of_bounds_crop_is_reported_instead_of_silently_padded` — sınır dışı
   kırpma sessizce padding yapıyor.

### Çalıştırma

```bash
python -m pytest "feature-extraction/pcct-radiomics-mirp" -q
```

```bash
python -m pytest "feature-extraction/phantomxpro-mirp" -q
```

```bash
python -m pytest "labeling/pcct-sep-error-corrected/pcct_sep_error_corrected" -q
```

Hızlı alt küme için `-m "not slow"`. Gerçek veri klasörleri yoksa ilgili testler
otomatik atlanıyor.

---

## Yayında beyan edilmesi gereken metin

Mevcut README'lerin hiçbiri eksiksiz değil. `pcct-radiomics-mirp` için doğru
liste:

> Radiomics features were extracted with MIRP 2.5.0 (IBSI-1 compliant feature
> definitions). Fixed bin size discretisation, bin width 25 HU, grey-level origin
> −2000 HU (the int16 encoding floor). Resegmentation range (−2000, 31743) HU —
> no voxel is excluded; the range serves as the discretisation anchor.
> **No interpolation was performed**; voxels remained anisotropic at
> 0.586 × 0.586 × 1.0 mm (FOV 300) to 0.977 × 0.977 × 1.0 mm (FOV 500).
> ROIs were cropped to the largest padding-free square, retaining 43–80 % of the
> segmented voxels depending on ROI size and FOV; **the analysis mask is
> therefore a rectangular cuboid, and morphological features describe the crop
> rather than the object**. Texture aggregation: 3D averaged and 3D volume-merged
> (2D/2.5D variants reported separately). No IBSI-2 filter-based features.
> 12 features were excluded as non-reproducible (Moran's I, Geary's C, IVH volume
> fractions, local intensity peaks).

`phantomxpro-mirp` için ek olarak: 1 mm izotropik yeniden örnekleme,
tricubic (spline order 3), **Gaussian anti-aliasing pre-filter (β = 0.98)**,
resegmentation (−1000, 400) HU **applied to all 117 structures including bone
and lung**, 3D volume-merged aggregation, TotalSegmentator 2.12.0 masks.

---

## Önerilen sıra

1. **`anti_aliasing: False`** ekleyin (B1) — tek satır, en büyük etki, MIRP'in
   kendi IBSI yoluyla hizalar.
2. **Sürüm sabitleyin** (B6) — `requirements.txt`, kardeş projeden kopyalayın.
3. **Checkpoint'e parametre hash'i** yazın (B6) — sessiz karışmayı önler.
4. **Dokümantasyonu koda göre düzeltin** (B5) — üç çelişkili kaynak tek doğruya iner.
5. B2 ve B3 birer **karar**: düzeltirseniz önceki tüm çıktılar geçersiz olur.
   Düzeltmeyecekseniz yukarıdaki beyan metnini kullanın — böylece uyum iddiası
   doğrulanabilir hale gelir.
