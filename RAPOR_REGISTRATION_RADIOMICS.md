# Registration-Based Spatial Standardization of PhantomX Radiomics

## 1. Proje Amacı

Bu çalışmanın amacı, longitudinal PhantomX CT görüntülerinde organların farklı
konumlarda bulunmasından kaynaklanan radyomik değişkenliği incelemektir.

Temel karşılaştırma:

```text
Registration-free analysis
vs
Registration-based spatial standardization
```

İncelenen hedef organlar:

- Spleen: TotalSegmentator label `1`
- Aorta: TotalSegmentator label `52`

## 2. Veri Seti

README ve mevcut veri yapısına göre:

- Siemens NAEOTOM Alpha PCCT
- Q3 / Qr40f
- Gerçek tüp voltajı: 120 kV
- 100 mAs
- 300 mm FOV
- 1 mm kesit kalınlığı
- 22 PhantomX oturumu
- 89 seri
- Oturum başına 3–10 tekrar

Mevcut veri kaynakları:

```text
PhantomX/LABELED/
    nifti/<series>/ct.nii.gz
    segmentation/<series>/seg.nii.gz
    features/<series>/features.csv
```

TotalSegmentator maskeleri çok etiketlidir. Aynı maskede organların yanı sıra
vertebra, kaburga ve sternum gibi registration QC için kullanılan stabil yapılar
da bulunmaktadır.

## 3. Git ve Güvenlik

Proje kökünde Git repository oluşturuldu. Ham veri ve üretilmiş büyük dosyalar

- DICOM
- NIfTI
- Segmentation
- CSV/Excel sonuçları
- Registration çıktıları

Önemli checkpoint commit’leri:

```text
cfcf60d  Project baseline
27494c5  Initial rigid registration pilot
8d08b5d  Registration monitoring GUI
356bc33  Resumable registration batch runner
f739434  Centroid translation fallback
5ef80b8  Registered MIRP feature extraction
9ef907c  Session-aware repeatability analysis
c2c34a6  FDR significance tests
8682a8e  Identity world-space control analysis
```

Kaynak CT, maskeler ve baseline feature tabloları değiştirilmedi.

## 4. Kullanılan Teknikler

### 4.1 DICOM Fiziksel Koordinatları

CT görüntülerindeki voxel index’leri tek başına fiziksel konumu ifade etmez.
Her görüntünün fiziksel konumu; DICOM metadata içindeki origin, spacing ve
direction bilgileriyle tanımlanır.

Bu projede karşılaştırmalar mümkün olduğunca fiziksel/world-space üzerinden
yapıldı. Böylece aynı voxel index’inde olmayan fakat gerçek fiziksel konumu
bilinen organlar doğru koordinat sisteminde değerlendirilebildi.

### 4.2 NIfTI

DICOM serileri 3D `ct.nii.gz` dosyalarına dönüştürülmüş durumda. NIfTI dosyaları
CT yoğunluklarını ve DICOM’dan taşınan fiziksel geometri bilgilerini içerir.

Registration ve radyomik hesaplar NIfTI dosyaları üzerinden yapıldı; ham DICOM
dosyaları değiştirilmedi.

### 4.3 Multi-label Segmentation

TotalSegmentator tek bir organ maskesi yerine çok etiketli bir segmentation
üretir. Her integer label bir anatomik yapıyı temsil eder:

```text
1  = spleen
52 = aorta
```

Registration sırasında stabil yapılar için vertebra, kaburga ve sternum
etiketlerinin birleşiminden bağımsız bir QC maskesi oluşturuldu.

### 4.4 Rigid Registration

Rigid registration görüntüyü yalnızca altı serbestlik derecesiyle dönüştürür:

- X, Y, Z yönlerinde translation
- X, Y, Z eksenleri etrafında rotation

Organın şeklini değiştirmez. Bu nedenle phantom repositioning ve table
position gibi global konum farklarını düzeltmek için uygundur; segmentation
şekil farklarını veya gerçek deformasyonu çözemez.

CT registration için SimpleITK kullanıldı. Büyük konum farklarında registration
2 mm geçici çözünürlükte çalıştırıldı; final CT ve maskeler fixed görüntünün
orijinal grid’ine taşındı.

### 4.5 Correlation Metric

Aynı CT protokolündeki görüntüler için correlation similarity metric kullanıldı.
Registration metric örneklemesi deterministic `REGULAR` olarak ayarlandı; bu
aynı girdide rastgele farklı transformlar oluşmasını önler.

### 4.6 Centroid Translation Fallback

Bazı uzak oturumlarda CT görüntüleri başlangıçta yeterince örtüşmediği için
intensity-based rigid optimizer geçerli örnek bulamadı. Bu durumda stabil
vertebra/kaburga/sternum maskelerinin fiziksel centroid farkından doğrudan
translation üretildi.

Bu fallback, hedef organ maskelerini optimize etmek için değil, bağımsız stabil
anatomik yapıların fiziksel konumunu eşitlemek için kullanıldı.

### 4.7 Resampling ve Interpolation

Görüntü resampling işlemleri için:

- CT: linear interpolation
- Label mask: nearest-neighbor interpolation

Maskelerde nearest-neighbor zorunludur; aksi halde label değerleri arasında
yapay fractional organ etiketleri oluşur.

Radyomik hesaplarda MIRP ayrıca 1 mm isotropic resampling yapar. Bu nedenle
registration’ın CT görüntüsüne ek interpolation etkisi oluşturabileceği ayrıca
identity/world-space kontrolüyle incelendi.

### 4.8 Radiomics

MIRP 2.5.0 ve IBSI uyumlu sabit ayarlar kullanıldı:

- 1 mm isotropic spacing
- Tricubic interpolation
- Anti-aliasing kapalı
- Fixed bin width: 25 HU
- Intensity range: `[-1300, 3100]`
- Ayrı 2D ve 3D feature hesapları

## 5. Dice Similarity Coefficient Nedir?

Dice iki binary maskenin ne kadar aynı voxel bölgesini paylaştığını ölçer:

```text
Dice = 2 × |A ∩ B| / (|A| + |B|)
```

Burada:

- `A`: fixed/reference organ maskesi
- `B`: moving veya registered organ maskesi
- `A ∩ B`: iki maskenin ortak kesişimi

Dice aralığı:

```text
0.0 = hiç overlap yok
1.0 = maskeler tamamen aynı
```

Örneğin:

```text
Dice = 0.90
```

iki organ maskesinin voxel hacimlerinin büyük bölümünün üst üste geldiğini
gösterir.

### 5.1 Dice Bu Projede Ne İfade Ediyor?

Bu projede Dice, registration sonrasında longitudinal tekrarların aynı fiziksel
organ bölgesini ne kadar iyi temsil ettiğini ölçer.

Örnek:

```text
Spleen Dice: 0.26 → 0.92
```

Bu, moving spleen maskesinin reference spleen maskesiyle spatial overlap’ının
registration sonrasında belirgin biçimde arttığını gösterir.

### 5.2 Dice Ne İfade Etmez?

Dice tek başına şunları kanıtlamaz:

- TotalSegmentator segmentasyonunun anatomik olarak doğru olduğunu
- Registration transformunun fiziksel olarak kesin doğru olduğunu
- Radyomik feature’ların daha tekrarlanabilir hale geldiğini
- Organ sınırlarının gerçek ground truth ile aynı olduğunu

Bu nedenle Dice tek başına kullanılmadı. Yanına şu metrikler eklendi:

- Centroid displacement: organ merkezleri arasındaki fiziksel mesafe, mm
- HD95: yüzeyler arasındaki 95. percentile mesafe, mm
- ASSD: ortalama simetrik yüzey mesafesi, mm
- Stable-structure QC: vertebra/kaburga/sternum hizası

### 5.3 Dice’ı Etkileyen Faktörler

Dice aşağıdaki durumlardan etkilenebilir:

- Phantom’ın farklı konumlandırılması
- CT FOV kapsamı
- Organın gerçekten farklı fiziksel konumda olması
- TotalSegmentator’ın tekrarlar arasında farklı sınır çizmesi
- Organın bir kısmının görüntü dışında kalması
- Rigid registration’ın organ deformasyonlarını çözememesi

Bu nedenle düşük Dice her zaman registration hatası anlamına gelmez; segmentation
veya FOV problemi de olabilir.

### 5.4 Dice Eşiği ve Gözlenen Segmentation Tavanı

Aynı oturumdaki farklı çekimlerin TotalSegmentator maskeleri pairwise olarak
karşılaştırıldı. Gözlenen en yüksek Dice:

- Spleen: `0.9962`
- Aorta: `0.9892`

Bu veri ve mevcut otomatik segmentation için her iki organda zorunlu
`Dice >= 0.99` kriteri gerçekçi değildir; aorta aynı oturum içindeki maskeler
arasında dahi bu değere ulaşmamıştır. Bu nedenle 0.99 ulaşılması gereken bir
optimizasyon hedefi olarak kullanılmamalı; raporlanacak ideal bir eşik olarak
belirtilmelidir. Transform kabulü bağımsız stabil anatomik yapı QC’sine dayanır.

Tavan analizi dosyası: `PhantomX/REGISTRATION/segmentation_ceiling.csv`.

## 6. Oluşturulan Yazılım Bileşenleri

### Registration

`Kod/registration-radiomics/registration.py`

- SimpleITK rigid registration
- Correlation veya mutual information metric desteği
- Deterministic regular sampling
- 2 mm registration-only resampling
- Nearest-neighbor mask resampling
- Centroid tabanlı başlangıç transformu
- Translation transform desteği

### Pilot

`Kod/registration-radiomics/pilot.py`

Bir fixed/moving pair için:

1. CT ve maskeleri okur.
2. DICOM physical/world-space baseline oluşturur.
3. Küçük displacement varsa rigid registration’ı atlar.
4. Büyük displacement varsa rigid registration dener.
5. Rigid registration başarısızsa stabil mask centroid translation fallback kullanır.
6. Stabil vertebra/kaburga/sternum QC’si yapar.
7. Kötü transformu reddeder ve identity sonucu kullanır.
8. Dice, centroid displacement, HD95 ve ASSD hesaplar.

### Batch

`Kod/registration-radiomics/batch.py`

- 88 pair’i sırayla veya worker’larla çalıştırır.
- Resume destekler.
- Aynı pair’in iki kez çalışmasını lock dosyasıyla önler.
- Worker sayısı ayarlanabilir.
- Batch manifest oluşturur.

### GUI

`Kod/registration-radiomics/gui.py`

- Dataset root seçimi
- `take1` / `take2` yollarını otomatik bulma
- Canlı işlem log’u
- Output klasörü seçimi
- Registration başlatma/durdurma

### Baseline

`Kod/registration-radiomics/baseline.py`

Registration-free fiziksel koordinat baseline’ını üretir.

### Feature Extraction

`Kod/registration-radiomics/registered_features.py`

Registered CT/mask çiftlerinden MIRP 2D ve 3D feature çıkarır.

Kullanılan temel MIRP ayarları:

- MIRP 2.5.0
- 1 mm isotropic resampling
- Tricubic interpolation
- Anti-aliasing kapalı
- Fixed bin size: 25 HU
- Resegmentation range: `[-1300, 3100]`
- Aynı 2D ve 3D texture aggregation ayarları

### İstatistik ve Karşılaştırma

- `compare_features.py`
- `repeatability_analysis.py`
- `significance_analysis.py`
- `worldspace_baseline.py`
- `compare_registered_worldspace.py`

## 7. Registration Yaklaşımı

Her pair için tek bir global transform kullanılması amaçlandı. Organ başına
ayrı transform kullanılmadı; aksi durumda organlar aynı ortak referans uzayına
taşınmış olmazdı.

Registration akışı:

```text
Moving CT
   ↓
Rigid registration veya centroid translation
   ↓
Fixed CT grid’ine resampling
   ↓
Moving maskenin nearest-neighbor resampling’i
   ↓
Geometrik QC
```

Registration seçimi:

```text
Target organ centroid displacement <= 5 mm
    → identity/world-space kullan

Target organ centroid displacement > 5 mm
    → rigid registration dene

Rigid başarısız veya stabil QC kötü
    → landmark rigid (Kabsch) fallback
    → stabil kemik centroid translation fallback

Fallback de başarısız
    → identity/world-space kullan
```

Transform kabulü spleen/aorta Dice’ına göre yapılmadı. Bağımsız stabil yapı
maskesi kullanıldı:

- Vertebrae
- Ribs
- Sternum

Rigid/landmark/translation adaylarının kabulü için stabil mask Dice’ı artmalı;
centroid displacement, HD95 ve ASSD de belirlenen toleranslar içinde kalmalıdır.
Bu, organ maskeleri üzerinde doğrudan optimizasyon yapıp yine aynı maskelerle
başarı ölçme riskini azaltır.

## 8. Registration Sonuçları

Final registration batch (V6):

```text
PhantomX/REGISTRATION/batch_to_20260805_v6/
```

Her pair için PNG overlay görselleri:

```text
PhantomX/REGISTRATION/batch_to_20260805_v6/visual_qc/
```

GUI’de `Visual QC viewer` ve `3D volume viewer` bu V6 sonuçlarını gösterir.

Toplam pair: `88`

V6 final batch sonuç özeti:

- Spleen ortalama Dice: `0.2566 → 0.9551`
- Aorta ortalama Dice: `0.2604 → 0.9229`
- Spleen için sıfır Dice pair: `0`
- Aorta için sıfır Dice pair: `0`
- Landmark rigid (Kabsch) fallback kullanılan pair: `78`
- Identity kullanılan pair: `9`
- Rigid registration kullanılan pair: `1`
- Sıfır target Dice kalan pair: `0`

V6’da rigid/landmark/translation transformu, stabil yapı QC’sinde Dice artışına
ek olarak centroid displacement, HD95 ve ASSD’nin belirlenen küçük toleranslarda
kalmış olmasını gerektirir. Bu daha sıkı kural, V6’daki en kötü pair
(`20260211 take3`) için rigid transformu reddedip landmark rigid fallback’i
kullandı.

En problemli örneklerde fallback etkisi:

```text
20260304 take1
Spleen Dice: 0.000 → 0.964
Aorta Dice:  0.000 → 0.925

20260526 take5
Spleen Dice: 0.000 → 0.965
Aorta Dice:  0.000 → 0.923

20260327 take1
Spleen Dice: 0.000 → 0.964
Aorta Dice:  0.000 → 0.932
```

Landmark fallback, aynı label ile eşleşen stabil yapılardaki fiziksel centroid
noktalarını kullanarak rigid rotation+translation kestirir (Kabsch/SVD). V6’da
QC kuralı sıkılaştırıldı: stabil yapı Dice’ı artmalı; centroid displacement,
HD95 ve ASSD de küçük belirlenmiş toleransları aşmamalıdır. Bu nedenle
`20260211 take3` için önceki optimizer rigid sonucu reddedilip landmark rigid
fallback ile yeniden hesaplandı.

V6 için her pair’in tüm 2D kesit görselleri:

```text
PhantomX/REGISTRATION/batch_to_20260805_v6/visual_qc/
```

Her pair için kullanılan yöntem ve QC:

```text
registration_qc.json
metrics.csv
```

## 9. Radyomik Feature Sonuçları

V6 registered feature çıktıları:

```text
all_features_registered_3d.csv
all_features_registered_2d.csv
```

Baseline ve registered feature karşılaştırması:

```text
analysis/feature_comparison_3d.csv
analysis/feature_comparison_2d.csv
```

Session-aware tekrar edilebilirlik tabloları:

```text
analysis/repeatability_summary_3d.csv
analysis/repeatability_summary_2d.csv
analysis/repeatability_comparison_3d.csv
analysis/repeatability_comparison_2d.csv
```

V6 session-aware repeatability sonuçları (88 pair, 22 oturum):

3D feature’lar:

- Toplam karşılaştırma: 434 feature/organ kombinasyonu
- CV iyileşen: 107
- Pairwise fark iyileşen: 94
- İkisi birden iyileşen: 73

2D feature’lar:

- Toplam karşılaştırma: 706 feature/organ kombinasyonu
- CV iyileşen: 129
- Pairwise fark iyileşen: 127
- İkisi birden iyileşen: 96

FDR-significant tekrar edilebilirlik testleri içinde:

- 3D: 58 anlamlı test; 11 iyileşme, 47 kötüleşme yönünde
- 2D: 211 anlamlı test; 18 iyileşme, 193 kötüleşme yönünde

## 10. Kritik Kontrol: Identity World-Space Baseline

Registration’ın etkisini interpolation etkisinden ayırmak için üçüncü analiz

```text
A: Original baseline
B: Identity/world-space resampled
C: Registered
```

Yorum:

```text
A → B: yalnızca resampling/interpolation etkisi
B → C: gerçek registration etkisi
A → C: toplam yöntem etkisi
```

World-space kontrol feature’ları:

```text
all_features_worldspace_3d.csv
all_features_worldspace_2d.csv
```

World-space ve registered doğrudan karşılaştırması:

```text
analysis/worldspace_vs_registered_3d.csv
analysis/worldspace_vs_registered_2d.csv
```

World-space kontrol feature’ları V5’teki identity-resampled CT/maskelerden
alındı; bu görüntüler registration’dan bağımsızdır ve V6 referansı/ham girdileri
aynı olduğundan V6 registered tablolarıyla eşleştirildi. Bu kontrol için 85 pair
ortak feature üretti; 20260402 oturumundaki üç pair’de identity fixed-grid maskesi
spleen/aorta ROI’si içermediği için MIRP feature’ı üretilemedi.

World-space→V6 registered karşılaştırmasında median relative feature farkı 3D’de
`0.107`, 2D’de `0.0769` oldu. Bu değerler feature ölçeklerinden etkilendiği için
tek başına başarı metriği değildir; feature ailesi ve organ bazında
yorumlanmalıdır.

Bu kontrol gereklidir; çünkü registration sonrası CT görüntüsü fixed grid’e
yeniden örneklenir ve interpolation texture feature’larını değiştirebilir.

## 11. Şu Anki Bilimsel Yorum

Şu ana kadarki sonuçlar iki farklı bulgu gösteriyor:

1. Rigid/translation registration organ geometrisini büyük ölçüde düzeltiyor.
2. Geometrik iyileşme, tüm radyomik feature’ların tekrarlanabilirliğinin
   iyileşeceği anlamına gelmiyor.

Bazı feature’lar registration sonrası daha kararlı hale gelirken bazı texture
feature’ları resampling nedeniyle daha fazla değişmektedir.

V6’da geometrik iyileşme belirgin olmakla birlikte, feature reproducibility
sonuçları feature’a göre değişmektedir. World-space ile registered feature’ların
medyan relative farkı 3D’de yaklaşık `0.107`, 2D’de `0.077` bulundu. Bu fark
registration etkisi ile resampling/interpolation etkisinin birlikte bulunduğunu
gösterir; iki etki B→C kontrolüyle ayrıca raporlanmalıdır.

Bu nedenle güvenli sonuç cümlesi şudur:

> Registration-based spatial standardization substantially improves geometric
> agreement between repeated organ masks, but its effect on radiomic
> reproducibility is feature-dependent and can be confounded by interpolation
> introduced during resampling.

## 12. Henüz Yapılmayanlar

- World-space kontrolünde ROI içermeyen üç 20260402 pair’i için exclusion raporu
- Feature ailelerine göre sonuçların gruplanması:
  - Shape
  - First-order
  - GLCM
  - GLRLM
  - GLSZM
  - NGTDM
  - NGLDM
- Near-zero ve numerik olarak anlamsız feature’ların filtrelenmesi
- Bland–Altman grafiklerinin oluşturulması
- Final tablo ve figure üretimi
- Companion organ-coordinate yöntemiyle aynı ölçütlerde karşılaştırma

## 13. Sonraki Analiz Sırası

1. `B → C` farklarını feature ailesi ve organ bazında özetle.
2. FDR sonuçlarını near-zero ve numerik açıdan kararsız feature’ları işaretleyerek
   yeniden sun.
3. Spleen ve aorta; aynı oturum ve farklı oturum sonuçlarını ayrı raporla.
4. Companion organ-coordinate yöntemiyle aynı metriklerde karşılaştır.
5. Registration’ın hangi feature’larda faydalı olduğunu feature bazında belirle;
   tüm feature’lar için tek tip fayda varsayma.
