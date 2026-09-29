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

## 4. Oluşturulan Yazılım Bileşenleri

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

## 5. Registration Yaklaşımı

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
    → stabil kemik centroid translation fallback

Fallback de başarısız
    → identity/world-space kullan
```

Transform kabulü spleen/aorta Dice’ına göre yapılmadı. Bağımsız stabil yapı

- Vertebrae
- Ribs
- Sternum

Bu seçim, organ maskelerini doğrudan optimize edip sonra aynı maskelerle başarı

## 6. Registration Sonuçları

Final registration batch:

```text
PhantomX/REGISTRATION/batch_to_20260805_v5/
```

Toplam pair:

88
```

Final batch sonuç özeti:

- Spleen ortalama Dice: `0.2566 → 0.9159`
- Aorta ortalama Dice: `0.2604 → 0.8958`
- Spleen için sıfır Dice pair: `0`
- Aorta için sıfır Dice pair: `0`
- Centroid translation fallback kullanılan pair: `79`
- Identity kullanılan pair: `9`
- Registration exception: `0`

En problemli örneklerde fallback etkisi:

```text
20260304 take1
Spleen Dice: 0.000 → 0.934
Aorta Dice:  0.000 → 0.927

20260526 take5
Spleen Dice: 0.000 → 0.872
Aorta Dice:  0.000 → 0.872

20260327 take1
Spleen Dice: 0.000 → 0.837
Aorta Dice:  0.000 → 0.887
```

Her pair için kullanılan yöntem ve QC:

```text
registration_qc.json
metrics.csv
```

## 7. Radyomik Feature Sonuçları

Registered feature çıktıları:

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

İlk sonuçlar:

3D feature’lar:

- Toplam karşılaştırma: 434 feature/organ kombinasyonu
- CV iyileşen: 50
- Pairwise fark iyileşen: 56
- İkisi birden iyileşen: 41

2D feature’lar:

- Toplam karşılaştırma: 706 feature/organ kombinasyonu
- CV iyileşen: 52
- Pairwise fark iyileşen: 55
- İkisi birden iyileşen: 43

FDR düzeltilmiş Wilcoxon sonuçları:

3D:

- FDR-significant test: 501
- İyileşme yönünde: 7
- Kötüleşen/değişmeyen: 494

2D:

- FDR-significant test: 946
- İyileşme yönünde: 6
- Kötüleşen/değişmeyen: 940

## 8. Kritik Kontrol: Identity World-Space Baseline

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

Bu kontrol gereklidir; çünkü registration sonrası CT görüntüsü fixed grid’e

## 9. Şu Anki Bilimsel Yorum

Şu ana kadarki sonuçlar iki farklı bulgu gösteriyor:

1. Rigid/translation registration organ geometrisini büyük ölçüde düzeltiyor.
2. Geometrik iyileşme, tüm radyomik feature’ların tekrarlanabilirliğinin
   iyileşeceği anlamına gelmiyor.

Bazı feature’lar registration sonrası daha kararlı hale gelirken bazı texture
feature’ları resampling nedeniyle daha fazla değişmektedir.

Bu nedenle güvenli sonuç cümlesi şudur:

> Registration-based spatial standardization substantially improves geometric
> agreement between repeated organ masks, but its effect on radiomic
> reproducibility is feature-dependent and can be confounded by interpolation
> introduced during resampling.

## 10. Henüz Yapılmayanlar

- World-space control feature extraction’ın 20260402 pair’leri için maske
  bulunamayan durumlarının ayrı exclusion raporu
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

## 11. Sonraki Analiz Sırası

1. World-space ve registered feature tablolarını feature ailesine göre özetle.
2. Sadece registration etkisini ölçmek için `B → C` karşılaştırmasını kullan.
3. FDR-significant feature’ları improved/worsened olarak ayır.
4. Spleen ve aorta sonuçlarını ayrı raporla.
5. Same-session ve cross-session sonuçlarını ayrı tablolarda göster.
6. Companion organ-coordinate yöntemiyle ICC, CV ve pairwise farkları karşılaştır.
7. Son kararı feature bazında ver; registration’ın tüm feature’lar için tek tip
   faydalı veya zararlı olduğunu varsayma.
