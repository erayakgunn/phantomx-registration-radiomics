# Registration Algoritması — Adımlar ve Farklı Durumlarda Davranış

**Hazırlayan:** Eray Akgün  
**Kapsam:** PhantomX longitudinal PCCT/SEP + Zurich test-retest  
**Site:** https://erayakgunn.github.io/registration-overlay/

---

## 1. Kısa özet

Çakıştırma, hedef organı doğrudan kullanmadan, **stabil kemik yapıları** (vertebra, kaburga, sternum) referans alarak yapılır. Amaç, hareketli seriyi fixed referansın fiziksel uzayına taşımaktır.

İki kavramsal aşama vardır:
1. **Öteleme (rotasyonsuz)**: kemik centroid'lerinden elde edilen uzaysal kaydırma; aynı zamanda rigid adayların **başlangıç (initialization)** transformudur.
2. **Rigid (rotasyon + öteleme)**: correlation rigid veya landmark rigid (Kabsch/SVD) adayları.

**Önemli netleştirme:** Nihai transform **tek bir rigid** (veya yalnız öteleme / identity) transformdur. Öteleme ile rigid **üst üste uygulanmaz**; öteleme, rigid optimizasyonunun başlangıcıdır. Sitede “centroid” ve “rigid” ayrı **görsel durumlar** olarak gösterilir; algoritma bunlardan **birini** seçer.

---

## 2. Girdi ve maskeler

- **CT** (fixed ve moving) + **multi-label segmentasyon** (`seg.nii.gz`).
- **Hedef organlar:** spleen (120 kVp'de label 1), aorta (label 52); 140 kVp'de spleen 3, aorta 6.
- **Stabil kemik maskesi:** vertebra 27–43, kaburga 92–116, sternum 116.
  - Kemik maskesi **registration ve QC referansıdır**.
  - **Hedef organ maskeleri transform seçiminde kullanılmaz** (sızıntı/circularity yok).
- Fiziksel/world koordinatlar kullanılır; maske nearest-neighbor, CT lineer (veya bspline/windowed-sinc) ile fixed grid'e taşınır.

---

## 3. Algoritma adımları

```
Girdi: fixed CT/seg, moving CT/seg, hedef organlar, kemik label'ları

1) Hedef organ centroid farkı hesaplanır: d = max(organ centroid displacement)
2) Eğer d <= 2 mm:
      transform = identity
      moving mask/CT fixed grid'e identity ile taşınır
      registration_method = "identity"
   Değilse:
   3) Kemik centroid'lerinden öteleme (rotasyonsuz) hesaplanır  ← başlangıç
   4) Aday transformlar üretilir:
      a) Correlation rigid (CT, kemik maskesi içinde, 6-DOF; başlangıç = adım 3)
      b) Landmark rigid — vertebra seti (Kabsch/SVD)
      c) Landmark rigid — tüm kemik seti (fallback)
   5) Her aday için kemik maskesi üzerinde QC:
      - stable Dice > önceki stable Dice
      - centroid farkı  <= önceki + 0.5 mm
      - HD95            <= önceki + 1.0 mm
      - ASSD            <= önceki + 0.25 mm
   6) QC'yi geçen adaylar arasından seçim:
      - Hold-out TRE (kaburga + sternum; vertebra fit'inde kullanılmadı)
      - En düşük TRE kazanır; eşitlikte yüksek stable Dice
      - "Tüm kemik" fallback'in temiz hold-out'u yok → TRE = ∞
   7) Hiçbir aday QC'yi geçmezse:
      - Öteleme-only (centroid translation) fallback
      - O da geçmezse identity
   8) Seçilen TEK transform CT ve maskeye bir kez uygulanır
      - CT: lineer/bspline; maske: nearest-neighbor
   9) Fixed grid'de hedef organ metrikleri: Dice, centroid, HD95, ASSD
```

**Sözel olarak:** İki tomografide de aynı fantom var ama masa/pozisyon farklı. Önce hedef organlar zaten 2 mm’den yakın mı diye bakıyoruz; yakınsa hiç dokunmuyoruz. Değilse kemiklerin ağırlık merkezlerini üst üste getirip kabaca kaydırıyoruz. Ardından iki farklı yolla döndürme+kaydırma arıyoruz: gri değerleri eşleştiren correlation ve kemik merkez noktalarını eşleştiren landmark. Her adayı kemikler üzerinde sınıyor; hiç kullanmadığımız kaburga/göğüs kemiğiyle ayrıca test edip en iyisini seçiyoruz. Hiçbiri geçmezse sadece kaydırmayı, o da olmazsa hiç dokunmamayı seçiyoruz.

Her pair için `registration_qc.json` şunları kaydeder: `stable_before/after`, `registration_attempted`, `registration_accepted`, `registration_method`, `target_max_centroid_displacement_mm`, `holdout_tre_mm`.

---

## 4. “İki aşama” ifadesinin netleştirilmesi

Hocanın ifadesindeki “önce rotasyonsuz kaydırma, sonra rigid ile dönme” **görsel/kavramsal** olarak doğrudur:

| Site durumu | Anlamı |
|---|---|
| **raw** | Hiç transform yok (identity); ham moving görüntü |
| **centroid** | Yalnız öteleme (rotasyonsuz) baseline |
| **rigid** | Nihai rigid (öteleme + rotasyon) |

**Sözel olarak:** Hocanın “önce kaydır, sonra döndür” tarifi, sitede gördüğün ham → centroid → rigid sırasını anlatır. Ama bilgisayar önce kaydırıp sonra ayrıca bir döndürme eklemiyor; kaydırma, döndürmeli çözümün başlangıç noktası. Sonuçta tek bir “kaydır+döndür” uygulanıyor; sitedeki üç sütun bu tek işlemin aşamalarını gözle görebilmen için.

Ancak **algoritmik olarak**:
- Öteleme, rigid'in **başlangıç** transformudur.
- Landmark rigid, correlation rigid'in üstüne **eklenmez**; ayrı bir **adaydır**.
- Nihai transform **tek** bir transformdur; “önce ötele, sonra döndür” şeklinde iki ayrı uygulama yoktur.

---

## 5. Kemik referansı ve sızıntısızlık

- **Kemik** kullanılır çünkü:
  - Fantomda stabil, yüksek kontrastlı ve organlardan bağımsız.
  - Organ segmentasyon hatası transform seçimine taşınmaz.
- QC ve seçim **kemik maskesi** üzerinden yapılır.
- **Hedef organ Dice'ı transform seçiminde kullanılmaz**; yalnızca sonuç metriği olarak raporlanır.
- Hold-out TRE (kaburga + sternum) vertebra fit'ine girmediği için **bağımsız bir doğrulama** sağlar.

**Sözel olarak:** Kemikleri referans alıyoruz çünkü fantomda sabit, yüksek kontrastlı ve organlardan bağımsız. Organları hizalama işine karıştırsaydık, sonra onların Dice’ını ölçmek “kendi kendini doğrulama” olurdu. Hold-out kontrolü de bir sınav gibi: omurga ile hizaladık, hiç kullanmadığımız kaburga ve göğüs kemiğiyle sınadık.

---

## 6. Farklı durumlarda nasıl çalışır?

| Durum | Davranış | Neden |
|---|---|---|
| **Aynı oturum test-retest** (Zurich) | `identity`; registration atlanır | Organ centroid farkı ≤ 0.94 mm (eşik 2 mm) |
| **Farklı tarih** (longitudinal) | centroid → rigid/landmark | Büyük masa/pozisyon farkı; global hizalama gerekir |
| **140 kVp** | `identity` | Kemik label'ı yok; landmark için <3 nokta |
| **Düşük kemik Dice / FOV dışı** | QC geçmez → öteleme veya identity | Kemik referansı güvenilir değil |
| **3 mm kesit** | Aynı akış; Dice daha düşük | Partial volume/interpolasyon |
| **İnce aorta** | Rigid global düzeltir; kalan fark sınır | Tübüler yapı; segmentasyon/partial volume |
| **Cross-date büyük kayma** | Rigid/landmark; Dice 0.03→0.96 | Global hizalama |
| **Kötü segmentasyon** | Identity/fallback | Kemik referansı da segmentasyona bağlı |

**Sözel olarak:** Aynı oturumun tekrarları zaten hizalı olduğu için hiç dokunmuyoruz; farklı tarihlerde masa kaydığı için kaydırma+döndürme devreye giriyor. 140 kVp’de kemik etiketi olmadığı için döndürme yapamıyoruz; kemikler kötü segmentlenmişse veya görüş alanı dışındaysa kontrolü geçemeyip hiç dokunmuyoruz.

### Sayısal örnekler
- **Longitudinal PCCT:** Dice spleen 0.257→0.964, aorta 0.260→0.933.
- **Zurich test-retest:** skip5 (identity) ile before == after; zorla landmark uygulanınca Dice **düşüyor** (PCCT aorta -0.014, SEP aorta -0.014).

**2 mm eşiğinin gerekçesi:** SEP `20260128` çifti 4.2 mm (≈7.2 px) kaymada 5 mm eşiğiyle identity kalıyordu (spleen 0.834 / aorta 0.878). 2 mm (≈3.4 px) ile landmark devreye girdi ve **spleen 0.955 / aorta 0.917** oldu. 5 mm fazla gevşekti; 2 mm, 1–2 voksel gürültünün üstünde ama gerçek kaymayı kaçırmıyor.

---

## 7. Sitede ne gösteriliyor?

- **Dataset seçici:** SEP, PCCT, Zürich.
- **Galeri:** her kartta soldan sağa **ham (identity) → centroid → rigid**.
- **Detay:** 3'lü (Ham | Centroid | Rigid), ayrıca Slider / Yan yana.
- **Metrikler:** Dice, centroid farkı, HD95, ASSD; VOI analizi.
- **Analitik sayfa:** tüm-çift Dice matrisi, histogram, tarih kazancı.

Site **görselleştirme** amaçlıdır; “centroid” ve “rigid” durumları algoritmanın **seçtiği tek transformu** anlatmak için yan yana konur, üst üste uygulandığı anlamına gelmez.

---

## 8. Sınırlılıklar

1. **Rigid-only:** Deformable yok. Fantom rijit olduğu için bu bilinçli; lokal sınır farkı rigid ile düzeltilemez.
2. **Kemik maskesi segmentasyona bağlı:** Kemik segmentasyonu zayıfsa QC/seçim etkilenir.
3. **Hold-out TRE nokta-tabanlı:** Kemik centroid'leri üzerinden; yüzey tabanlı TRE değil.
4. **140 kVp:** Kemik etiketi olmadığı için rotasyon düzeltmesi yapılamıyor.
5. **Dice tavanı:** Segmentasyon/partial volume/rekon; registration değil. Aynı oturum tavanı spleen 0.977, aorta 0.950.
6. **Dice referansa bağlı:** take1 tek referans; kohort konsensüsü ayrı bir **sensitivity** analizi (rapor: `KONSENSUS_REFERANS_RAPORU.md`).

**Sözel olarak:** Rigid, sadece global konum ve yönü düzeltir; organın kendi sınırındaki 1 voksel oynamayı, partial volume’u, segmentasyon farkını düzeltemez. Bu yüzden Dice 0.99 değil ~0.96/0.93; bu registration’ın değil segmentasyonun tavanı. 0.99 için daha iyi segmentasyon veya daha iyi görüntüleme gerekir; rigid’i zorlamak sadece yeniden örnekleme ekler.

---

## 9. Hocaya bir paragraflık özet

> “Çakıştırma, hedef organı kullanmadan stabil kemik yapıları (vertebra, kaburga, sternum) üzerinden yapılıyor. Önce kemik centroid'lerinden rotasyonsuz öteleme hesaplanıyor; bu, rigid adayların başlangıcı oluyor. Ardından correlation rigid (CT, 6-DOF) ve landmark rigid (Kabsch/SVD; önce vertebra, sonra tüm kemik) adayları üretiliyor. Her aday kemik maskesi üzerinde Dice/centroid/HD95/ASSD toleranslarıyla QC ediliyor; QC'yi geçenler arasından **kaburga+sternum hold-out TRE**'si en düşük olan seçiliyor. Hiçbiri geçmezse öteleme-only, o da geçmezse identity uygulanıyor. Nihai transform **tek** bir rigid/öteleme/identity; öteleme ile rigid üst üste uygulanmıyor. Aynı oturum test-retest'te centroid farkı ≤1 mm olduğu için identity kalıyor; registration'ın gerçek katkısı farklı tarihli serilerde (Dice 0.03→0.96). Sitedeki ham/centroid/rigid sütunları bu üç durumu görselleştirir.”

---

## 10. Terimler: centroid, correlation, landmark, rigid

### 10.1 Rigid (katı dönüşüm)
- Matematik: `x' = R x + t`; `R` 3×3 rotasyon, `RᵀR = I`, `det R = +1`; 6 serbestlik (3 rotasyon + 3 öteleme).
- Şekli bozmaz: mesafe/açı korunur; ölçekleme (scale) ve shear yok.
- Kodda `sitk.Euler3DTransform` (3 Euler açısı + 3 öteleme).
- Karşıtları: translation-only (3-DOF), affine (12-DOF), deformable (çok sayıda serbestlik).

- **Sözel olarak:** Elindeki sert bir plastik fantomu masada kaydırıp döndürmek gibi. Şekli bozmadan sadece konumunu ve yönünü değiştiriyorsun; esnetmek, büyütmek yok.

### 10.2 Centroid (ağırlık merkezi)
- Maskenin fiziksel ağırlık merkezi (voxel koordinatlarının ortalaması, fiziksel uzaya çevrilir).
- **Centroid translation:** `t = c_moving − c_fixed`, `R = I`; yalnız öteleme (3-DOF).
- Bizde kullanılan centroid **kemik maskesinin** centroid'i (hedef organın değil) → rigid adayların **başlangıcı** ve QC başarısızsa **fallback**.
- Sitedeki **“centroid” sütunu** = bu kemik-centroid ötelemesi.
- Hedef organ centroid farkı yalnızca **skip kararı** için: ≤ 2 mm ise identity.

- **Sözel olarak:** İki çubuğun tam ortasını üst üste koymak gibi. Hareketli görüntüdeki kemiklerin ağırlık merkezini, sabit görüntüdeki kemiklerin ağırlık merkezine getiriyorsun; sadece kaydırma var, döndürme yok.

### 10.3 Correlation (normalized cross-correlation, NCC)
- İki CT'nin **gri değer benzerliği**; Dice ile ilgisi yok.
- Amaç: `max_T NCC(I_fixed, I_moving ∘ T)`, `T` rigid (6-DOF).
- Kemik maskesi içinde değerlendirilir; %20 regular sampling; linear interpolasyon; çok çözünürlük [4,2,1] + smoothing [2,1,0] mm; gradient descent (150 iterasyon).
- **Intensity tabanlı rigid** adayı; başlangıç = kemik centroid ötelemesi; çıktı = rigid transform.
- QC'de `registration_method` = `"rigid"` (transform kind) olarak görünür.

- **Sözel olarak:** İki yarı saydam fotoğrafı üst üste tutup, desenler birbirini tutana kadar oynatmak gibi. Bilgisayar her denemede “bu iki görüntü ne kadar benziyor?” diye puan verir; en yüksek puanı veren döndürme+kaydırmayı seçer. Sadece kemik bölgesine baktığı için organlardan etkilenmez.

### 10.4 Landmark (nokta tabanlı rigid)
- Karşılıklı noktalar: **kemik label centroid'leri** (her vertebra, kaburga, sternum).
- Amaç: `min_{R,t} Σ‖R qᵢ + t − pᵢ‖²`; **Kabsch/SVD** ile kapalı-form çözüm.
- Intensity kullanmaz → rekon/gürültü farklarına dayanıklı; az sayıda nokta varsa kaba.
- Adaylar: `landmark_rigid_vertebrae` (sadece vertebra seti) ve `landmark_rigid_fallback` (tüm kemik seti).

- **Sözel olarak:** Kemiklerin merkezlerine birer nokta koyup, aynı noktaları iki görüntüde eşleştirmek gibi. Sonra “bu noktaları en iyi üst üste getirecek kaydırma+döndürme ne?” diye hesaplanır. Gri değerlere bakmadığı için kontrast/rekon farklarına dayanıklıdır.

### 10.5 Nasıl bir araya geliyor?
1. Hedef organ centroid farkı ≤ 2 mm → **identity**.
2. Değilse **kemik centroid ötelemesi** hesaplanır (başlangıç).
3. **Adaylar:** correlation rigid, landmark (vertebra), landmark (tüm kemik).
4. **QC:** kemik maskesi Dice artmalı; centroid +0.5 mm, HD95 +1.0 mm, ASSD +0.25 mm toleransı.
5. **Seçim:** QC'yi geçenler arasından hold-out TRE (kaburga + sternum) en düşük olan.
6. **Fallback:** hiçbiri geçmezse centroid translation; o da geçmezse identity.
7. **Nihai transform tektir**; öteleme ile rigid üst üste uygulanmaz.

**Sözel olarak:** Önce “zaten yakın mı?” diye bakıyoruz; yakınsa hiç dokunmuyoruz. Değilse kemik merkezlerinden kabaca kaydırıyoruz. Sonra iki aday deniyoruz: gri değerlerden (correlation) ve kemik noktalarından (landmark) bulunan kaydırma+döndürme. Her adayı kemikler üzerinde kontrol ediyor, hiç kullanmadığımız kaburga/göğüs kemiğiyle sınıyor ve en iyisini seçiyoruz. Hiçbiri geçmezse sadece kaydırıyor, o da olmazsa hiç dokunmuyoruz.

### 10.6 Karıştırılmaması gerekenler
- **Correlation ≠ Dice korelasyonu**: CT intensite benzerliği (NCC); Dice ile aynı şey değil.
- **Landmark ≠ manuel işaretleme**: otomatik kemik label centroid'leri.
- **Centroid ≠ başlı başına registration algoritması**: translation-only bir transform.
- **Rigid ≠ centroid'in üstüne eklenen ikinci adım**: centroid başlangıç/fallback; nihai tek transform.

### 10.7 Hangi durumda hangisi seçilir?
| Durum | Seçilen |
|---|---|
| Aynı oturum test-retest (hizalı) | `identity` |
| Cross-date büyük kayma | correlation rigid veya landmark |
| Kemik az / 140 kVp | `identity` |
| Correlation başarısız, landmark var | landmark |
| Hiçbiri QC geçmez | centroid translation |
| O da geçmez | `identity` |
