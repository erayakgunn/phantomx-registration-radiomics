# PhantomXPro — CT Radyomik Öznitelik Çıkarma Aracı

CT görüntülerinden (DICOM) **organları otomatik segmente eder** ve her organdan
**radyomik öznitelikler** (şekil, yoğunluk, doku ölçütleri) çıkarıp tek bir Excel'de
açılabilir tabloya (CSV) döker. IBSI standardına uyumludur.

Akış özeti:

```
   Ham DICOM klasörü
        │
        ▼  1) İlgili serileri seç (120 kVp) + slice'ları sırala
   ┌──────────────┐
   │  NIfTI (3B)  │  2) DICOM → tek 3B hacim
   └──────────────┘
        │
        ▼  3) TotalSegmentator (yapay zekâ) ~117 organı işaretler
   ┌──────────────┐
   │  Organ maskesi│
   └──────────────┘
        │
        ▼  4) MIRP her organdan radyomik öznitelik hesaplar
   ┌──────────────────────────────┐
   │  all_features_combined.csv   │  ← SONUÇ (her satır = bir organ)
   └──────────────────────────────┘
```

**İki farklı CT cihazını otomatik tanır** ve doğru işler:
- **Somatom Edge Plus** (klasik CT / EID)
- **NAEOTOM Alpha** (foton-sayan CT / PCCT) — VMI/native spektral rekonstrüksiyonlar dahil

---

## 📋 İçindekiler
1. [Ne gerekir (bilgisayar şartları)](#1-ne-gerekir)
2. [Kurulum — adım adım (Python bilmeyene göre)](#2-kurulum--adım-adım)
3. [Çalıştırma](#3-çalıştırma)
4. [Çıktılar — ne üretilir](#4-çıktılar)
5. [Kaldığı yerden devam (durdur/sürdür)](#5-kaldığı-yerden-devam)
6. [Ayarlar (isteğe bağlı)](#6-ayarlar-isteğe-bağlı)
7. [Sorun giderme](#7-sorun-giderme)
8. [Dosyalar ne işe yarar](#8-dosyalar-ne-işe-yarar)
9. [Sık sorulan sorular](#9-sık-sorulan-sorular)

---

## 1. Ne gerekir

| Gereksinim | Açıklama |
|---|---|
| **Windows 10/11** | Test edildi. (Linux de olur, komutlar biraz değişir.) |
| **Python 3.11 veya üstü** | Programlama dili; aşağıda nasıl kurulacağı anlatılıyor. |
| **NVIDIA ekran kartı (GPU)** | Şart değil ama **çok önerilir** — yoksa segmentasyon çok yavaş olur. |
| **İnternet** | Yalnızca **ilk çalıştırmada** (yapay zekâ modeli bir kez indirilir, ~birkaç GB). |
| **~30 GB boş RAM** | 16 GB de olur (program otomatik olarak daha az paralel çalışır). |
| **Boş disk** | Girdi verisinin ~2 katı kadar (ara dosyalar için). |

> **Python nedir?** Bu aracın yazıldığı dil. Bir kez kurmanız yeterli; kod yazmayacaksınız,
> sadece hazır komutları kopyalayıp yapıştıracaksınız.

---

## 2. Kurulum — adım adım

> Bu bölümü **hiç Python bilmeyen biri** takip edebilir. Her adımı sırayla yapın.
> Komutları **PowerShell**'e yapıştıracaksınız (aşağıda nasıl açılacağı yazıyor).

### PowerShell nasıl açılır?
Başlat menüsüne `PowerShell` yazın, çıkan **Windows PowerShell**'e tıklayın. Siyah/mavi
bir pencere açılır. Komutları buraya yapıştırıp **Enter**'a basacaksınız.
(Yapıştırmak için pencereye **sağ tıklayın**.)

---

### Adım 2.1 — Python'u kur
1. https://www.python.org/downloads/ adresine gidin, **"Download Python 3.12"** (veya 3.11) indirin.
2. Kurulumu başlatın. **ÖNEMLİ:** ilk ekrandaki **"Add Python to PATH"** kutusunu **işaretleyin**,
   sonra "Install Now" deyin.
3. Kurulum bitince PowerShell açıp şunu yazın; sürüm yazısı görürseniz tamamdır:
   ```powershell
   python --version
   ```

### Adım 2.2 — Kod klasörünü yerleştir
Size gönderilen **PhantomXPro** klasörünü bilgisayarınıza kopyalayın (örneğin Masaüstüne).
PowerShell'de bu klasöre girin (yolu kendinize göre değiştirin):
```powershell
cd "C:\Users\KULLANICI_ADINIZ\Desktop\PhantomXPro"
```

### Adım 2.3 — Ana ortamı kur (radyomik hesaplama)
Aşağıdaki komutları **sırayla** yapıştırın. Bu, aracın çalışması için gereken paketleri
kendi izole "ortamına" kurar (sisteminizi kirletmez):
```powershell
python -m venv env-main
env-main\Scripts\python -m pip install --upgrade pip
env-main\Scripts\python -m pip install -r requirements.txt
```
> **venv nedir?** "Sanal ortam" — bu projeye özel, ayrı bir Python kutusu. Böylece
> paketler birbirine karışmaz.

### Adım 2.4 — Segmentasyon ortamını kur (TotalSegmentator + yapay zekâ)
TotalSegmentator, PyTorch adlı büyük bir kütüphaneye bağlı olduğu için **ayrı** bir ortama
kurulur (çakışmayı önlemek için).
```powershell
python -m venv env-ts
```
**Önce PyTorch'u** ekran kartınıza göre kurun (https://pytorch.org/get-started/locally/):
```powershell
# NVIDIA ekran kartı varsa (CUDA 12.8 örneği):
env-ts\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# Ekran kartı yoksa (yavaş, CPU):
# env-ts\Scripts\python -m pip install torch
```
**Sonra TotalSegmentator'ı** kurun:
```powershell
env-ts\Scripts\python -m pip install -r requirements-totalsegmentator.txt
```

### Adım 2.5 — İki ortamı birbirine bağla (bir kez)
Ana araç, TotalSegmentator'ın nerede olduğunu bilmeli. Aşağıdaki komutları çalıştırın
(yolları kendi klasörünüze göre düzeltin — `C:\...\PhantomXPro` kısmını):
```powershell
setx PHANTOMX_TOTALSEG_EXE "C:\Users\KULLANICI_ADINIZ\Desktop\PhantomXPro\env-ts\Scripts\TotalSegmentator.exe"
setx PHANTOMX_PY311        "C:\Users\KULLANICI_ADINIZ\Desktop\PhantomXPro\env-ts\Scripts\python.exe"
```
> `setx` sonrası **PowerShell'i kapatıp yeniden açın** (ayarlar yeni pencerede geçerli olur).

### Adım 2.6 — Kurulumu test et (isteğe bağlı ama önerilir)
Her şeyin doğru kurulduğunu 1 dakikada doğrulayın:
```powershell
env-main\Scripts\python test_pipeline.py
```
Sonda **"52 geçti, 0 başarısız"** benzeri bir yazı görürseniz kurulum tamamdır. ✅

Ayrıntılı test takımı (pytest, gerçek maskeli testler dahil):
```powershell
env-main\Scripts\python -m pytest tests -q
```

---

## 3. Çalıştırma

**Tek komut.** DICOM klasörünüzün yolunu verin (tırnak içinde):
```powershell
env-main\Scripts\python batch_120kvp.py "C:\DICOM_klasorunuzun_yolu"
```

**Örnek:**
```powershell
env-main\Scripts\python batch_120kvp.py "C:\Users\Administrator\Desktop\HastaVerisi"
```

Program çalışırken ekranda şunları görürsünüz:
```
[batch] Cihaz algılandı: NAEOTOM Alpha (foton-sayan) — kVp kaynağı: KVP etiketi [72 seri]
[batch] 72 adet 120 kVp CT serisi bulundu -> 72 benzersiz, 0 gerçek kopya.
[batch] Organ seçimi: 2 yapı -> aorta, spleen
[batch] (1/56) ...  (373 slice) — 14:22:10
[seg] çalıştırılıyor: ...
[mirp] ...: 2 organ, 2 öbek, 5 işçi ...
[batch]   -> tamam
...
```

### Önce denemek isterseniz (tavsiye edilir)
Tek seride hızlı bir deneme:
```powershell
env-main\Scripts\python batch_120kvp.py "C:\DICOM_yolu" --limit 1
```
Ya da sadece analiz (hiçbir şey işlemeden kaç seri var görmek için):
```powershell
env-main\Scripts\python batch_120kvp.py "C:\DICOM_yolu" --plan-only
```

---

## 3b. Birden çok cihazı işleyip çıktıyı denetlemek

`batch_120kvp.py` **tek** bir kaynak işler ve her koşum birleşik tabloyu yalnız
o koşumun serilerinden derler — iki cihazı arka arkaya çalıştırırsanız ikincisi
birincinin `all_features_combined.csv` ve `MANIFEST.json` dosyalarını **ezer**.
`run_and_verify.py` bunu çözer ve üstüne çıktıyı **denetler**:

```powershell
env-main\Scripts\python run_and_verify.py "C:\yol\EID" "C:\yol\PCCT"
```

Sırayla: ön kontrol (TS/GPU/paketler/disk/organ seçimi) → her kaynağı işler ve
kaynağa özel raporları `data\features\_runs\<kaynak>\` altında saklar → tüm
serileri **tek** birleşik tabloda toplar → doğrular.

| Bayrak | Ne yapar |
|---|---|
| `--limit N` | Kaynak başına ilk N seri (duman testi). |
| `--plan-only` | Yalnız ön kontrol + tarama; işleme yok. |
| `--verify-only` | İşleme yok; mevcut çıktıyı denetler. |

Denetim, pipeline'ın kendi kanıtlarını okur (yeniden hesap yapmaz) ve şunları
arar: işlenmemiş seri, çıkarımı başarısız organ, seçim dışı organ, **CT ile
maskenin ızgara uyuşmazlığı**, seriler arası MIRP ayar/sürüm kayması, birleşik
tabloda yinelenen (seri, organ) satırı, geçersiz `morph_volume`, eksik QC PNG.
Sonuç `VERIFY_report.txt`e yazılır; **çıkış kodu 0 ise temiz.**

> **Aynı anda iki koşum çalıştırmayın.** İkisi de aynı `data\features\` altına
> yazar ve Windows'ta dosya kilidi çakışması olur. `run_and_verify.py`
> kaynakları bilerek **sırayla** işler.

### Ne kadar sürer?
- **İlk çalıştırma:** yapay zekâ modelini indirir (~birkaç dk, internet hızına bağlı, bir kez).
- **Seri başına:** ~15–20 dk (GPU ile). 70 seri ≈ ~1 gün. Kesintiye dayanıklıdır (aşağıya bakın).

---

## 4. Çıktılar

Çıktı **kaynak klasörün içinde** oluşur — kod klasöründe değil. Böylece her veri
seti kendi çıktısını yanında taşır ve klasörü taşımak/arşivlemek yeterlidir:

```
C:\...\14072026-EID\
    dicoms\<seri adı>\          ← seçilen serilerin DICOM'ları (TAŞINIR)
    nifti\<seri adı>\ct.nii.gz
    segmentation\<seri adı>\seg.nii.gz + label_map.json
    overlay\<seri adı>\*.png    ← kesit başına QC görüntüsü
    features\<seri adı>\features.csv + features.done.json
    features\all_features_combined.csv / .xlsx   ← O KLASÖRÜN tüm serileri
    features\MANIFEST.json, DUPLICATES_report.txt, INCOMPLETE_report.txt,
             VERIFY_report.txt
```

Klasör adları, tablodaki **`series`** sütunuyla birebir aynı okunur addır
(Bölüm 4'teki şema) — yani bir satırdan klasöre, klasörden satıra bakmak doğrudan.

> **Her kaynak klasör kendi içinde birleştirilir.** Bir klasör tek bir cihazın
> tek bir tarihteki çekimidir; `all_features_combined` o klasörün *tüm
> serilerini* kapsar, başka kaynağı içermez. Birden çok kaynak verirseniz her
> biri kendi tablosunu ve kendi doğrulama raporunu alır — cihazlar arası tek
> bir tablo üretilmez.

> **DICOM'lar TAŞINIR, kopyalanmaz.** Seçilen serilerin dosyaları özgün
> yerlerinden `dicoms\<seri adı>\` altına alınır; dışa aktarımda klasörlere
> bölünmüş seriler böylece tek klasörde birleşir. Filtreye takılan dosyalar
> (140 kVp, topogram, rapor) yerinde kalır. Her taşınan dosyanın eski ve yeni
> yolu `dicoms\MOVE_LOG.json`e yazılır — işlem geri alınamaz olduğu için
> izlenebilirlik oradadır.

Ana çıktı dosyaları:

| Dosya | Ne işe yarar |
|---|---|
| **`all_features_combined.csv`** | **ANA SONUÇ.** Tüm serilerin birleşik tablosu. Her satır = bir *(seri × organ)*. |
| **`all_features_combined.xlsx`** | Aynı tablonun **Excel çalışma dosyası** hâli (çift tıkla açılır, otomatik oluşur). |
| `<seri>\features.csv` | Her seri için ayrı tablo. |
| `MANIFEST.json` | İşlenen tüm serilerin listesi (cihaz, kVp, rekon tipi, slice sayısı, kaç klasörden birleştirildiği). |
| `DUPLICATES_report.txt` | Klasörlere **bölünmüş** seriler (birleştirildi) ve varsa gerçek kopya slice dosyaları. |
| `INCOMPLETE_report.txt` | Bir organ çıkarılamadıysa ya da maskede hiç bulunmadıysa burada görünür — **hiçbir şey sessizce kaybolmaz**. |
| `VERIFY_report.txt` | `run_and_verify.py` ile üretilen çıktı denetimi (Bölüm 3b). |

**Ana tablodaki (CSV) sütunlar** — soldan sağa:

| blok | sütunlar |
|---|---|
| organ | `image_mask_label` |
| kimlik | **`series`** (okunur protokol adı), `series_key` (izlenebilirlik), `patient_id` |
| çekim | `date`, `institution`, `manufacturer`, `model`, `recon_k`, `k_strength`, `kvp`, `kvp_nominal`, `mAs`, `FOV`, `thick_mm`, `recon_type`, `recon_label`, `me_keV`, `take` |
| DICOM | ~160 `dcm_*` ham etiket |
| öznitelik | ~200 radyomik sütun |

**`series` adı** şu şemadadır ve iki cihazı da kapsar:

```
<kurum>_<üretici>_<model>_<tarih>_<güç>_<kernel>_<kVp>_<mAs>_<FOV>_<kesit>_<rekon>_take<n>_recon<kesit>mm
```

```
USZ_Siemens_Healthineers_NAEOTOM_Alpha_20260714_Q3_Qr40f_120_100_300_1_MonoE70keV_take1_recon1mm
USZ_SIEMENS_SOMATOM_Edge_Plus_20260714_S3_Br40f_120_100_300_1_PolyE_take1_recon1mm
```

- **Rekonstrüksiyon gücü** cihaza göre doğru yerden okunur: NAEOTOM'da QIR
  seviyesi (`Q1`–`Q4`, `SeriesDescription`'dan), SOMATOM'da kernel gücü
  (`S1`–`S5`, çok-değerli `ConvolutionKernel` etiketinden, ör. `Br40f\3`).
- **`kvp_nominal`** protokol etiketidir (PCCT'de 80/100/120/140), **`kvp`** ise
  gerçek tüp voltajı (`KVP` tag'i, PCCT'de hep 120). İkisi ayrı sütundadır;
  aksi halde bütün PolyE varyantları aynı ada düşerdi.
- **`recon_label`**: `PolyE` / `MonoE45keV` / `MonoE70keV`.
- Adlar **benzersizdir**; ayrık sütunlar sayesinde Excel'de süzme ve gruplama
  doğrudan yapılabilir.

> **İpucu:** Excel'de `recon_type` ve `kvp` sütunlarına göre süzerek farklı
> rekonstrüksiyonları/kVp'leri ayrı ayrı inceleyebilirsiniz.

Ara dosyalar: `nifti\` (3B görüntü), `segmentation\` (organ maskeleri).
Bunlara elle dokunmanıza gerek yok.

`overlay\<seri>\` — her seri için **kesit başına ayrı QC PNG'si**:
`axial_z0017.png`, `axial_z0032.png`, `coronal_y0295.png`. Organ maskeleri CT
üzerine bindirilir. Maskenin *doğru yerde* olduğunu doğrulayan tek şey budur
(geometri testleri ızgaraların uyuştuğunu söyler, maskenin nereye düştüğünü
söylemez). Yalnız seçili organlar çizilir. Kapatmak için `config.WRITE_OVERLAYS`.

Her kesit ayrı dosyadır: yan yana dizilen bir montajda paneller küçülüyor ve
asıl soru (maske doğru dokunun üstünde mi) çözünürlük yetmediği için
görünmüyordu. Her PNG kendi açıklama şeridini taşır — yalnız o kesitte görünen
organları listeler.

**Renkler:** Okabe-Ito + Paul Tol paletleri (renk körlüğüne duyarlı, yayın
standardı). Aynı anda gösterilen organlar asla aynı rengi almaz ve bir organ
her seride aynı renkte çıkar, böylece seriler gözle karşılaştırılabilir.

> **Not:** Bu klasörler `config.py` içindeki `DATA` altındadır ve koşum başında
> `config.set_output_root(<kaynak>)` ile **kaynak klasöre** bağlanır — yani
> yukarıdaki ağaçta olduğu gibi **girdi verisinin yanında** oluşur, kod
> klasöründe değil.

## 4c. Hangi organlar işlenir

Varsayılan olarak **yalnız şu 2 yapı** işlenir: `spleen`, `aorta`

Süre neredeyse tamamen organ sayısıyla ölçeklendiği için bu, en büyük hızlanma
kaldıracıdır (117 yapı yerine 2). Değiştirmek için `config.py`:

```python
TARGET_ORGANS = {"spleen", "aorta"}
# TARGET_ORGANS = None   # → maskede bulunan TÜM yapılar (117'ye kadar)
```

> **Seçim neden bu ikisi?** Aynı fantomun 142 rekonstrüksiyonunda ölçüldü:
> dalak %0,34, aorta %0,73 varyasyon katsayısı (CV) ile tekrar ediyor.
> **İliopsoas elendi** — taramalar kasın yalnız kraniyal ucunu kapsıyor (~6 mL,
> gerçek kas 200-400 mL) ve %35-40 oynuyor; aynı fiziksel nesnede bu sapma
> ölçümün temsili olmadığını gösterir. **Karaciğer** bütün olarak çok kararlı
> (CV %0,28) ama fantomda damar bulunmadığı için `liver_segments` görevinin
> ürettiği bölümleme anatomik dayanaktan yoksun (ölçüldü: segment toplamı bütün
> karaciğerden sistematik +%5,2 fazla, segment CV'leri %1-6,6).

**Daraltma geri alınabilir, yeniden çıkarım gerektirmez.** Seri başına
`features.csv` dosyaları o seri işlendiğindeki tüm organları saklar; birleşik
tablo derlenirken **güncel** seçime göre süzülür. Listeyi genişletip
`run_and_verify.py --interim` çalıştırmanız yeterli — MIRP her ROI'yi bağımsız
hesapladığı için bir organın değerleri yanında kaç organ işlendiğinden
etkilenmez.

- **Ad yazılır, etiket numarası değil.** Geçerli adlar `totalseg_total_labels.json`
  içindedir. Yanlış bir ad yazarsanız program **ilk saniyede** net bir hatayla
  durur (sessizce 0 organ çıkarmaz).
- Bazı yapılar TotalSegmentator'da sol/sağ ayrıdır (`iliopsoas_left` /
  `iliopsoas_right`, `kidney_left` / `kidney_right`); böyle bir yapıyı işlemek
  için **iki adı da** listeye yazın.
- Bir hedef organ maskede **hiç bulunamazsa** (ör. taranan bölge onu kapsamıyorsa)
  bu `INCOMPLETE_report.txt`'te *çıkarım hatasından ayrı* bir başlıkta raporlanır.
- Seçim, Excel çıktısının `methods` sayfasındaki **beyana** da yazılır.

---

## 4b. Klasörleri DICOM'a göre adlandırma (isteğe bağlı)

İşlem bitince, çıktı klasörlerini karışık kimlik yerine **DICOM metadata'sından**
üretilen okunur adlarla yeniden adlandırabilirsiniz:

```powershell
# Önce ÖNİZLEME (hiçbir şeyi değiştirmez — adları ve çakışmaları gösterir):
env-main\Scripts\python apply_naming.py "C:\DICOM_yolu" --dry-run

# Sonra UYGULA (features/nifti/segmentations klasörlerini yeniden adlandırır):
env-main\Scripts\python apply_naming.py "C:\DICOM_yolu"
```

**Ad şeması** (yalnız DICOM metadata; klasör/dosya adına bakılmaz):
```
<tarih>_<kurum>_<üretici>_<model>_<kernel>_<kVp>_<mAs>_<FOV>_<kesit>
```
Örnek (SOMATOM Edge Plus): `20260629_USZ_SIEMENS_SOMATOM_Edge_Plus_Br36f_120_100_300_3`

- **PCCT (NAEOTOM):** sona rekonstrüksiyon ekleri gelir — QIR seviyesi (Q1..Q4) ve
  `native`/`VMI70keV` (bunlar `SeriesDescription`'dan okunur, tıpkı kVp gibi).
  Örnek: `..._Qr40f_120_100_300_1_Q3_VMI70keV`.
- **Test-retest:** `_take1` / `_take2` eki eklenir.
- **Çakışma olmaz:** şemanın ayırt edemediği iki farklı seri asla aynı ada düşmez
  (gerekirse `_s<SeriesNo>` eklenir) — **hiçbir seri ezilmez/kaybolmaz**.
- Ayrıca `NAMING_MAP.csv` (ad ↔ kimlik ↔ kaynak ↔ tüm alanlar) yazılır ve
  `all_features_combined.csv`'ye ilk sütun olarak **`metadata_name`** eklenir.
- Yeniden adlandırma **"kaldığı yerden devam"ı bozmaz** (aşağıya bakın).

---

## 5. Kaldığı yerden devam

İş uzun sürer. Bilgisayar kapanır, program çökerse ya da `Ctrl+C` ile durdurursanız:
**aynı komutu tekrar çalıştırın** — biten seriler otomatik atlanır, kalınan yerden devam eder.

- İlerlemeyi görmek için (ayrı bir PowerShell penceresinde):
  ```powershell
  (Get-ChildItem data\features -Recurse -Filter features.done.json).Count
  ```
  Bu sayı, biten seri sayısıdır.
- Bir seriyi **zorla yeniden işletmek** isterseniz, o serinin klasöründeki
  `features.done.json` dosyasını silip komutu tekrar çalıştırın.

---

## 6. Ayarlar (isteğe bağlı)

Çoğu kullanıcı hiçbir ayara dokunmaz. Gerekirse `config.py` dosyasını bir metin
düzenleyiciyle (Not Defteri olur) açıp şu satırları değiştirebilirsiniz:

| Ayar | Ne yapar |
|---|---|
| `TARGET_KVP = 120.0` | Hangi kVp işlensin (varsayılan 120). |
| `TARGET_RECON_TYPES = None` | `None` = tüm rekonlar (native + VMI). Sadece belirli olanı istemek için ör. `{"native"}` veya `{"VMI_70keV"}`. |
| `TARGET_ORGANS` | **Hangi organlardan öznitelik çıkarılsın.** Varsayılan: `{"spleen", "aorta"}`. Tüm 117 yapı için `None` yapın. Ayrıntı: Bölüm 4c. |
| `TS_DEVICE = "gpu"` | Ekran kartı yoksa `"cpu"` yapın. |
| `WORKER_MEM_MULTIPLIER = 28.0` | Bellek (RAM) hatası alırsanız **artırın** (ör. 40). Daha hızlı ve bol RAM'iniz varsa düşürün. |

Cihaz tipi (EID/PCCT) ve kVp kaynağı **otomatik** belirlenir; ayar gerekmez.

---

## 7. Sorun giderme

| Belirti | Çözüm |
|---|---|
| `TotalSegmentator çalıştırılabiliri bulunamadı` | Adım 2.5'teki `setx` komutlarını yapıp **PowerShell'i yeniden açın**. |
| `No module named 'mirp'` | Komutu `env-main\Scripts\python ...` ile çalıştırdığınızdan emin olun (sadece `python` değil). |
| İlk çalıştırma çok yavaş / takılmış gibi | Model indiriliyordur (internet); bir kez olur, sabırla bekleyin. |
| Bellek hatası (OOM) | `config.py` → `WORKER_MEM_MULTIPLIER` değerini artırın (40 → 50). |
| Ekran kartı kullanılmıyor | `env-ts\Scripts\python -c "import torch; print(torch.cuda.is_available())"` çalıştırın. `False` çıkarsa PyTorch'u CUDA sürümüyle yeniden kurun (Adım 2.4). |
| Çok uzun hasta yolu / `FileNotFoundError` | Windows uzun-yol desteğini açın: yönetici PowerShell'de `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`, sonra yeniden başlatın. |
| `all_features_combined.csv` yazılamadı | Dosya Excel'de açıksa `_yeni` ekiyle kaydedilir; Excel'i kapatıp tekrar çalıştırın. |

---

## 8. Dosyalar ne işe yarar

| Dosya | Görevi |
|---|---|
| `batch_120kvp.py` | **Ana program.** Tek bir kaynağı işler. |
| `run_and_verify.py` | **Çok kaynaklı koşum + çıktı denetimi** (Bölüm 3b). |
| `config.py` | Tüm ayarlar (kVp, organ seçimi, cihaz, bellek, MIRP). |
| `sort_spatially.py` | Cihaz/kVp/rekon tespiti + slice sıralama. |
| `run_totalsegmentator.py` | DICOM→NIfTI dönüşümü + organ segmentasyonu. |
| `feature_extraction_mirp.py` | Radyomik öznitelik hesabı (MIRP, IBSI) + organ seçimi. |
| `dicom_metadata.py` | Çekim parametrelerini DICOM'dan tabloya taşır (`dcm_*` sütunları). |
| `excel_report.py` | Biçimlendirilmiş `.xlsx` + `methods` beyan sayfası. |
| `overlay_qc.py` | Segmentasyonun görsel kontrolü (QC PNG'leri). |
| `series_naming.py` | DICOM metadata'sından okunur seri adı üretir (Bölüm 4b). |
| `apply_naming.py` | Çıktı klasörlerini bu adlarla yeniden adlandırır (Bölüm 4b). |
| `test_pipeline.py` | Kurulum doğrulama testleri (Adım 2.6). |
| `totalseg_total_labels.json` | Organ numara↔ad haritası (repo ile gelir). |
| `requirements*.txt` | Kurulacak paket listeleri. |

---

## 9. Sık sorulan sorular

**S: Verimin PCCT mi klasik CT mi olduğunu ayarlamam gerekir mi?**
Hayır. Program DICOM içeriğinden (cihaz modeli, ImageType) otomatik anlar ve doğru
kVp kaynağını seçer. Çalıştırma başında hangi cihazı algıladığını yazar.

**S: Sadece 120 kVp mı işleniyor?**
Varsayılan evet. `config.py` → `TARGET_KVP` ile değiştirilebilir. kVp **her
cihazda DICOM `KVP` etiketinden (0018,0060)** okunur — klasör adından da,
`SeriesDescription` metninden de değil.

> **Neden `SeriesDescription`'a güvenilmiyor:** NAEOTOM Alpha'da tüp voltajı
> 120 kV'de sabittir; açıklamasında "80kVp"/"100kVp" yazan seriler de `KVP`
> etiketinde 120 raporlar. Bunlar 80/100 kV **çekimi değil**, aynı 120 kV
> çekiminden türetilmiş farklı PolyE rekonstrüksiyonlarıdır ve ayrı seriler
> olarak işlenir. Eski davranışa dönmek için `config.KVP_FROM_DESCRIPTION = True`.

**S: VMI (spektral) serilerle native karışır mı?**
Karışmaz. Her rekonstrüksiyon ayrı işlenir ve çıktıda `recon_type` sütunuyla
etiketlenir. İstemezseniz `TARGET_RECON_TYPES` ile kısıtlayabilirsiniz.

**S: Radyomik değerler standart mı?**
Evet, **IBSI uyumlu** (1 mm izotropik yeniden örnekleme, tricubic interpolasyon,
sabit 25 HU bin genişliği). Yeniden segmentasyon aralığı **[-1300, 3100] HU**:
bu aralık maske içinde ölçülen tüm yoğunlukları kapsar, yani **hiçbir vokseli
elemez** — yalnızca sabit-bin-genişliği için gri seviye orijinini sabitler.
(Eskiden [-1000, 400] yumuşak doku aralığı kullanılıyordu; ölçüldü, kemiği
tanımlayan yoğunlukları siliyordu — `vertebrae_T10` voksellerinin %69,5'i.)
Öznitelik ailelerinin tümü hesaplanır. Kullanılan tam beyan, Excel çıktısının
**`methods`** sayfasında canlı ayarlardan üretilir.

**Tablodaki her öznitelik KESİN hesaplanır.** `morph_moran_i` ve
`morph_geary_c` çıktıya alınmaz: MIRP bu ikisini 1000 vokselin üstünde
Monte-Carlo ile *kestiriyor* ve tohum dosya yolunu içerdiği için veri
taşındığında değerler değişiyor — ne IBSI'nin tanımladığı kesin hesap ne de
tekrarlanabilir. Tam hesap (`no_approximation=True`) O(n²) olduğundan bu
boyutlarda pratik değil.

Sonucu belirleyen parametreler MIRP varsayılanından **miras alınmaz, config'de
yazılır** — sürüm değişince sessizce kaymasınlar diye. Buna IVH
ayrımlaştırması da dahil: `ivh_discretisation_method = "none"`, yani IVH
doğrudan HU üzerinden hesaplanır (CT kalibre ve tam sayı değerli olduğu için
IBSI'nin öngördüğü davranış budur; 14 `ivh_*` özniteliğini bu tek parametre
belirliyor). `tests/test_ibsi_params.py` bu ayarların tümünü altın değerlerle
kilitler.

**S: Sonucu nasıl kullanırım?**
`all_features_combined.csv`'yi Excel/Python/R ile açıp analiz edin. Her satır bir
organın öznitelik vektörüdür.

---

*Sorularınız için: kodu size ileten kişiyle iletişime geçin.*
