# AcibademKent — CalibraMics Etiketleme Kodu (SOMATOM Force destekli)

USZ PCCT pipeline'ının (`pcct_labeling`) **Siemens SOMATOM Force (Dual-Source)**
desteği eklenmiş tam kopyası. Aynı kod USZ NAEOTOM Alpha (PCCT) ve Philips
verilerini de işler — Force'a özgü yollar cihaz tespitiyle otomatik devreye girer.

## Force'a özgü eklenen özellikler

1. **Cihaz tespiti + adlandırma** (`labeling/name_builder.py`)
   - `is_siemens_force()`: Manufacturer=SIEMENS + Model=SOMATOM Force
   - `build_force_base()`: isim şablonu
     `<Kurum>_SIEMENS_SOMATOM_Force_<Tarih>_<Kernel>_<kV>_<mAs>_<FOV>_<kalınlık>_PolyE_takeN_reconXmm`
   - Bu cihazda MonoE/VMI yoktur (tüm rekonlar polikromatik) → recon-type sabit `PolyE`.
   - Serileri ayıran DICOM tag'leri: `ConvolutionKernel` (Br40d/Br59d), `KVP`
     (gerçek tüp kV), `Exposure` (mAs), `ReconstructionDiameter`, `SliceThickness`.
   - Tekrar çekimler (test-retest): `AcquisitionDateTime` / `IrradiationEventUID`
     ile ayrışır → `take1`/`take2` (mevcut PCCT take mantığı yeniden kullanılır).

2. **Çoklu-fantom bölme** (`labeling/phantom_instances.py`, YENİ)
   - Acıbadem çekimlerinde tek spiral taramada uç uca 3 fantom kopyası +
     koltuk bulunuyor. Bu modül, her ~4. dilimde Hough çember testiyle
     fantom z-aralıklarını bulur, hava boşluklarından böler.
   - Seri N>1 örnek içeriyorsa `<isim>_P1.._PN` klasörleri oluşur; her örnek
     bağımsız annotate edilir (STAR16/STAR8/INSERT1/INSERT2 seti örnek başına).
     Ters yerleştirilmiş kopyalar mevcut yön-düzeltmeyle otomatik çevrilir.
   - Hava boşluğu + koltuk dilimleri kopyalanmaz.
   - Sadece Force serilerinde çalışır (USZ/PCCT akışı değişmedi).

3. **Kurum override** (`label_series.py --institution`)
   - Bu cihazın `InstitutionName` tag'i anonim ("Ser.Nr.246663") olduğundan
     kurum adı CLI'dan verilir.

## Kullanım

```bash
cd codes
python label_series.py ^
  --input  "C:\...\PHANTOM_CALIBRAMICS_123123123_DualSource" ^
  --output-dir "C:\...\AcibademKent_SOMATOM_Force_20251023_LABELED" ^
  --institution AcibademKent ^
  --log-file "C:\...\AcibademKent_SOMATOM_Force_20251023_LABELED\run.log"
```

- Topogram / Dose Report / Patient Protocol serileri otomatik atlanır.
- Çıktı yapısı USZ ile birebir aynıdır (`dicoms/ Inserts/ Stars/ Textures/
  overlays/ coronal/ painted/ metadata.json` + `labels.csv/json`), dolayısıyla
  mevcut MIRP radiomics scripti (`PCCT - Radiomics Code`) değişiklik gerektirmeden
  bu çıktı üzerinde çalışır.

## Gereksinimler

`codes/requirements.txt` (numpy, scipy, scikit-image, opencv-python, pydicom,
matplotlib, tqdm) + duplicate audit için `openpyxl`. Python 3.12+.

## Doğrulanmış test (2026-07-09)

20251023 Acıbadem çekimi, 1 seri → 3 fantom örneği: her örnekte
STAR16 (≈38–41 mm) + STAR8 (40) + INSERT1 (40) + INSERT2 (40), 0 hata;
ters yerleşimli orta kopya (P2) dahil overlay'ler görsel olarak doğrulandı.
