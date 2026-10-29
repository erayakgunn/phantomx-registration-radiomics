# Zurich Test-Retest — Çalışma Notları

Tarih: 29 Ekim 2026
Veri: `C:\Users\eray\Desktop\zurich_pcct`, `C:\Users\eray\Desktop\zurich_sep`
Çalışma alanları: `zurich_pcct_work`, `zurich_sep_work` (junction'lı `LABELED` + `REGISTRATION`)

## 1. Veri setleri

### PCCT (`zurich_pcct`)
Test-retest fantom çalışması (PhantomX & CalibraMics), NAEOTOM Alpha (seri 127032), 20260929.
- `PCTT_20260929_1`: 4 protokol × 10 tekrar = 40 seri
  - 120 kVp, 100 mAs, 1 mm PolyE / MonoE70keV
  - 120 kVp, 100 mAs, 3 mm PolyE / MonoE70keV
- `PCTT_20260929_2`: 2 protokol × 10 tekrar = 20 seri
  - 120 kVp, 300 mAs, 1 mm PolyE / MonoE70keV
  - (140 kVp klasörleri var, nifti/seg/features üretilmemiş)
- Her seride `nifti/ct.nii.gz`, `segmentation/seg.nii.gz` (TotalSegmentator; aorta 52, spleen 1), `features/`, `dicoms/`, `overlay/`.

### SEP (`zurich_sep`)
SOMATOM Edge Plus, 20260929.
- `SEP_20260929_1`: 2 protokol × 10 = 20 seri (Br40f, S2_Br40f; 120 kV, 100 mAs, 1 mm)
- `wetransfer_sep_20260929_2_...`: 40 seri
  - Br40f 120 kV 300 mAs (10)
  - S3_Br40f 120 kV 100 mAs (20)
  - S3_Br40f 120 kV 300 mAs (10)

## 2. Registration sonuçları (take1 referansı, `--skip-threshold-mm 0`)

### PCCT (her protokol 9 pair)
| Protokol | Spleen | Aorta |
|---|---|---|
| 1mm MonoE | 0.9643 → 0.9692 | 0.9380 → 0.9437 |
| 1mm PolyE | 0.9651 → 0.9678 | 0.9384 → 0.9394 |
| 3mm MonoE | 0.9619 → 0.9629 | 0.9282 → 0.9294 |
| 3mm PolyE | 0.9645 → 0.9650 | 0.9331 → 0.9334 |
| 300mAs MonoE | 0.9672 → 0.9682 | 0.9402 → 0.9419 |
| 300mAs PolyE | 0.9668 → 0.9668 | 0.9391 → 0.9403 |

### SEP
| Protokol | n | Spleen | Aorta |
|---|---|---|---|
| Br40f | 9 | 0.9629 → 0.9629 | 0.9296 → 0.9296 |
| S2 | 9 | 0.9675 → 0.9675 | 0.9479 → 0.9479 |
| Br40f 300mAs | 9 | 0.9679 → 0.9679 | 0.9476 → 0.9476 |
| S3 | 19 | 0.9702 → 0.9702 | 0.9403 → 0.9403 |
| S3 300mAs | 9 | 0.9678 → 0.9678 | 0.9479 → 0.9479 |

SEP'te organ centroid farkı ~0.3 mm, HD95 ~1.16 mm → registration identity seçiyor.

## 3. Tavan analizi (0.99 neden ulaşılamaz)

- Aynı ham çekimin MonoE vs PolyE rekonstrüksiyonları (aynı `IrradiationEventUID`) arasında:
  - **Spleen 0.9768**, **Aorta 0.9502** (identity Dice).
- 1-voxel sınır kayması için küre yaklaşımı: `Dice ≈ 1 − 3δ/(2r)`
  - Spleen (~293k voxel, r≈41): δ=1 → **0.964**
  - Aorta (~68k voxel, r≈25): δ=1 → **0.941**
- Gözlenen değerler tam olarak 1-voxel sınır farkı seviyesinde. 0.99 için sub-voxel uyum gerekir.

## 4. Denenen güçlendirmeler (temsili pair)

| Yöntem | Spleen | Aorta |
|---|---|---|
| Rigid (mevcut) | 0.9672 | 0.9335 |
| Affine (CT correlation) | ıraksadı | ıraksadı |
| B-spline CT grid 8 / 4 | 0.9586 / 0.9606 | 0.9331 / 0.9236 |
| B-spline spleen mask | 0.9629 | — |
| Rigid + Demons | 0.9676 | 0.9373 |

En kötü pair'de (3mm PolyE s39) Demons: spleen 0.9457→0.9560, aorta 0.9045→0.9094.

**Sonuç:** Rigid registration tavanına ulaşıldı; kalan fark segmentasyon/partial volume.

## 5. Konsensüs (ensemble) segmentasyon planı

Amaç: test-retest Dice'ı rekonstrüksiyon tavanına (0.9768 / 0.9502) yaklaştırmak.

1. **Çoklu rekonstrüksiyon:** Her take için MonoE + PolyE segmentasyonlarını birleştir (çoğunluk oyu veya STAPLE).
2. **TTA:** Tek rekonstrüksiyonda CT'yi flip/rotate/resample ederek birden fazla segmentasyon üret ve birleştir.
3. **STAPLE:** Segmentasyon güvenilirliklerini tahmin edip gizli "gerçek" maskeyi kestir.
4. **Aynı rekonstrüksiyon:** Tüm take'leri tek rekonstrüksiyonda segmentle (rekonstrüksiyon farkını sıfırla).
5. **Beklenti:** 0.97 / 0.95 civarı; tam 0.9768/0.9502 garanti değil (test-retest ekstra gürültü).
6. **Uyarı:** Konsensüs ground truth değil, kestirilmiş referanstır; raporda belirtilmeli.

## 6. Rigid registration sınırı

- 6-DOF rigid ile poz hatası zaten ~0.3 mm; ICP/MI/daha ince grid gibi rigid iyileştirmeler en fazla ~0.001–0.005 katkı verir.
- 0.99 veya rekonstrüksiyon tavanı rigid ile ulaşılabilir değil.
- Konunun dışına çıkmadan (rigid içinde) yapılabilecek anlamlı iyileştirme kalmadı.
- Sonraki adım: segmentasyon harmonizasyonu (konsensüs/STAPLE/re-segmentation) veya deformable (farklı yöntem, şekli değiştirir).
