# Zurich Test-Retest — VOI Raporu

Tarih: 31 Ekim 2026
Veri: `zurich_pcct` (NAEOTOM Alpha), `zurich_sep` (SOMATOM Edge Plus)
Çalışma alanları: `zurich_pcct_work`, `zurich_sep_work`

## 1. Amaç
Test-retest çakıştırma kalitesini ölçmek ve merkezden **aynı fiziksel VOI** alarak
farklı tarihlerde (cross-date) aynı hacmi yakalayabilmek.

## 2. VOI tanımı
- Referans take1'in organ merkezinden sabit fiziksel küp: **liver 30 mm**, **aorta 10 mm**.
- VOI grid'i referans uzayında tanımlı; her take/registered pair aynı fiziksel hacme resample edilir.
- Script'ler: `voi_extract.py` (native), `voi_registered.py` (cross-date), `voi_dice.py` (VOI Dice).

## 3. Tam-organ Dice (test-retest, take1 referansı)
| Veri | Spleen | Aorta |
|---|---|---|
| PCCT (6 protokol) | 0.962–0.969 | 0.928–0.944 |
| SEP (5 protokol) | 0.963–0.970 | 0.930–0.948 |

## 4. VOI kapsama ve HU (test-retest)
| Veri | Organ | Kapsama | Merkez kayması |
|---|---|---|---|
| PCCT | liver | 0.987–0.999 | 0.08–0.14 mm |
| PCCT | aorta | 0.984–0.991 | 0.16–0.39 mm |
| SEP | liver | 0.998–0.999 | 0.14–0.29 mm |
| SEP | aorta | 0.983–0.992 | 0.14–0.37 mm |

Take'ler arası ortalama HU kararlılığı: **0.13–1.19 HU**.

## 5. VOI-içi organ Dice (asıl sonuç)
| Kohort | Organ | n | VOI Dice (ort) | min | max | Tam-organ Dice |
|---|---|---|---|---|---|---|
| PCCT test-retest | aorta | 54 | **0.9944** | 0.9892 | 0.9979 | 0.9380 |
| PCCT test-retest | liver | 54 | **0.9975** | 0.9939 | 0.9999 | 0.9813 |
| PCCT cross-date | aorta | 88 | **0.9937** | 0.9897 | 0.9993 | 0.9329 |
| PCCT cross-date | liver | 88 | **0.9956** | 0.9936 | 0.9978 | 0.9800 |
| SEP test-retest | aorta | 55 | **0.9952** | 0.9901 | 0.9977 | 0.9422 |
| SEP test-retest | liver | 55 | **0.9994** | 0.9985 | 0.9998 | 0.9817 |
| SEP cross-date | aorta | 67 | **0.9967** | 0.9882 | 1.0000 | 0.9350 |
| SEP cross-date | liver | 67 | **0.9995** | 0.9967 | 1.0000 | 0.9813 |

## 6. Yorum
- **VOI içinde Dice ~0.99** ve bu, tam-organ Dice'ından (~0.93 aorta / ~0.98 liver) belirgin yüksek.
  Sebep: VOI merkezi/boundary'den uzak bölgeyi örnekliyor; kalan fark sınır/segmentasyon kaynaklı.
- **Cross-date ≈ test-retest**: farklar <0.002. Yani farklı tarihlerde hizalama + aynı VOI,
  test-retest tekrarlanabilirliği seviyesine ulaşıyor.
- Test-retest tavanı (tam organ) ~0.96/0.93 iken, VOI yaklaşımı **0.99** hedefine ulaşıyor.

## 7. Aykırılar
- SEP `_s80` ve `S3 take1`: centroid shift 7.9 / 4.1 mm (bilinen düşük-kalite seriler).
- Buna rağmen VOI-içi Dice ~0.99 (liver 0.999, aorta 0.996–0.997).
- Yani VOI yaklaşımı bu aykırılara karşı dayanıklı.

## 8. Sonuç
Merkezden aynı VOI alındığında, hem test-retest hem cross-date için organ Dice'ı **~0.99**
seviyesine çıkıyor ve cross-date performansı test-retest ile eşleşiyor. Bu, hocanın
"çakıştırma + merkezden aynı hacim" hedefinin karşılandığını gösterir.
