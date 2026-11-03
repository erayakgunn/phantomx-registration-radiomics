# Konsensüs Referansı ile Dice Analizi — Açıklayıcı Rapor

**Tarih:** 2 Kasım 2026  
**Kapsam:** Zurich test-retest (PCCT + SEP) ve longitudinal PCCT/SEP  
**İlgili dosyalar:** `ZURICH_CONSENSUS_REFERANS.xlsx`, `PCCT_SEP_LONG_CONSENSUS.xlsx`

---

## 1. Amaç ve problem

Sorulan soru: **Görüntüleri yeniden çekmeden, maskeleri değiştirmeden ve veriyi bozmadan Dice daha fazla iyileştirilebilir mi?**

Kısa cevap: Aynı maskeler için Dice **referansa bağlıdır**. Mevcut analizlerde referans **20260805 take1** (tek bir seri). Take1, kohortun yalnızca bir örneğidir. Onun yerine **kohort konsensüsü** kullanılırsa Dice değişir; bu, segmentasyonu iyileştirmeden **referans seçiminin etkisini** ölçen meşru bir analizdir.

Bu raporda ne yaptığımızı, ne elde ettiğimizi ve bunun ne anlama geldiğini/ne anlama gelmediğini açıklıyoruz.

---

## 2. Ne yaptık?

### 2.1 Değişmeyenler
- **Görüntüler değişmedi.**
- **Per-series segmentasyon maskeleri değişmedi.**
- Yeniden çekim, yeniden segmentasyon, post-processing veya deformable warp **yapılmadı.**

### 2.2 Değişen tek şey: referans
Her seri için:
1. Diğer serilerin **fixed grid'e taşınmış** maskeleri alındı.
2. Bu maskelerin **çoğunluk oyu** (≥ %50) ile bir **konsensüs maske** üretildi.
3. Değerlendirilen seri **konsensüsten dışlandı** (leave-one-out, LOO) → döngüsellik yok.
4. İki Dice hesaplandı:
   - `Dice_take1` = seri ↔ 20260805 take1
   - `Dice_consensus` = seri ↔ LOO kohort konsensüsü

### 2.3 Veri kapsamı
- **Zurich test-retest:** PCCT 8 protokol × 10 take; SEP 5 grup. Toplam **254 per-image** karşılaştırma (aorta + spleen).
- **Longitudinal:** PCCT **88** seri; SEP **67** seri.

---

## 3. Bulgular

### 3.1 Zurich test-retest — tüm görüntüler

| Veri seti | Organ | N | Take1 ref | Konsensüs ref | Δ | İyileşen / Kötüleşen |
|---|---|---:|---:|---:|---:|---:|
| PCCT | Aorta | 72 | 0.9325 | **0.9475** | **+0.0149** | 60 / 12 |
| PCCT | Spleen | 72 | 0.9653 | **0.9724** | **+0.0071** | 61 / 11 |
| SEP | Aorta | 55 | 0.9476 | **0.9883** | **+0.0407** | **55 / 0** |
| SEP | Spleen | 55 | 0.9676 | **0.9963** | **+0.0287** | 54 / 1 |

- SEP'te aort **55/55 iyileşti**; dalak 54/55.
- PCCT 140 kVp en büyük PCCT kazancı: aort 0.9245→0.9499 (+0.0254), dalak 0.9607→0.9735 (+0.0128).
- SEP aort ~0.99, SEP dalak ~0.996 seviyesine çıktı.

### 3.2 Longitudinal PCCT / SEP — tüm görüntüler

| Veri seti | Organ | N | Take1 ref | Konsensüs ref | Δ | İyileşen / Kötüleşen |
|---|---|---:|---:|---:|---:|---:|
| PCCT | Aorta | 88 | 0.9306 | **0.9467** | **+0.0161** | 79 / 9 |
| PCCT | Spleen | 88 | 0.9644 | **0.9730** | **+0.0085** | 80 / 8 |
| SEP | Aorta | 67 | 0.9274 | **0.9498** | **+0.0224** | 51 / 16 |
| SEP | Spleen | 67 | 0.9620 | **0.9733** | **+0.0113** | 47 / 20 |

- Kazanç **gerçek ama daha küçük**: longitudinal kohort farklı tarih/protokol içerdiği için konsensüs daha "gevşek".
- Bazı seriler take1'e daha yakın olduğu için konsensüste kötüleşiyor.

---

## 4. Bu neden iyileşiyor? (ve ne değil)

### 4.1 Neden iyileşiyor
- Konsensüs, kohortun **merkez/medyan şekli**; take1 ise tek bir örnek.
- Bir seriyi **merkeze** göre karşılaştırmak, **uçtaki bir örneğe** göre karşılaştırmaktan matematiksel olarak daha yüksek Dice verir.
- Yani kazanç kısmen **beklenen ve matematiksel**; "yeni bir segmentasyon" değil.

### 4.2 Ne değil
- Konsensüs **ground truth değil**; aynı model maskelerinden türetildi.
- **Segmentasyon doğruluğunu artırmaz**; yalnızca **referansa bağlı skoru** daha stabil hale getirir.
- **Registration'ın test-retest'te faydalı olduğunu göstermez.** Test-retest'te identity doğru; registration cross-date'te gerekli (Dice 0.03→0.96).
- 0.99'a ulaşan SEP değerleri **konsensüs referansına** aittir; take1 referansıyla aynı sayı değildir.

### 4.3 Doğru çerçeve
Bu sonuç şu şekilde sunulmalıdır:
> "Dice referansa duyarlıdır. Take1 tek bir örnek yerine leave-one-out kohort konsensüsü kullanıldığında, aynı maskelerle Dice artmaktadır. Bu, segmentasyon doğruluğu kazancı değil, **referans sağlamlığı / kohort konsensüsüne göre tekrarlanabilirlik** sonucudur."

---

## 5. Sınırlılıklar

1. **Konsensüs bağımsız değil:** aynı TotalSegmentator maskelerinden türetildi; model hatası konsensüste de var.
2. **LOO döngüselliği azaltır ama tamamen kaldırmaz:** değerlendirilen seri dışlanıyor; ancak konsensüs yine aynı model ailesinden.
3. **Longitudinal kohort heterojen:** farklı tarih/rekon/pozisyon; konsensüs daha gevşek, kazanç daha küçük.
4. **Referans değişince Dice'ın anlamı değişir:** take1 ve konsensüs Dice'ları aynı tabloda karıştırılmamalı; etiketlenmeli.
5. **PCCT 1 mm hâlâ ~0.95/0.97:** konsensüs bile 0.99'a çıkarmıyor; kalan fark segmentasyon/partial volume tavanı.
6. **Aorta ince/tübüler:** 1 voksel sınır farkı Dice'ı spleen'den daha çok etkiler.

---

## 6. Hocaya sorulacak sorular

1. **Konsensüs referansı** birincil referans olarak kabul edilmeli mi, yoksa **ikincil/sensitivity** analiz olarak mı raporlanmalı?
2. Çoğunluk oyu yerine **STAPLE** veya **medoid referans** tercih edilir mi?
3. Longitudinal çalışmada **take1 sabit referansı korunup** konsensüs yalnızca **robustness** olarak mı verilmeli?
4. Site/rapor: **iki referans yan yana** mı gösterilmeli (take1 + konsensüs), yoksa tek referans mı?
5. Bu sonuç "veri bozmadan iyileşme" olarak mı, yoksa "**referans etkisi / referans sağlamlığı**" olarak mı sunulmalı?
6. Fantomun **bilinen geometrisi** ground truth olarak kullanılabilir mi? (O zaman Dice gerçek doğruluğu ölçer.)

---

## 7. Öneri

- **İki referansı birlikte raporla:**
  - `take1` → önceki raporlarla uyumlu, sabit referans.
  - `konsensüs` → referans sağlamlığı / sensitivity.
- Konsensüsü **birincil sonuç yapma**; "aynı veri, farklı referans" olarak konumlandır.
- **Registration'ı test-retest'te zorlama** (identity doğru); cross-date'te kullan.
- **0.99 hedefini** segmentasyon/rekon tavanına göre yorumla; 0.99'a konsensüs referansıyla SEP'te ulaşıldı, PCCT 1 mm'de ulaşılamıyor.
- Uzun vadede gerçek iyileşme için: **daha iyi segmentasyon** (ensemble/manuel/STAPLE) veya **daha iyi acquisition/rekon**; deformable ise veriyi bozar.

---

## 8. Tek cümlelik özet

> Aynı görüntü ve aynı per-series maskelerle, referansı take1'den **leave-one-out kohort konsensüsüne** çevirdiğimizde Dice artıyor (Zurich SEP aort 0.948→0.988; PCCT aort 0.933→0.948); bu **veri bozmadan** elde edilen bir **referans-seçimi** kazancıdır, segmentasyon doğruluğu kazancı değildir.
