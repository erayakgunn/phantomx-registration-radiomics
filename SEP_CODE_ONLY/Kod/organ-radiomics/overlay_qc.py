"""Segmentasyonun görsel kontrolü: organ maskelerini CT üzerine bindirir.

Bu hat şimdiye kadar `seg.nii.gz` yazıp geçiyordu. Geometri tutarlılığı testlerle
doğrulanıyor ama maskenin DOĞRU YERDE olduğunu kimse görmüyordu: yanlış bir
segmentasyon ancak öznitelikler tuhaf çıktığında, yani analiz aşamasında fark
ediliyordu. Fantom hattında bu kontrol baştan beri var (overlay PNG'leri);
buradaki karşılığı bu.

ÖNEMLİ: Buradaki pencereleme ve renklendirme YALNIZCA GÖRÜNTÜLEME içindir.
Öznitelikler ham NIfTI'den hesaplanıyor; bu modül hiçbir veriye dokunmuyor ve
hiçbir çıktısı çıkarıma girmiyor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Görüntüleme penceresi (yalnızca insan gözü için): yumuşak doku.
_WINDOW_LEVEL = 40.0
_WINDOW_WIDTH = 400.0

# Yayın kalitesinde nitel palet: Okabe-Ito + Paul Tol (bright/muted) — ikisi de
# renk körlüğüne duyarlı tasarlanmış ve Nature türü dergilerde standart.
# Gri CT üzerinde ayırt edilebilmesi için siyah/gri tonlar dışarıda bırakıldı.
# Sıra, komşu renklerin ton olarak birbirinden uzak olmasına göre dizildi.
_PALETTE_HEX = (
    "#E69F00",  # turuncu        (Okabe-Ito)
    "#0072B2",  # koyu mavi      (Okabe-Ito)
    "#009E73",  # yeşil          (Okabe-Ito)
    "#CC79A7",  # pembe-mor      (Okabe-Ito)
    "#F0E442",  # sarı           (Okabe-Ito)
    "#D55E00",  # kiremit        (Okabe-Ito)
    "#56B4E9",  # açık mavi      (Okabe-Ito)
    "#882255",  # bordo          (Tol muted)
    # İlk 8 renk bilerek Okabe-Ito'nun tam seti + bordo: 8 etiketli bir küme
    # (ör. karaciğerin 8 Couinaud segmenti) en ayrık renkleri alsın. Sonrakiler
    # daha yakın tonlar; 8'i aşan kümelerde ayrım açıklama şeridinden okunur.
    "#44AA99",  # turkuaz        (Tol muted)
    "#AA4499",  # mor            (Tol muted)
    "#117733",  # koyu yeşil     (Tol muted)
    "#EE6677",  # mercan         (Tol bright)
    "#332288",  # lacivert       (Tol muted)
    "#DDCC77",  # kum            (Tol muted)
    "#66CCEE",  # camgöbeği      (Tol bright)
    "#999933",  # zeytin         (Tol muted)
)


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    """#RRGGBB -> (B, G, R); OpenCV kanal sırası."""
    r, g, b = (int(value[i:i + 2], 16) for i in (1, 3, 5))
    return (b, g, r)


_PALETTE = tuple(_hex_to_bgr(h) for h in _PALETTE_HEX)


def build_colour_map(labels) -> dict[int, tuple[int, int, int]]:
    """Etiket -> BGR. Aynı anda gösterilen organlar ASLA aynı rengi almaz.

    Eski davranış rengi `etiket % palet_boyu` ile seçiyordu; etiket numaraları
    seyrek olduğu için bu, birlikte gösterilen organlarda çakışıyordu — bu
    projenin seçiminde iliopsoas_left (88) aorta (52) ile, iliopsoas_right (89)
    karaciğer (5) ile aynı renge düşüyordu, yani QC görüntüsü iki organı ayırt
    edilemez kılıyordu.

    Renk artık etiketin SIRASINA göre veriliyor: gösterilen küme sabit olduğu
    sürece (config.TARGET_ORGANS) her seride aynı organ aynı rengi alır, ki
    seriler arası gözle karşılaştırma mümkün kalsın.

    Etiket sayısı paletten büyükse (117 yapı) renkler dönmek zorunda; o durumda
    çakışma kaçınılmazdır ve organ adı açıklama şeridinden okunur.
    """
    return {int(lab): _PALETTE[i % len(_PALETTE)]
            for i, lab in enumerate(sorted(int(x) for x in labels))}


def _to_display(slice_hu: np.ndarray) -> np.ndarray:
    """HU dilimini pencereleyip 3 kanallı gri görüntüye çevirir."""
    import cv2

    low = _WINDOW_LEVEL - _WINDOW_WIDTH / 2.0
    high = _WINDOW_LEVEL + _WINDOW_WIDTH / 2.0
    clipped = np.clip(slice_hu, low, high)
    scaled = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


def _draw_labels(canvas: np.ndarray, seg_slice: np.ndarray,
                 colours: dict[int, tuple[int, int, int]] | None = None,
                 alpha: float = 0.40) -> list[int]:
    """Her etiketi renkle doldurur ve sınırını çizer; çizilen etiketleri döndürür.

    Dolgu yarı saydam: altındaki CT görünmeye devam ediyor, yoksa maskenin doğru
    dokunun üstünde olup olmadığı — kontrolün asıl sorusu — görülemiyor. Sınır
    çizgisi tam opak kalıyor ki komşu organlar birbirinden ayırt edilebilsin.
    """
    import cv2

    labels = sorted(int(v) for v in np.unique(seg_slice) if v != 0)
    if not labels:
        return []
    if colours is None:
        colours = build_colour_map(labels)

    fill = np.zeros_like(canvas)
    drawn: list[int] = []
    outlines: list[tuple[list, tuple[int, int, int]]] = []

    for label in labels:
        mask = (seg_slice == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        colour = colours.get(label, _PALETTE[label % len(_PALETTE)])
        fill[mask.astype(bool)] = colour
        outlines.append((contours, colour))
        drawn.append(label)

    if not drawn:
        return []

    # Yalnızca etiketli pikselleri harmanla; arka plan CT'si olduğu gibi kalsın.
    painted = seg_slice != 0
    canvas[painted] = cv2.addWeighted(canvas, 1.0 - alpha, fill, alpha, 0.0)[painted]

    for contours, colour in outlines:
        cv2.drawContours(canvas, contours, -1, colour, 1)
    return drawn


def _pick_axial_slices(seg: np.ndarray, count: int = 3,
                       min_gap_fraction: float = 0.12,
                       min_share_of_best: float = 0.4) -> list[int]:
    """En çok organ içeren, birbirinden yeterince uzak dilimleri seçer.

    İki şart birlikte gerekli. Sadece "en kalabalık üç dilim" demek üçünü de üst
    üste getiriyor, o yüzden aralarına mesafe şartı var. Ama tek başına mesafe
    şartı da kontenjanı doldurmak için hacmin ucundaki neredeyse boş bir dilimi
    seçiyordu — iki organlı bir kesit kontrol için işe yaramıyor. Bu yüzden bir
    dilim, en iyi dilimin organ sayısının belli bir oranına ulaşmıyorsa hiç
    alınmıyor: üç yerine iki iyi panel, üçüncüsü boş olan üçlüden yeğdir.
    """
    per_slice = [len(np.unique(seg[z])) - (1 if (seg[z] == 0).any() else 0)
                 for z in range(seg.shape[0])]
    best = max(per_slice, default=0)
    if best <= 0:
        return []

    threshold = max(1, int(best * min_share_of_best))
    min_gap = max(1, int(seg.shape[0] * min_gap_fraction))
    chosen: list[int] = []
    for z in sorted(range(len(per_slice)), key=lambda i: -per_slice[i]):
        if per_slice[z] < threshold:
            break
        if all(abs(z - taken) >= min_gap for taken in chosen):
            chosen.append(z)
        if len(chosen) == count:
            break
    return sorted(chosen)


def find_detached_blocks(seg: np.ndarray, min_gap: int = 20
                         ) -> list[tuple[int, int]]:
    """Maskenin birbirinden kopuk z blokları.

    Anatomi tek bir sürekli blokta olmalı. Bu veri setinde fantomun anatomik
    kısmı ve desen bölümü aynı seride: ölçüldü, TotalSegmentator desen
    bölümündeki 61.965 vokseli `colon` sanıyor. Orada organ yok, yani o
    voksellerin öznitelikleri anlamsız — ve bu ancak gözle bakınca fark
    ediliyordu. Kopuk bir blok varsa QC görüntüsü artık bunu yazıyor.

    İkiden fazla blok döndüyse en büyüğü dışındakiler şüphelidir.
    """
    occupied = np.flatnonzero(seg.astype(bool).any(axis=(1, 2)))
    if occupied.size == 0:
        return []

    blocks: list[tuple[int, int]] = []
    start = previous = int(occupied[0])
    for index in occupied[1:]:
        index = int(index)
        if index - previous > min_gap:
            blocks.append((start, previous))
            start = index
        previous = index
    blocks.append((start, previous))
    return blocks


def _segmented_z_range(seg: np.ndarray, margin_fraction: float = 0.05
                       ) -> tuple[int, int]:
    """Maskenin bulunduğu z aralığı, biraz payla.

    Hacim çoğu zaman organların çok ötesine uzanıyor (bu veri setinde fantomun
    desen bölümü de aynı seride). Koronal kesiti bütün hacim boyunca göstermek
    panelin yarısını kontrol edilecek hiçbir şeyin olmadığı bir alana harcıyor.
    """
    occupied = np.flatnonzero(seg.astype(bool).any(axis=(1, 2)))
    if occupied.size == 0:
        return 0, seg.shape[0]
    margin = max(1, int(seg.shape[0] * margin_fraction))
    return (max(0, int(occupied[0]) - margin),
            min(seg.shape[0], int(occupied[-1]) + margin + 1))


def _label_strip(width: int, labels: list[int], label_map: dict[str, str],
                 colours: dict[int, tuple[int, int, int]] | None = None,
                 row_height: int = 18) -> np.ndarray:
    """Renk → organ adı açıklaması."""
    import cv2

    if colours is None:
        colours = build_colour_map(labels)
    per_row = max(1, width // 190)
    rows = (len(labels) + per_row - 1) // per_row
    strip = np.full((max(row_height, rows * row_height), width, 3), 30, np.uint8)

    for index, label in enumerate(labels):
        row, column = divmod(index, per_row)
        x = 8 + column * (width // per_row)
        y = row * row_height + row_height - 6
        colour = colours.get(label, _PALETTE[label % len(_PALETTE)])
        cv2.rectangle(strip, (x, y - 9), (x + 10, y - 1), colour, -1)
        name = label_map.get(str(label), f"label_{label}")[:22]
        cv2.putText(strip, name, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return strip


def _annotate(canvas: np.ndarray, *lines: str,
              colour: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Panelin üstüne satır satır yazı basar.

    Metin ASCII: OpenCV'nin Hershey fontları Türkçe karakterleri basamıyor ve
    bunları '?' olarak çiziyor. Kod içi açıklamalar Türkçe, görüntüye gömülen
    yazı İngilizce — okunabilir olması gömülü yazının tek işi.

    Uzun uyarılar ayrı satıra düşüyor; tek satıra sıkıştırılınca panelin
    dışına taşıyordu.
    """
    import cv2

    for index, text in enumerate(lines):
        y = 16 + index * 18
        cv2.putText(canvas, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
    return canvas


def _with_legend(canvas: np.ndarray, drawn: list[int], label_map: dict[str, str],
                 colours: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Panelin altına yalnız O PANELDE çizilen etiketlerin açıklamasını ekler."""
    legend = _label_strip(canvas.shape[1], sorted(set(drawn)), label_map, colours)
    return np.vstack([canvas, legend])


def build_panels(ct: np.ndarray, seg: np.ndarray,
                 label_map: dict[str, str],
                 keep_labels: set[int] | None = None,
                 all_slices: bool = False
                 ) -> list[tuple[str, np.ndarray]] | None:
    """Her kesit için AYRI bir QC paneli: [(ad, görüntü), ...].

    ct ve seg aynı ızgarada, (z, y, x) sırasında beklenir.

    Paneller tek bir montaja yapıştırılmıyor. Yan yana dizilince her panel
    ekranda küçülüyor ve kontrolün asıl sorusu — şu organın maskesi doğru
    dokunun üstünde mi — çözünürlük yetmediği için görünmez hâle geliyordu.
    Ayrı dosyalar tam çözünürlükte açılıp yakınlaştırılabiliyor.

    Her panel KENDİ açıklama şeridini taşır: bir kesitte görünmeyen organın
    adını listelemek, o organın orada olduğu izlenimi veriyordu.

    `keep_labels` verilirse yalnız o etiketler çizilir (None → hepsi). Öznitelik
    çıkarımı bir organ alt kümesiyle koşuyorsa görsel kontrol de onlara
    odaklanmalı. Kesit seçimi de filtrelenmiş maskeye göre yapılır, yani
    paneller hedef organların bulunduğu yerden geçer.
    """
    if ct.shape != seg.shape:
        return None
    if keep_labels is not None:
        seg = np.where(np.isin(seg, list(keep_labels)), seg, 0)
    if not seg.any():
        return None

    panels: list[tuple[str, np.ndarray]] = []

    # Renk haritası TÜM paneller için bir kez kurulur. Kaynağı, panelde ne
    # çizildiği değil GÖSTERİLEN KÜME: böylece bir organ bir kesitte yoksa
    # diğer organların rengi kaymaz ve seriler arası karşılaştırma bozulmaz.
    colours = build_colour_map(
        keep_labels if keep_labels is not None
        else (int(v) for v in np.unique(seg) if v != 0))

    # `all_slices`: maskenin bulunduğu HER kesit için bir panel. Temsilî üç
    # kesit hızlı bir bakış için yeterli ama bir sınırın nerede kaydığını
    # göstermiyor; kesit kesit incelemek gerektiğinde bu kullanılır. Maskesiz
    # kesitler atlanır — üstünde kontrol edilecek hiçbir şey yok.
    if all_slices:
        z_list = [int(z) for z in np.flatnonzero(seg.any(axis=(1, 2)))]
    else:
        z_list = _pick_axial_slices(seg)

    for z in z_list:
        canvas = _to_display(ct[z])
        drawn = _draw_labels(canvas, seg[z], colours)
        canvas = _annotate(canvas, f"axial z={z}")
        panels.append((f"axial_z{z:04d}",
                       _with_legend(canvas, drawn, label_map, colours)))

    # Koronal: en çok etiketli satırdan geçen kesit. Eksenel görüntüler bir
    # organın kraniokaudal uzanımını göstermiyor; kayma en çok orada belli oluyor.
    coronal_index = int(np.argmax(seg.astype(bool).sum(axis=(0, 2))))
    z0, z1 = _segmented_z_range(seg)
    canvas = _to_display(ct[z0:z1, coronal_index, :][::-1])
    drawn = _draw_labels(canvas, seg[z0:z1, coronal_index, :][::-1], colours)
    lines = [f"coronal y={coronal_index}"]

    blocks = find_detached_blocks(seg)
    if len(blocks) > 1:
        largest = max(blocks, key=lambda b: b[1] - b[0])
        stray = [b for b in blocks if b is not largest]
        lines.append("WARNING: mask split into "
                     f"{len(blocks)} detached blocks")
        lines.append("stray at z=" + ", ".join(f"{a}-{b}" for a, b in stray))
    canvas = _annotate(canvas, *lines,
                       colour=(90, 220, 255) if len(blocks) > 1
                       else (255, 255, 255))
    panels.append((f"coronal_y{coronal_index:04d}",
                   _with_legend(canvas, drawn, label_map, colours)))

    return panels or None


def write_overlays(ct_path: Path, seg_path: Path, label_map: dict[str, str],
                   out_dir: Path,
                   keep_labels: set[int] | None = None,
                   all_slices: bool = False) -> list[Path]:
    """Her kesiti AYRI bir PNG olarak `out_dir` altına yazar.

    Dönen liste yazılan dosyalardır; boş liste = üretilemedi. Başarısızlık
    çıkarımı durdurmaz — QC bir kolaylıktır.
    """
    try:
        import cv2
        import SimpleITK as sitk
    except Exception as exc:  # noqa: BLE001
        print(f"[qc] UYARI: görselleştirme kütüphanesi yok ({exc}); atlandı.")
        return []

    written: list[Path] = []
    try:
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
        panels = build_panels(ct, seg, label_map, keep_labels, all_slices)
        if not panels:
            print(f"[qc] {out_dir.name}: maske boş ya da geometri uyuşmuyor.")
            return []

        out_dir.mkdir(parents=True, exist_ok=True)
        for name, image in panels:
            out_path = out_dir / f"{name}.png"
            tmp = out_path.with_name(out_path.name + ".tmp.png")
            if not cv2.imwrite(str(tmp), image):
                continue
            tmp.replace(out_path)          # atomik: yarım PNG görünmez
            written.append(out_path)
        return written
    except Exception as exc:  # noqa: BLE001
        # Görsel kontrol bir kolaylık; üretilememesi çıkarımı geçersiz kılmaz.
        print(f"[qc] UYARI: {out_dir.name} QC görüntüleri üretilemedi ({exc}).")
        return written
