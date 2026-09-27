"""Segmentasyonun görsel kontrolü.

QC görüntüsü çıkarıma girmiyor, ama işe yaraması onun doğru şeyi göstermesine
bağlı: boş bir panel ya da fark edilmeyen bir uyarı, kontrolü hiç yapmamakla
aynı kapıya çıkıyor.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import overlay_qc as qc


def _volume(shape=(60, 48, 48)) -> tuple[np.ndarray, np.ndarray]:
    """Bilinen bir CT + maske çifti: ortada bir küre, üstte ikinci bir yapı."""
    ct = np.full(shape, -1000.0, dtype=np.float32)
    seg = np.zeros(shape, dtype=np.uint8)

    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    body = ((yy - shape[1] / 2) ** 2 + (xx - shape[2] / 2) ** 2) <= 18 ** 2
    ct[body] = 40.0

    core = (((zz - 20) ** 2 + (yy - shape[1] / 2) ** 2
             + (xx - shape[2] / 2) ** 2) <= 9 ** 2)
    seg[core] = 1
    seg[(zz > 15) & (zz < 26) & (yy > 30) & (yy < 38) & (xx > 10) & (xx < 20)] = 5
    return ct, seg


LABEL_MAP = {"1": "liver", "5": "spleen"}


def test_panels_are_separate_images_each_with_a_legend():
    ct, seg = _volume()
    panels = qc.build_panels(ct, seg, LABEL_MAP)

    assert panels is not None and len(panels) >= 2
    names = [name for name, _ in panels]
    # Eksenel kesitler ve koronal AYRI paneller — tek montaj değil.
    assert any(n.startswith("axial_z") for n in names)
    assert sum(n.startswith("coronal_y") for n in names) == 1
    assert len(set(names)) == len(names), "panel adları benzersiz olmalı"
    for name, image in panels:
        assert image.ndim == 3 and image.shape[2] == 3
        # Her panelin altında KENDİ açıklama şeridi olmalı. Şeridin zemini
        # sabit (30,30,30); en alt satırın çoğunlukla o zemin olması, şeridin
        # eklendiğini panel yüksekliğinden bağımsız olarak doğrular (koronal
        # panelin yüksekliği maskenin z aralığına bağlı, sabit değil).
        assert (image[-1] == 30).all(axis=1).mean() > 0.5, name
    # Eksenel panel: tam kesit + şerit olduğu için kesitten yüksek olmalı.
    axial = [img for n, img in panels if n.startswith("axial_z")]
    assert all(img.shape[0] > ct.shape[1] for img in axial)


def test_each_panel_legend_lists_only_what_that_panel_shows():
    """Bir kesitte görünmeyen organı listelemek, orada olduğu izlenimi verir."""
    ct, seg = _volume()
    seg_one = np.zeros_like(seg)
    seg_one[seg == 1] = 1                      # yalnız 'liver' bırak
    panels = qc.build_panels(ct, seg_one, LABEL_MAP)

    assert panels is not None
    strip_height = panels[0][1].shape[0] - ct.shape[1]
    # Tek etiket → tek satırlık şerit; iki etiketli şeritten yüksek olmamalı.
    full = qc.build_panels(ct, seg, LABEL_MAP)
    assert full is not None
    assert strip_height <= full[0][1].shape[0] - ct.shape[1]


def test_labels_are_filled_not_only_outlined():
    """İçi boyalı olmalı: sadece kontur, maskenin içindeki dokuyu göstermiyor."""
    ct, seg = _volume()
    z = 20
    canvas = qc._to_display(ct[z])
    before = canvas.copy()
    qc._draw_labels(canvas, seg[z])

    painted = seg[z] != 0
    changed = (canvas != before).any(axis=2)
    # Maskenin İÇİ de değişmiş olmalı, yalnızca sınırı değil.
    interior = painted & ~_boundary(painted)
    assert interior.any(), "test hacmi çok ince, iç piksel yok"
    assert changed[interior].mean() > 0.9


def test_fill_is_translucent_so_the_ct_still_shows():
    """Tam opak dolgu altındaki dokuyu gizler; kontrolün sorusu tam da o."""
    ct, seg = _volume()
    z = 20
    canvas = qc._to_display(ct[z])
    qc._draw_labels(canvas, seg[z])

    painted = seg[z] != 0
    colour = np.array(qc.build_colour_map([1, 5])[1], dtype=np.uint8)
    exact = (canvas[painted] == colour).all(axis=1)
    assert exact.mean() < 0.5, "dolgu opak görünüyor, CT tamamen kapanmış"


def test_selected_organs_get_distinct_colours():
    """Güncel seçim — dalak (1) ve aorta (52) — ayrı renk almalı."""
    colours = qc.build_colour_map([1, 52])
    assert len(set(colours.values())) == 2
    # Aynı küme her seride aynı rengi vermeli (seriler arası karşılaştırma).
    assert qc.build_colour_map([52, 1]) == colours


def test_colour_map_does_not_collide_on_sparse_labels():
    """GERÇEK ARIZA: renk `etiket % palet` ile seçiliyordu; etiket numaraları
    seyrek olduğu için birlikte gösterilen organlarda çakışıyordu (88 ile 52,
    89 ile 5 aynı renge düşüyordu). Seçim genişletilirse tekrar etmesin."""
    seyrek = [1, 5, 52, 88, 89, 104, 116]
    colours = qc.build_colour_map(seyrek)
    assert len(set(colours.values())) == len(seyrek), "renkler çakışıyor"


def test_palette_entries_are_distinct_and_visible_on_grey():
    assert len(set(qc._PALETTE)) == len(qc._PALETTE), "palette tekrar eden renk var"
    for bgr in qc._PALETTE:
        # Gri CT üzerinde seçilebilmesi için ne çok koyu ne de doygunluksuz.
        assert max(bgr) - min(bgr) > 40 or max(bgr) > 200, bgr


def test_background_ct_is_untouched():
    ct, seg = _volume()
    z = 20
    canvas = qc._to_display(ct[z])
    before = canvas.copy()
    qc._draw_labels(canvas, seg[z])

    outside = seg[z] == 0
    # Sınır çizgisi bir piksel dışarı taşabilir; genelinde değişmemiş olmalı.
    assert (canvas[outside] == before[outside]).all(axis=1).mean() > 0.95


def _boundary(mask: np.ndarray) -> np.ndarray:
    import cv2
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    return mask & ~eroded.astype(bool)


# --------------------------------------------------------------------------- #
# Dilim seçimi
# --------------------------------------------------------------------------- #

def test_axial_slices_are_spread_apart():
    ct, seg = _volume()
    chosen = qc._pick_axial_slices(seg)
    assert chosen == sorted(chosen)
    if len(chosen) > 1:
        gaps = np.diff(chosen)
        assert gaps.min() >= max(1, int(seg.shape[0] * 0.12))


def test_nearly_empty_slices_are_not_chosen_just_to_fill_the_quota():
    """Kontenjanı doldurmak için iki organlı bir kesit seçmek kontrolü boşa
    çıkarıyordu — gerçek veride z=7'de tam olarak bu oluyordu.

    Kurgu gerçek veriyi taklit ediyor: zengin bölgede dilim başına ~10 ayrı
    etiket, uçta ise yalnızca bir tanesi. Etiketler yan yana yerleştiriliyor,
    üst üste değil; iç içe yazınca sonuncusu diğerlerini eziyor ve dilim başına
    etiket sayısı yapay olarak düşüyor.
    """
    seg = np.zeros((100, 40, 40), dtype=np.uint8)
    for index in range(10):                       # zengin bölge: 10 ayrı organ
        column = 2 + index * 4
        seg[40:60, 5:35, column:column + 3] = index + 1
    seg[95, 1:4, 1:4] = 7                         # uçta tek etiketli dilim

    per_slice = [len(np.unique(seg[z])) - 1 for z in range(100)]
    assert max(per_slice) == 10 and per_slice[95] == 1

    chosen = qc._pick_axial_slices(seg)
    assert 95 not in chosen
    assert all(40 <= z < 60 for z in chosen)


def test_no_slices_when_mask_is_empty():
    assert qc._pick_axial_slices(np.zeros((10, 5, 5), np.uint8)) == []
    assert qc.build_panels(np.zeros((10, 5, 5), np.float32),
                           np.zeros((10, 5, 5), np.uint8), {}) is None


def test_mismatched_geometry_is_refused():
    ct = np.zeros((10, 5, 5), np.float32)
    seg = np.ones((8, 5, 5), np.uint8)
    assert qc.build_panels(ct, seg, {}) is None


# --------------------------------------------------------------------------- #
# Kopuk blok uyarısı
# --------------------------------------------------------------------------- #

def test_contiguous_mask_reports_one_block():
    _, seg = _volume()
    assert len(qc.find_detached_blocks(seg)) == 1


def test_detached_mask_is_detected():
    """Gerçek veride ölçüldü: TotalSegmentator fantomun desen bölümündeki
    61.965 vokseli `colon` sanıyor. Anatomi tek blokta olmalı."""
    seg = np.zeros((200, 10, 10), dtype=np.uint8)
    seg[10:60, 2:8, 2:8] = 1            # anatomi
    seg[150:170, 2:8, 2:8] = 5          # desen bölümündeki sahte etiket

    blocks = qc.find_detached_blocks(seg)
    assert len(blocks) == 2
    assert blocks[0][0] == 10 and blocks[1][0] == 150


def test_small_gaps_do_not_count_as_detached():
    """Bir organın birkaç dilimde görünmemesi normal; eşik onu tolere etmeli."""
    seg = np.zeros((100, 10, 10), dtype=np.uint8)
    seg[10:40, 2:8, 2:8] = 1
    seg[45:70, 2:8, 2:8] = 1            # 5 dilimlik boşluk
    assert len(qc.find_detached_blocks(seg)) == 1


@pytest.mark.slow
def test_real_series_overlays_are_written_as_separate_files(seg_series, tmp_path):
    name, ct_path, seg_path, label_map_path = seg_series[0]
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

    written = qc.write_overlays(ct_path, seg_path, label_map, tmp_path / name)
    assert len(written) >= 2, "eksenel ve koronal ayrı dosyalar olmalı"
    assert all(p.is_file() and p.stat().st_size > 5_000 for p in written)
    assert len({p.name for p in written}) == len(written)
    assert any(p.name.startswith("coronal_y") for p in written)
    assert any(p.name.startswith("axial_z") for p in written)
    # Yarım/geçici dosya bırakılmamalı.
    assert not list((tmp_path / name).glob("*.tmp.png"))


def test_write_overlays_survives_a_missing_input(tmp_path):
    """QC bir kolaylık; üretilememesi çıkarımı durdurmamalı."""
    assert qc.write_overlays(tmp_path / "yok.nii.gz", tmp_path / "yok2.nii.gz",
                             {}, tmp_path / "out") == []
