"""Aşama 4 — keşif ve muhasebe.

İki ayrı sessiz kayıp yolu buradan geçiyordu:

  * Keşif kapısı (`_has_insert_dicom_masks`) okuyucudan (`discover_insert_roi_dirs`)
    daha geçirgendi. Kapıyı geçip okuyucuya takılan bir seri işlenmeye alınıyor,
    sıfır insert ROI üretiyor ve HİÇBİR yere yazılmıyordu — reddedilenler
    listesine de, atlananlar listesine de. Sayılacak bir ROI hiç sayılamadığı
    için atlanan da yoktu.
  * Reddedilen klasörler ekrana basılıp muhasebeden düşüyordu, yani 80 klasörden
    77 seri çıkan bir çalıştırma exit 0 dönüyordu.

Buradaki testler ikisini de kilitliyor. Kapı artık okuyucuya delege ediyor;
aşağıdaki eşdeğerlik testi biri kapıyı yeniden ELLE yazdığı anda kırılır.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import make_patch_dicom


def _write_dcms(directory: Path, n: int = 3, size: int = 9) -> Path:
    for k in range(n):
        make_patch_dicom(directory / f"slice_{k:04d}.dcm",
                         np.full((size, size), 50.0), slice_index=k)
    return directory


def _series(root: Path, name: str, insert_rel: str | None) -> Path:
    """Bir seri klasörü: ham CT + istenen yola insert patch'leri."""
    series_dir = root / name
    _write_dcms(series_dir / "dicoms")
    if insert_rel is not None:
        _write_dcms(series_dir / insert_rel)
    return series_dir


# --------------------------------------------------------------------------- #
# Kapı ile okuyucu ayrışamaz
# --------------------------------------------------------------------------- #

# (etiket, insert patch'lerinin göreli yolu, okuyucu bulmalı mı)
LAYOUTS = [
    ("flat_5mm",            "Inserts/5mm",                    True),
    ("insert_nn_5mm",       "Inserts/5mm/Insert_01",          True),
    ("insert_nn_10mm",      "Inserts/10mm/Insert_02",         True),
    # Aşağıdakiler eskiden kapıyı geçip okuyucuda sıfır ROI veriyordu.
    ("nested_too_deep",     "Inserts/5mm/Insert_01/extra",    False),
    ("size_with_space",     "Inserts/5 mm/Insert_01",         False),
    ("size_with_underscore", "Inserts/5_mm/Insert_01",        False),
    # INSERT_SIZES dışındaki boyut: bilinçli olarak çıkarılmıyor.
    ("only_15mm",           "Inserts/15mm/Insert_01",         False),
    ("no_inserts_at_all",   None,                             False),
]


@pytest.mark.parametrize("label,rel,reader_finds", LAYOUTS,
                         ids=[row[0] for row in LAYOUTS])
def test_gate_never_admits_what_the_reader_cannot_read(
    pipeline, tmp_path, label, rel, reader_finds
):
    """Kapının cevabı okuyucunun cevabıyla birebir aynı olmalı.

    Kapı geçirgen olduğunda sonuç sessiz kayıptı: seri kabul ediliyor, insert
    üretmiyor, hiçbir raporda görünmüyor. Stars/Textures okunabiliyorsa
    çalıştırma exit 0 bile dönüyordu.
    """
    series_dir = _series(tmp_path, label, rel)

    gate = pipeline._has_insert_dicom_masks(series_dir)
    reader = bool(pipeline.discover_insert_roi_dirs(series_dir))

    assert reader is reader_finds, "test verisi beklenen düzeni kurmuyor"
    assert gate == reader, (
        f"{label}: kapı {gate} diyor, okuyucu {reader} buluyor — ayrışma geri "
        f"geldi, yani bir seri sessizce sıfır insert ROI ile işlenebilir"
    )


def test_only_insert_sizes_are_admitted(pipeline, tmp_path):
    """Diskteki 15mm klasörü seriyi tek başına ayakta tutmamalı."""
    series_dir = _series(tmp_path, "s15", "Inserts/15mm/Insert_01")
    assert not pipeline._has_insert_dicom_masks(series_dir)

    # 5mm eklenince kabul edilmeli, ama yalnızca 5mm okunmalı.
    _write_dcms(series_dir / "Inserts" / "5mm" / "Insert_01")
    assert pipeline._has_insert_dicom_masks(series_dir)
    sizes = {size for size, _, _ in pipeline.discover_insert_roi_dirs(series_dir)}
    assert sizes == {"5mm"}


# --------------------------------------------------------------------------- #
# Reddedilenler görünür ve gerekçeli
# --------------------------------------------------------------------------- #

def test_near_miss_rejection_names_the_layout_rule(pipeline, tmp_path):
    """Yakın ıskalama, "seri değil"den ayrı bir gerekçe almalı.

    Klasör adındaki bir yazım hatası her insert ROI'sine mal oluyor; gerekçe
    okuyucuya ne yapacağını söylemeli.
    """
    _series(tmp_path, "typo", "Inserts/5 mm/Insert_01")

    pipeline._REJECTED_SERIES.clear()
    found = pipeline.discover_series(tmp_path)

    assert found == [], "okunamayan insert düzeni seri olarak kabul edildi"
    assert len(pipeline._REJECTED_SERIES) == 1
    name, reason = pipeline._REJECTED_SERIES[0]
    assert name == "typo"
    assert "Inserts/ exists" in reason
    assert "match exactly" in reason


def test_absent_inserts_folder_gets_the_plain_reason(pipeline, tmp_path):
    """Inserts/ hiç yoksa bu bir yazım hatası değil, sadece seri değil."""
    _series(tmp_path, "plain", None)

    pipeline._REJECTED_SERIES.clear()
    assert pipeline.discover_series(tmp_path) == []
    _, reason = pipeline._REJECTED_SERIES[0]
    assert "no Inserts/" in reason
    assert "match exactly" not in reason


def test_a_readable_series_is_not_rejected(pipeline, tmp_path):
    """Kontrol testi: düzgün seri kabul edilmeli ve hiçbir ret üretmemeli."""
    _series(tmp_path, "good", "Inserts/5mm/Insert_01")

    pipeline._REJECTED_SERIES.clear()
    found = pipeline.discover_series(tmp_path)

    assert [s["series_name"] for s in found] == ["good"]
    assert pipeline._REJECTED_SERIES == []


# --------------------------------------------------------------------------- #
# Kaybedilen hiçbir şey exit 0 ile geçmez
# --------------------------------------------------------------------------- #

def test_clean_run_reports_no_loss(pipeline):
    assert pipeline._incompleteness_summary([], [], []) is None


def test_rejected_folders_alone_make_a_run_incomplete(pipeline):
    """ASIL DÜZELTME: eskiden reddedilenler basılıp muhasebeden düşüyordu.

    Çökme yok, atlanan ROI yok — ama çalışma kitabında bir seri eksik. Otomatik
    bir çağıran bunu tam bir çıkarımdan ayırt edemiyordu.
    """
    summary = pipeline._incompleteness_summary(
        [], [], [("KSW_..._300_1", "Inserts/ exists but holds no readable patches")]
    )
    assert summary is not None
    assert "1 rejected folder(s)" in summary


def test_rejected_folders_are_counted_once_each(pipeline):
    """_print_rejected_series tekilleştiriyor; sayı onunla uyuşmalı."""
    rejected = [("A", "no CT DICOM folder found"),
                ("A", "no CT DICOM folder found"),
                ("B", "no Inserts/5mm|10mm folder")]
    assert "2 rejected folder(s)" in pipeline._incompleteness_summary([], [], rejected)


def test_every_loss_channel_appears_in_the_summary(pipeline):
    summary = pipeline._incompleteness_summary(
        [("S1", "crashed")],
        [("S2", "Inserts", "Insert_03", "empty mask")],
        [("S3", "no Inserts/5mm|10mm folder")],
    )
    assert "1 rejected folder(s)" in summary
    assert "1 failed series" in summary
    assert "1 skipped ROI(s)" in summary
