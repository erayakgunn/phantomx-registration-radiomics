"""DICOM başlık sütunlarının öznitelik tablosuna taşınması.

Protokol karşılaştıran bir çalışmada bir satırın hangi çekimden geldiği
tablodan okunabilmeli; aksi halde her sorguda ham DICOM'a dönmek gerekiyor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import dicom_metadata
import excel_report
import feature_extraction_mirp as fx


def _write_slice(path: Path, **overrides):
    """Testlik tek dilim CT DICOM'u."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.Modality = "CT"
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Manufacturer = "Siemens Healthineers"
    ds.ManufacturerModelName = "NAEOTOM Alpha"
    ds.KVP = "120"
    ds.ConvolutionKernel = ["Qr40", "3"]          # MultiValue → tek metne inmeli
    ds.SliceThickness = "1.0"
    ds.PixelSpacing = [0.5859375, 0.5859375]
    ds.ReconstructionDiameter = "300"
    ds.SeriesDescription = "100kVp_100mAs_1mm_300FOV"
    for key, value in overrides.items():
        setattr(ds, key, value)

    ds.Rows = ds.Columns = 4
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((4, 4), dtype=np.int16).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def test_harvest_reads_the_priority_tags(tmp_path):
    _write_slice(tmp_path / "slice_0000.dcm")
    harvested = dicom_metadata.harvest(tmp_path, "slice_0000.dcm")

    assert harvested["Manufacturer"] == "Siemens Healthineers"
    assert harvested["KVP"] == "120"
    assert harvested["ReconstructionDiameter"] == "300"


def test_multivalue_tags_collapse_to_one_string(tmp_path):
    """ConvolutionKernel çoklu değer olabiliyor; hücreye tek metin girmeli."""
    _write_slice(tmp_path / "s.dcm")
    harvested = dicom_metadata.harvest(tmp_path, "s.dcm")

    assert harvested["ConvolutionKernel"] == "Qr40\\3"
    assert harvested["PixelSpacing"] == "0.5859375\\0.5859375"


def test_missing_tags_are_omitted_not_invented(tmp_path):
    """Olmayan etiket için boş değer uydurulmamalı — sütun hiç olmamalı."""
    _write_slice(tmp_path / "s.dcm")
    harvested = dicom_metadata.harvest(tmp_path, "s.dcm")

    assert "CTDIvol" not in harvested
    assert "" not in harvested.values()


def test_unreadable_source_yields_empty_dict(tmp_path):
    """DICOM okunamıyorsa çıkarım durmamalı; metadata yalnızca bağlam."""
    (tmp_path / "bozuk.dcm").write_bytes(b"not a dicom")
    assert dicom_metadata.harvest(tmp_path, "bozuk.dcm") == {}
    assert dicom_metadata.harvest(tmp_path / "yok") == {}


def test_columns_are_prefixed_and_ordered(tmp_path):
    """Önek çakışmayı önlüyor; sıra PRIORITY_TAGS sırası."""
    _write_slice(tmp_path / "s.dcm")
    columns = dicom_metadata.as_columns(dicom_metadata.harvest(tmp_path, "s.dcm"))

    assert all(name.startswith("dcm_") for name in columns)
    assert "dcm_KVP" in columns
    # MIRP'in image_* meta sütunlarıyla ve öznitelik adlarıyla çakışmamalı.
    assert not any(name.startswith("image_") for name in columns)

    order = list(columns)
    assert order.index("dcm_Manufacturer") < order.index("dcm_KVP")


def test_sidecar_is_cached_and_reused(tmp_path, monkeypatch):
    """İkinci çağrı kaynağa dönmemeli — DICOM'lar taşınsa da tablo üretilebilsin."""
    import config

    source = tmp_path / "src"
    _write_slice(source / "slice_0000.dcm")
    monkeypatch.setattr(config, "NIFTI", tmp_path / "nifti")

    meta = {"series_dir": str(source), "ordered_files": ["slice_0000.dcm"]}
    first = dicom_metadata.load_or_harvest("SERIES_A", meta)
    assert first["KVP"] == "120"

    sidecar = tmp_path / "nifti" / "SERIES_A" / "dicom_metadata.json"
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == first

    # Kaynak yok olsa bile önbellekten dönmeli.
    import shutil
    shutil.rmtree(source)
    assert dicom_metadata.load_or_harvest("SERIES_A", meta) == first


def test_finalize_places_dicom_columns_after_identity():
    frame = pd.DataFrame({"image_mask_label": ["liver"], "stat_mean": [42.0]})
    result = fx._finalize_series_frame(
        frame, "KEY", "P1", recon_type="native", kvp=120.0,
        dicom_columns={"dcm_KVP": "120", "dcm_Manufacturer": "Siemens"},
    )

    columns = list(result.columns)
    assert columns[:4] == ["series_key", "patient_id", "recon_type", "kvp"]
    assert set(columns[4:6]) == {"dcm_KVP", "dcm_Manufacturer"}
    assert columns.index("stat_mean") > columns.index("dcm_KVP")


def test_finalize_without_dicom_columns_is_unchanged():
    """Metadata yoksa tablo eskisi gibi olmalı — geriye dönük uyum."""
    frame = pd.DataFrame({"image_mask_label": ["liver"], "stat_mean": [42.0]})
    result = fx._finalize_series_frame(frame, "KEY", "P1")
    assert list(result.columns)[:4] == ["series_key", "patient_id", "recon_type", "kvp"]
    assert not [c for c in result.columns if c.startswith("dcm_")]


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame({c: ["x"] for c in (
        "series_key", "patient_id", "recon_type", "kvp", "image_mask_label",
        "dcm_StudyDate", "dcm_AcquisitionTime", "dcm_Modality", "dcm_KVP",
        "dcm_ConvolutionKernel", "dcm_StationName", "dcm_ImageType",
        "image_modality", "image_voxel_size_x", "stat_mean", "morph_volume",
    )})


def test_layout_assigns_every_column_to_exactly_one_block():
    frame = _sample_frame()
    layout = excel_report._layout(frame)

    assert [name for name, _ in layout] != []
    assert len(layout) == len(frame.columns)
    assert len({name for name, _ in layout}) == len(frame.columns)


def test_layout_block_membership():
    """Renk anlamı: yeşil=organ, sarı=kimlik+kalan DICOM, mavi=çekim koşulları."""
    blocks = dict(excel_report._layout(_sample_frame()))

    assert blocks["image_mask_label"] == excel_report.ORGAN
    # Kimlik sütunları sarı: satırı bulmaya yarar, analizde okunmaz.
    assert blocks["series_key"] == excel_report.BOOKKEEPING
    assert blocks["patient_id"] == excel_report.BOOKKEEPING
    # Çekim koşulları mavi — satırlar arası farkı bunlar yaratıyor.
    for column in ("recon_type", "kvp", "dcm_StudyDate", "dcm_AcquisitionTime",
                   "dcm_Modality", "dcm_KVP", "dcm_ConvolutionKernel"):
        assert blocks[column] == excel_report.ACQUISITION, column
    # Başlığın geri kalanı sarı.
    assert blocks["dcm_StationName"] == excel_report.DICOM
    assert blocks["dcm_ImageType"] == excel_report.DICOM
    assert blocks["image_modality"] == excel_report.MIRP_META
    assert blocks["stat_mean"] == excel_report.FEATURE


def test_acquisition_time_sits_next_to_the_date():
    """Bir seriyi zaman içinde konumlandırmak ikisini birlikte gerektiriyor."""
    order = [name for name, _ in excel_report._layout(_sample_frame())]
    assert order.index("dcm_AcquisitionTime") == order.index("dcm_StudyDate") + 1


def test_organ_is_the_first_column():
    order = [name for name, _ in excel_report._layout(_sample_frame())]
    assert order[0] == "image_mask_label"


def test_only_identifying_columns_are_frozen(tmp_path):
    """Çekim parametreleri de solda ve renkli, ama dondurulmuyor: 10 sütun daha
    sabitlemek ekranın yarısını yer ve asıl bakılan öznitelikleri görünmez kılar."""
    import openpyxl

    path = excel_report.write_workbook(_sample_frame(), tmp_path / "c.csv")
    ws = openpyxl.load_workbook(path)[f"{config.PRIMARY_MODE}_base"]

    # organ + series_key + patient_id = 3 sütun sabit.
    assert ws.freeze_panes == "D2"
    assert ws.cell(1, 1).value == "image_mask_label"


def test_workbook_writes_dicom_block_between_identity_and_features(tmp_path):
    import openpyxl

    frame = pd.DataFrame({
        "series_key": ["S1"], "patient_id": ["P1"], "recon_type": ["native"],
        "kvp": [120.0], "image_mask_label": ["liver"],
        "dcm_KVP": ["120"], "image_modality": ["ct"], "stat_mean": [42.0],
    })
    path = excel_report.write_workbook(frame, tmp_path / "combined.csv")
    assert path is not None

    ws = openpyxl.load_workbook(path)[f"{config.PRIMARY_MODE}_base"]
    header = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    assert header.index("dcm_KVP") > header.index("kvp")
    assert header.index("dcm_KVP") < header.index("stat_mean")
