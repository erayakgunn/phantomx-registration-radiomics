"""Etiketleme exporter testleri için ortak altyapı.

`codes/` bir paket değil, düz bir modül dizini; exporter `from core.roi_extractor
import ...` şeklinde import ettiği için dizinin sys.path'te olması gerekiyor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODES_DIR = Path(__file__).resolve().parent.parent / "codes"


def _labeling_root() -> Path:
    """Dizin ağacında kardeş etiketleme kopyalarının aranacağı kök.

    Üç ağaç `labeling/` altında yaşıyor, ama bu proje tek başına da
    kopyalanabiliyor. Sabit bir seviye saymak (parents[3]) bağımsız kopyada
    ev dizinine kadar çıkıp alakasız her şeyi tarardı; onun yerine `labeling`
    adlı atayı arıyoruz ve bulamazsak aramayı bu projeyle sınırlıyoruz.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if ancestor.name == "labeling":
            return ancestor
    # Tek başına kopyalanmış: aramayı bu projeyle sınırla. Kardeş arayan glob
    # `codes/`'un üstünde bir dizin daha beklediği için hiçbir şey bulamaz ve
    # karşılaştırma testi atlanır — bağımsız kopyada zaten anlamsız.
    return CODES_DIR.parent


LABELING_ROOT = _labeling_root()

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs a full run or the real dataset")


@pytest.fixture(scope="session")
def exporter():
    import output.exporter as module
    return module


@pytest.fixture(scope="session")
def roi_extractor():
    import core.roi_extractor as module
    return module


@pytest.fixture(scope="session")
def roi_cls():
    from core.roi_extractor import ROI
    return ROI


@pytest.fixture(scope="session")
def texture_validator():
    import analysis.texture_validator as module
    return module


@pytest.fixture(scope="session")
def validator():
    import core.validator as module
    return module


@pytest.fixture
def make_roi(roi_cls):
    """Minimal bir ROI nesnesi kurar."""
    def _make(pixels: np.ndarray, *, label="Insert_01_10mm", section="INSERT1",
              slice_idx=0, shape_type="circle", size_cm=1.0):
        return roi_cls(
            label=label,
            section=section,
            slice_idx=slice_idx,
            center_x=10.0,
            center_y=10.0,
            pixels=pixels,
            shape_type=shape_type,
            size_cm=size_cm,
            angle_deg=0.0,
        )
    return _make


def synthetic_ct_slice(height: int = 64, width: int = 64, *, base_hu: float = -1000.0,
                       disc_hu: float = 120.0, centre=None, radius: float = 8.0):
    """Bilinen içerikli tek bir CT dilimi (HU cinsinden)."""
    slice_hu = np.full((height, width), base_hu, dtype=np.float32)
    cy, cx = centre if centre is not None else (height / 2.0, width / 2.0)
    yy, xx = np.mgrid[0:height, 0:width]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    slice_hu[inside] = disc_hu
    return slice_hu
