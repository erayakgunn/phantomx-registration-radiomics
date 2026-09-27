"""Proje kökü conftest'i.

`test_pipeline.py` iki modda çalışacak şekilde yazılmış: pytest altında ve
`python test_pipeline.py` ile kendi bağımsız çalıştırıcısıyla. Bağımsız
çalıştırıcı `tmp_seg` argümanını `_make_tmp_seg()` ile kendisi enjekte ediyor
(bkz. `_run_standalone`), ama pytest tarafında aynı isimde bir fixture hiçbir
yerde tanımlı değildi — dolayısıyla `test_build_roi_map` pytest ile
çalıştırıldığında "fixture 'tmp_seg' not found" hatası veriyordu.

Buradaki fixture, testin kendi yardımcı fonksiyonunu yeniden kullanarak o boşluğu
kapatıyor; test dosyasına dokunulmuyor ve pipeline davranışı değişmiyor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


@pytest.fixture
def tmp_seg(tmp_path):
    """(seg.nii.gz yolu, mevcut etiketler) — test_pipeline.py'nin beklediği biçim."""
    from test_pipeline import _make_tmp_seg
    return _make_tmp_seg(tmp_path)
