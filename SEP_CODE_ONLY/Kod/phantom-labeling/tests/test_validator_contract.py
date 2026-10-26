"""Doğrulayıcı sözleşmesi — desen doğrulaması gerçekten desene mi bakıyor?

Bağlam: 336 serilik PCCT koşusunda 443 uyarı çıktı ve hepsi STAR8
bölümündendi. Dışa aktarılan ROI patch'lerinin %100'ü ölçüldüğünde
yerleşimin doğru olduğu, uyarıların doğrulayıcı kusuru olduğu görüldü:

  * ``_validate_honeycomb`` açısal tepe sayımını SABİT bir frekans
    bandında yapıyordu; ince petek kartlarının temel örgü frekansı bandın
    dışında kalınca ``n_peaks=2`` okunuyor ve skor tam olarak ``1-4/6 =
    0.333`` çıkıyordu.
  * ``_validate_sinusoidal`` 1B profile ``curve_fit`` ile tek bir sinüs
    oturtup R²'ye bakıyordu. Uyarı alan ve almayan patch'ler ölçüldüğünde
    aynı 2B kontrastı (std 321 / 326 HU) ve aynı FFT başlangıç frekansını
    (bin 5 / bin 7) taşıyordu; fark yalnızca optimizasyonun uyarılı olanda
    sahte bir yerel minimuma kaymasıydı (f 0.898 -> 0.761, R² 0.998 -> 0.055).

Buradaki testler bu iki hata modunu SENTETİK desenlerle sabitliyor: örgü
adımı değiştiğinde doğrulayıcı hâlâ çalışmalı, ve gerçekten yanlış bir
desen hâlâ reddedilmeli (aksi halde "düzeltme" sadece kapıyı açmak olur).
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# sentetik desen üreticileri
# --------------------------------------------------------------------------- #

def hex_lattice(size: int = 66, pitch: float = 8.0, hu_lo: float = -900.0,
                hu_hi: float = 100.0) -> np.ndarray:
    """Altıgen (petek) örgü — 6-kat simetri.

    ``pitch`` örgü adımı: küçük değer = ince petek (Honeysmall), büyük
    değer = kaba petek (Honeylarge). Doğrulayıcı ikisinde de çalışmalı.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    k = 2.0 * np.pi / pitch
    # Üç adet 60° arayla yerleştirilmiş düzlem dalgası = altıgen örgü.
    field = np.zeros((size, size))
    for ang in (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0):
        field += np.cos(k * (xx * np.cos(ang) + yy * np.sin(ang)))
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    return hu_lo + field * (hu_hi - hu_lo)


def grating(size: int = 66, pitch: float = 8.0, angle_deg: float = 0.0,
            square_wave: bool = False) -> np.ndarray:
    """Tek yönlü çizgi deseni (2-kat). ``square_wave`` bar deseni üretir."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    t = np.radians(angle_deg)
    phase = 2.0 * np.pi / pitch * (xx * np.cos(t) + yy * np.sin(t))
    w = np.sign(np.sin(phase)) if square_wave else np.sin(phase)
    return -400.0 + w * 500.0


def chirp_grating(size: int = 66, p0: float = 5.0, p1: float = 12.0) -> np.ndarray:
    """Frekansı yavaşça değişen sinüs — fantomun Sinusoidal kartı.

    Global tek-sinüs fit'i bu desende düşük R² verir; doğru doğrulayıcı
    yine de "bu bir sinüzoidal dalga" demelidir.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    pitch = p0 + (p1 - p0) * (xx / max(size - 1, 1))
    phase = np.cumsum(2.0 * np.pi / pitch, axis=1)
    return -400.0 + np.sin(phase) * 500.0


def uniform_plastic(size: int = 66, hu: float = -350.0,
                    noise: float = 6.0, seed: int = 0) -> np.ndarray:
    """Desensiz düz malzeme — ROI karttan kaymışsa görülecek şey."""
    rng = np.random.default_rng(seed)
    return hu + rng.normal(0.0, noise, (size, size))


# --------------------------------------------------------------------------- #
# petek: örgü adımından bağımsız olmalı
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pitch,card", [(4.5, "Honeysmall"),
                                        (7.0, "Honeymedium"),
                                        (12.0, "Honeylarge")])
def test_honeycomb_accepted_at_every_pitch(texture_validator, pitch, card):
    """Asıl regresyon: eski kod ince peteği 0.333 verip reddediyordu."""
    ok, method, score = texture_validator.validate_texture(hex_lattice(pitch=pitch), card)
    assert ok, f"{card} (pitch={pitch}) reddedildi: {method} skor={score:.3f}"


def test_honeycomb_rejects_a_line_grating(texture_validator):
    """Kapı açılmamalı: tek yönlü çizgi deseni petek DEĞİLDİR."""
    ok, _, _ = texture_validator.validate_texture(grating(pitch=7.0), "Honeymedium")
    assert not ok


def test_honeycomb_rejects_featureless_plastic(texture_validator):
    """ROI karttan kayarsa düz malzeme görülür — bu reddedilmeli."""
    ok, _, _ = texture_validator.validate_texture(uniform_plastic(), "Honeylarge")
    assert not ok


# --------------------------------------------------------------------------- #
# sinüzoidal: chirp kabul, bar deseni ret
# --------------------------------------------------------------------------- #

def test_sinusoidal_accepts_a_chirped_wave(texture_validator):
    ok, method, score = texture_validator.validate_texture(chirp_grating(), "Sinusoidal")
    assert ok, f"chirp reddedildi: {method} skor={score:.3f}"


def test_sinusoidal_accepts_a_pure_sine(texture_validator):
    ok, _, _ = texture_validator.validate_texture(grating(pitch=9.0), "Sinusoidal")
    assert ok


def test_sinusoidal_rejects_a_square_bar_pattern(texture_validator):
    """Kare dalga güçlü harmonikler taşır; sinüs kartı ile karışmamalı."""
    ok, _, _ = texture_validator.validate_texture(
        grating(pitch=9.0, square_wave=True), "Sinusoidal")
    assert not ok


def test_sinusoidal_rejects_featureless_plastic(texture_validator):
    ok, _, _ = texture_validator.validate_texture(uniform_plastic(), "Sinusoidal")
    assert not ok


# --------------------------------------------------------------------------- #
# STAR16 kartları: değişmemiş yollar hâlâ çalışıyor mu
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("card,angle", [("Vertical", 0.0), ("Horizontal", 90.0)])
def test_line_cards_still_validate(texture_validator, card, angle):
    ok, _, _ = texture_validator.validate_texture(
        grating(pitch=8.0, angle_deg=angle), card)
    assert ok


# --------------------------------------------------------------------------- #
# yıldız: kol sayısı doğrudan patch'ten okunmalı
# --------------------------------------------------------------------------- #

def star_patch(size: int = 68, arms: int = 16, hu_lo: float = -950.0,
               hu_hi: float = 50.0, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """N kollu yıldız; ``dx/dy`` merkezi kaydırır (hizalama dayanıklılığı)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cy = cx = (size - 1) / 2.0
    t = np.arctan2(yy - cy - dy, xx - cx - dx)
    r = np.hypot(xx - cx - dx, yy - cy - dy)
    wedge = (np.cos(arms * t) > 0).astype(np.float64)
    patch = hu_lo + wedge * (hu_hi - hu_lo)
    patch[r > (size / 2.0 - 1)] = np.nan          # daire dışı = maske
    return patch


@pytest.mark.parametrize("arms", [8, 16])
def test_star_order_read_from_patch(validator, arms):
    assert validator._dominant_rotational_order(star_patch(arms=arms))[0] == arms


def test_miscentring_never_turns_16_arms_into_8(validator):
    """Kayma tehlikeli KARIŞIKLIĞI üretmemeli.

    Merkez kaydığında N-kat desenin gücü N±1'e sızar — üretimde bu olmaz
    (patch rafine merkeze göre kesiliyor), ama olduğunda sonuç 15/17 gibi
    GEÇERSİZ bir dereceye kayıp uyarı üretmeli; asla "8" gibi geçerli ama
    YANLIŞ bir yıldıza dönüşmemeli. Sessiz yanlış etiket, gürültülü uyarıdan
    çok daha kötüdür.
    """
    for dx, dy in ((1.5, -1.0), (-2.0, 0.5), (0.0, 2.5)):
        order, _ = validator._dominant_rotational_order(star_patch(arms=16, dx=dx, dy=dy))
        assert order != 8, f"16 kollu yıldız dx={dx},dy={dy} kaymasında 8 okundu"


def test_star_order_distinguishes_8_from_16(validator):
    """Asıl regresyon: classify_star 8 kollu yıldıza da STAR16 diyordu."""
    o8 = validator._dominant_rotational_order(star_patch(arms=8))[0]
    o16 = validator._dominant_rotational_order(star_patch(arms=16))[0]
    assert (o8, o16) == (8, 16)


def test_star_order_is_unsure_on_featureless_patch(validator):
    """Desensiz patch güvenli bir derece üretmemeli (rastgele bir sayıya
    kilitlenip 'doğrulandı' dememeli)."""
    patch = uniform_plastic(size=68, hu=-900.0, noise=5.0)
    _, ratio = validator._dominant_rotational_order(patch)
    assert ratio < validator.STAR_MIN_MODULATION
