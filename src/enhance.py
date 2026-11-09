"""Classical image enhancement methods (no deep learning).

Colour handling: OpenCV stores colour images as BGR. For three of the methods
we convert to CIE Lab, where channel L holds the brightness (luminance) and
channels a, b hold the colour. We change only L and leave a, b untouched, so
colours are not mixed up by the enhancement. In OpenCV's 8-bit Lab, L is
rescaled to 0..255.
"""
import cv2
import numpy as np

from . import config as cfg


def _apply_to_L(img_bgr, func):
    """Convert to Lab, apply `func` to the L channel only, convert back."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = func(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def baseline(img_bgr):
    """Method 1: no processing (just the degraded image)."""
    return img_bgr.copy()


def _stretch_channel(ch, low_pct, high_pct):
    lo, hi = np.percentile(ch, [low_pct, high_pct])
    if hi <= lo:  # flat channel: nothing to stretch
        return ch.copy()
    out = (ch.astype(np.float64) - lo) / (hi - lo) * 255.0
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def stretch_luminance(img_bgr, low_pct=cfg.STRETCH_LOW_PCT, high_pct=cfg.STRETCH_HIGH_PCT):
    """Method 2 (main): linear contrast stretching on the Lab luminance channel.

    The darkest 1% of pixels are mapped to 0 and the brightest 1% to 255;
    everything in between is rescaled linearly. Ignoring 1% at each end stops a
    few extreme (often noisy) pixels from controlling the whole mapping.
    """
    return _apply_to_L(img_bgr, lambda L: _stretch_channel(L, low_pct, high_pct))


def stretch_per_channel(img_bgr, low_pct=cfg.STRETCH_LOW_PCT, high_pct=cfg.STRETCH_HIGH_PCT):
    """Method 2b (supplementary): the same stretch on each B, G, R channel separately.

    Each channel gets its own scaling, so the balance between channels can
    change and colours may shift. Included only to discuss colour shifts.
    """
    chans = [_stretch_channel(img_bgr[:, :, c], low_pct, high_pct) for c in range(3)]
    return cv2.merge(chans)


def equalize_luminance(img_bgr):
    """Method 3: global histogram equalization on the Lab luminance channel.

    Equalization re-maps grey levels so that all levels are used about equally
    often. It flattens the histogram, which usually boosts contrast strongly.
    """
    return _apply_to_L(img_bgr, cv2.equalizeHist)


def estimate_gamma(img_bgr, target_mean=cfg.GAMMA_TARGET_MEAN):
    """Pick gamma so that the mean luminance of the image becomes `target_mean`.

    Uses only the degraded image (no access to the original reference).
    If mean^gamma = target  ->  gamma = log(target) / log(mean).
    """
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64) / 255.0
    mean = float(np.clip(L.mean(), 1e-3, 1 - 1e-3))
    gamma = np.log(target_mean) / np.log(mean)
    return float(np.clip(gamma, cfg.GAMMA_MIN, cfg.GAMMA_MAX))


def gamma_correction(img_bgr, gamma=None):
    """Method 4: gamma correction on the Lab luminance channel.

    out = 255 * (in / 255) ** gamma.  gamma < 1 brightens (lifts dark areas),
    gamma > 1 darkens. It is a non-linear curve, but it does NOT stretch the
    overall range of grey levels the way the other methods do.
    Returns (image, gamma_used).
    """
    if gamma is None:
        gamma = estimate_gamma(img_bgr)
    lut = np.array([255.0 * (i / 255.0) ** gamma for i in range(256)])
    lut = np.clip(np.round(lut), 0, 255).astype(np.uint8)
    return _apply_to_L(img_bgr, lambda L: cv2.LUT(L, lut)), gamma
