"""Evaluation metrics: PSNR, SSIM, MAE (+ two helper measurements)."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr(reference, test):
    """Peak Signal-to-Noise Ratio in dB (higher = closer to the reference).

    PSNR = 10 * log10(255^2 / MSE), where MSE is the mean squared pixel error.
    """
    mse = np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(reference, test):
    """Structural Similarity (1.0 = identical), computed with scikit-image."""
    return float(structural_similarity(reference, test, channel_axis=2, data_range=255))


def mae(reference, test):
    """Mean Absolute Error in grey levels (lower = better)."""
    return float(np.mean(np.abs(reference.astype(np.float64) - test.astype(np.float64))))


def luminance_contrast(img_bgr):
    """Standard deviation of the Lab L channel (a simple contrast measure)."""
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].std())


def chroma_shift(reference, test):
    """Mean absolute difference of the Lab a and b channels vs the reference.

    Larger values mean colours moved further away from the original colours.
    """
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float64)
    tst = cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float64)
    return float(np.mean(np.abs(ref[:, :, 1:] - tst[:, :, 1:])))
