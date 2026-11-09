"""Create a synthetic degraded version of an image (low contrast + noise)."""
import numpy as np


def degrade(img_bgr, alpha, pivot, sigma, seed):
    """Reduce contrast and add Gaussian noise.

    Step 1 (contrast):  out = alpha * (in - pivot) + pivot
        With alpha < 1 every pixel moves towards the grey level `pivot`,
        so the image looks flat and washed out.
    Step 2 (noise):     out = out + N(0, sigma^2)
        Random values from a Gaussian (bell-curve) distribution are added.
    The random generator is created from a fixed `seed`, so running the code
    twice gives exactly the same degraded image.
    """
    rng = np.random.default_rng(seed)
    x = img_bgr.astype(np.float64)
    x = alpha * (x - pivot) + pivot
    x = x + rng.normal(0.0, sigma, size=x.shape)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)
