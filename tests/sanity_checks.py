"""Independent sanity checks for the experiment.   Run:  python tests/sanity_checks.py

Run `python run_experiment.py` first (the last check reads the generated results).
Each check re-computes something in a different way and compares it with the pipeline.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src import enhance, metrics  # noqa: E402
from src.degrade import degrade  # noqa: E402

FAILED = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not condition:
        FAILED.append(name)


def L_of(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]


rng = np.random.default_rng(0)
photo = cv2.imread(str(next(cfg.ORIGINAL_DIR.glob("*.png"))))  # any original image

# 1. Degradation ----------------------------------------------------------------
d1 = degrade(photo, cfg.CONTRAST_ALPHA, cfg.CONTRAST_PIVOT, cfg.NOISE_SIGMA, cfg.SEED)
d2 = degrade(photo, cfg.CONTRAST_ALPHA, cfg.CONTRAST_PIVOT, cfg.NOISE_SIGMA, cfg.SEED)
check("degradation is deterministic (same seed -> identical image)", np.array_equal(d1, d2))
d_nonoise = degrade(photo, cfg.CONTRAST_ALPHA, cfg.CONTRAST_PIVOT, 0.0, cfg.SEED)
expected = np.clip(np.round(cfg.CONTRAST_ALPHA * (photo.astype(float) - cfg.CONTRAST_PIVOT) + cfg.CONTRAST_PIVOT), 0, 255)
check("contrast formula out = a*(in-128)+128 is exact when sigma=0", np.array_equal(d_nonoise, expected.astype(np.uint8)))
grey = np.full((400, 400, 3), 128, np.uint8)
noise_std = (degrade(grey, 1.0, 128, cfg.NOISE_SIGMA, cfg.SEED).astype(float) - 128).std()
check("noise standard deviation is about sigma", abs(noise_std - cfg.NOISE_SIGMA) < 0.2, f"measured {noise_std:.2f}")
check("degradation lowers luminance contrast", metrics.luminance_contrast(d1) < metrics.luminance_contrast(photo))

# 2. Metrics ---------------------------------------------------------------------
check("PSNR matches cv2.PSNR", abs(metrics.psnr(photo, d1) - cv2.PSNR(photo, d1)) < 1e-6)
const = np.full((64, 64, 3), 100, np.uint8), np.full((64, 64, 3), 110, np.uint8)
check("PSNR of a constant error of 10 equals 20*log10(255/10)",
      abs(metrics.psnr(*const) - 20 * np.log10(255 / 10)) < 1e-9)
check("PSNR of identical images is infinite", metrics.psnr(photo, photo) == float("inf"))
check("SSIM of identical images is 1", abs(metrics.ssim(photo, photo) - 1.0) < 1e-12)
check("SSIM matches scikit-image called directly",
      abs(metrics.ssim(photo, d1) - structural_similarity(photo, d1, channel_axis=2, data_range=255)) < 1e-12)
check("MAE of a constant error of 10 equals 10", abs(metrics.mae(*const) - 10.0) < 1e-12)

# 3. Enhancement methods -----------------------------------------------------------
s = enhance.stretch_luminance(d1)
lo, hi = np.percentile(L_of(s), [cfg.STRETCH_LOW_PCT, cfg.STRETCH_HIGH_PCT])
check("stretch maps the 1st/99th luminance percentiles to about 0/255", lo <= 3 and hi >= 252, f"{lo:.0f}/{hi:.0f}")
ab_in = cv2.cvtColor(d1, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(float)
ab_out = cv2.cvtColor(s, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(float)
check("luminance stretch leaves colour channels (Lab a,b) almost unchanged",
      np.abs(ab_in - ab_out).mean() < 3.0, f"mean change {np.abs(ab_in - ab_out).mean():.2f}")

Lin = L_of(d1)
hist = np.bincount(Lin.ravel(), minlength=256)
cdf = hist.cumsum()
cdf_min = cdf[cdf > 0][0]
lut = np.clip(np.round((cdf - cdf_min) / (Lin.size - cdf_min) * 255), 0, 255).astype(np.uint8)
check("histogram equalization equals an independent NumPy CDF implementation",
      np.array_equal(cv2.equalizeHist(Lin), lut[Lin]))
e = L_of(enhance.equalize_luminance(d1))
check("equalization raises luminance contrast", e.std() > Lin.std(), f"{Lin.std():.1f} -> {e.std():.1f}")

g_id, _ = enhance.gamma_correction(d1, gamma=1.0)
check("gamma = 1 leaves the image (almost) unchanged", np.abs(g_id.astype(int) - d1.astype(int)).max() <= 2)
dark = np.clip(photo * 0.3, 0, 255).astype(np.uint8)
g_dark, gm = enhance.gamma_correction(dark)
check("estimated gamma < 1 for a dark image, and it brightens it", gm < 1 and L_of(g_dark).mean() > L_of(dark).mean(), f"gamma {gm:.2f}")
mean_L = L_of(dark).astype(float).mean() / 255
check("estimated gamma maps the mean luminance to the target 0.5",
      abs(mean_L ** gm - cfg.GAMMA_TARGET_MEAN) < 1e-6 or gm in (cfg.GAMMA_MIN, cfg.GAMMA_MAX))

# 4. Source code uses no deep learning -------------------------------------------------
forbidden = ("torch", "tensorflow", "keras", "onnx", "cv2.dnn", "sklearn")
src_text = " ".join(p.read_text() for p in list(cfg.ROOT.glob("src/*.py")) + [cfg.ROOT / "run_experiment.py"])
check("no deep-learning library or cv2.dnn is used", not any(w in src_text for w in forbidden))

# 5. Saved results agree with the CSV (recomputed from the PNG files on disk) -------------
if cfg.METRICS_CSV.exists():
    bad = 0
    n = 0
    with open(cfg.METRICS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == "original_reference":
                continue
            orig = cv2.imread(str(cfg.ORIGINAL_DIR / f"{row['image']}.png"))
            if row["method"] == "baseline":
                img = cv2.imread(str(cfg.DEGRADED_DIR / f"{row['image']}_degraded.png"))
            else:
                img = cv2.imread(str(cfg.ENHANCED_DIR / f"{row['image']}_{row['method']}.png"))
            n += 1
            if abs(metrics.psnr(orig, img) - float(row["psnr_db"])) > 1e-3 or abs(metrics.ssim(orig, img) - float(row["ssim"])) > 1e-3:
                bad += 1
    check(f"PSNR/SSIM in metrics.csv match values recomputed from the saved images ({n} rows)", bad == 0 and n > 0)
else:
    print("SKIP  metrics.csv not found - run run_experiment.py first")

print("\nAll checks passed." if not FAILED else f"\n{len(FAILED)} check(s) FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
