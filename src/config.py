"""Central configuration: every parameter of the experiment lives here.

Keeping all numbers in one file makes the experiment reproducible and makes it
easy to document them in the report.
"""
from pathlib import Path

# ---- Paths (all relative to the project root) -------------------------------
ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = ROOT / "data" / "original"
DEGRADED_DIR = ROOT / "data" / "degraded"
ENHANCED_DIR = ROOT / "results" / "enhanced"
FIGURES_DIR = ROOT / "results" / "figures"
METRICS_CSV = ROOT / "results" / "metrics.csv"
PARAMS_JSON = ROOT / "results" / "parameters.json"

# ---- Degradation parameters --------------------------------------------------
SEED = 42              # fixed random seed for the Gaussian noise
CONTRAST_ALPHA = 0.45  # contrast factor: 1.0 = unchanged, 0.45 = strongly reduced
CONTRAST_PIVOT = 128   # grey level the contrast reduction is centred on
NOISE_SIGMA = 5.0      # std. dev. of Gaussian noise, in grey levels (0-255 scale)

# ---- Enhancement parameters --------------------------------------------------
STRETCH_LOW_PCT = 1.0    # linear stretch: percentile mapped to black
STRETCH_HIGH_PCT = 99.0  # linear stretch: percentile mapped to white
GAMMA_TARGET_MEAN = 0.5  # gamma is chosen so the mean luminance becomes ~0.5
GAMMA_MIN, GAMMA_MAX = 0.4, 2.5  # safety limits for the estimated gamma

# ---- Figures -----------------------------------------------------------------
# Zoom region (x, y, width, height) in pixels, used to inspect noise and detail.
CROP_BOXES = {
    "astronaut": (150, 10, 160, 160),
    "chelsea": (100, 60, 140, 140),
    "coffee": (180, 150, 200, 200),
    "rocket": (240, 230, 180, 180),
}
