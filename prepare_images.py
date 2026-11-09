"""OPTIONAL helper: export the four sample photos from scikit-image to data/original/.

The exported PNG files are already included in the repository, so you do NOT
need to run this. It documents exactly where the images came from.
To use your own photos instead, just put .png/.jpg files in data/original/
(and update src/config.py CROP_BOXES for the zoomed figures).
"""
import cv2
from skimage import data

from src import config as cfg

SAMPLES = {
    "astronaut": data.astronaut,
    "chelsea": data.chelsea,
    "coffee": data.coffee,
    "rocket": data.rocket,
}

if __name__ == "__main__":
    cfg.ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, loader in SAMPLES.items():
        rgb = loader()
        cv2.imwrite(str(cfg.ORIGINAL_DIR / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print("saved", name, rgb.shape)
