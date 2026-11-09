"""Run the COMPLETE experiment:  python run_experiment.py

For every image in data/original/ it will:
  1. create a degraded copy (low contrast + Gaussian noise, fixed seed),
  2. apply the enhancement methods to the degraded copy,
  3. compute PSNR / SSIM / MAE against the original,
  4. save the processed images, comparison figures, and results/metrics.csv.
"""
import csv
import json
import sys

import cv2
import numpy as np

from src import config as cfg
from src import enhance, metrics, visualize
from src.degrade import degrade

# Methods that appear in the main comparison + one supplementary variant.
METHODS = ["baseline", "stretch_luminance", "hist_eq_lab", "gamma_lab", "stretch_per_channel"]


def main():
    for d in (cfg.ORIGINAL_DIR, cfg.DEGRADED_DIR, cfg.ENHANCED_DIR, cfg.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in cfg.ORIGINAL_DIR.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        sys.exit(f"No images found in {cfg.ORIGINAL_DIR}")

    rows = []
    gammas = {}
    for path in paths:
        name = path.stem
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR, uint8
        if original is None:
            sys.exit(f"Could not read {path}")

        # --- 1. degrade -------------------------------------------------------
        degraded = degrade(original, cfg.CONTRAST_ALPHA, cfg.CONTRAST_PIVOT,
                           cfg.NOISE_SIGMA, cfg.SEED)
        cv2.imwrite(str(cfg.DEGRADED_DIR / f"{name}_degraded.png"), degraded)

        # --- 2. enhance -------------------------------------------------------
        gamma_img, gamma_used = enhance.gamma_correction(degraded)
        gammas[name] = round(gamma_used, 4)
        results = {
            "baseline": enhance.baseline(degraded),
            "stretch_luminance": enhance.stretch_luminance(degraded),
            "hist_eq_lab": enhance.equalize_luminance(degraded),
            "gamma_lab": gamma_img,
            "stretch_per_channel": enhance.stretch_per_channel(degraded),
        }
        for key, img in results.items():
            if key != "baseline":
                cv2.imwrite(str(cfg.ENHANCED_DIR / f"{name}_{key}.png"), img)

        # --- 3. metrics (always against the ORIGINAL image) --------------------
        for key in METHODS:
            img = results[key]
            rows.append({
                "image": name,
                "method": key,
                "psnr_db": round(metrics.psnr(original, img), 4),
                "ssim": round(metrics.ssim(original, img), 4),
                "mae": round(metrics.mae(original, img), 4),
                "contrast_std_L": round(metrics.luminance_contrast(img), 2),
                "chroma_shift_ab": round(metrics.chroma_shift(original, img), 3),
                "gamma_used": gammas[name] if key == "gamma_lab" else "",
            })

        # --- 4. figures --------------------------------------------------------
        images = {"original": original, "degraded": degraded, **results}
        scores = {r["method"]: (r["psnr_db"], r["ssim"]) for r in rows if r["image"] == name}
        scores["degraded"] = scores["baseline"]
        # Reference rows for the original itself: contrast of the clean image.
        visualize.comparison_figure(name, images, scores, cfg.CROP_BOXES.get(name, (0, 0, 120, 120)),
                                    cfg.FIGURES_DIR / f"{name}_comparison.png")
        visualize.histogram_figure(name, images, cfg.FIGURES_DIR / f"{name}_histograms.png")
        rows.append({
            "image": name, "method": "original_reference", "psnr_db": "", "ssim": "", "mae": "",
            "contrast_std_L": round(metrics.luminance_contrast(original), 2),
            "chroma_shift_ab": "", "gamma_used": "",
        })
        print(f"[ok] {name}")

    # --- save CSV + parameters ---------------------------------------------------
    with open(cfg.METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    params = {
        "seed": cfg.SEED, "contrast_alpha": cfg.CONTRAST_ALPHA, "contrast_pivot": cfg.CONTRAST_PIVOT,
        "noise_sigma": cfg.NOISE_SIGMA, "stretch_percentiles": [cfg.STRETCH_LOW_PCT, cfg.STRETCH_HIGH_PCT],
        "gamma_target_mean": cfg.GAMMA_TARGET_MEAN, "gamma_per_image": gammas,
        "opencv": cv2.__version__, "numpy": np.__version__,
    }
    cfg.PARAMS_JSON.write_text(json.dumps(params, indent=2))

    metric_rows = [r for r in rows if r["method"] != "original_reference"]
    visualize.summary_figure(metric_rows, cfg.FIGURES_DIR / "summary_metrics.png")

    print(f"\nMetrics written to {cfg.METRICS_CSV}")
    for col in ("psnr_db", "ssim"):
        print(f"\n{col}")
        print(f"{'image':<11}" + "".join(f"{m:>21}" for m in METHODS))
        for name in sorted({r['image'] for r in metric_rows}):
            vals = {r["method"]: r[col] for r in metric_rows if r["image"] == name}
            print(f"{name:<11}" + "".join(f"{vals[m]:>21.3f}" for m in METHODS))


if __name__ == "__main__":
    main()
