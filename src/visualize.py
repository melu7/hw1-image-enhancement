"""Figures: side-by-side comparisons and luminance histograms."""
import cv2
import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

PANEL_ORDER = ["original", "degraded", "stretch_luminance", "hist_eq_lab", "gamma_lab"]
PANEL_TITLES = {
    "original": "Original (reference)",
    "degraded": "Degraded (baseline)",
    "stretch_luminance": "Linear stretch (Lab L)",
    "hist_eq_lab": "Hist. equalization (Lab L)",
    "gamma_lab": "Gamma correction (Lab L)",
}


def _rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def comparison_figure(name, images, scores, crop_box, out_path):
    """Row 1: full images. Row 2: zoom on the same region (shows noise/detail).

    images: dict panel_key -> BGR image.  scores: dict panel_key -> (psnr, ssim).
    """
    x, y, w, h = crop_box
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
    for col, key in enumerate(PANEL_ORDER):
        img = images[key]
        title = PANEL_TITLES[key]
        if key in scores:
            p, s = scores[key]
            title += f"\nPSNR {p:.2f} dB | SSIM {s:.3f}"
        axes[0, col].imshow(_rgb(img))
        axes[0, col].set_title(title, fontsize=10)
        axes[1, col].imshow(_rgb(img[y:y + h, x:x + w]), interpolation="nearest")
        axes[1, col].set_title("zoom", fontsize=9)
        for row in (0, 1):
            axes[row, col].axis("off")
        # mark the zoom region on the full image
        axes[0, col].add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="yellow", lw=1.5))
    fig.suptitle(f"Image: {name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def histogram_figure(name, images, out_path):
    """Luminance (Lab L) histograms for each panel: shows how contrast changes."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 2.8), sharey=False)
    for ax, key in zip(axes, PANEL_ORDER):
        L = cv2.cvtColor(images[key], cv2.COLOR_BGR2LAB)[:, :, 0]
        ax.hist(L.ravel(), bins=64, range=(0, 255), color="0.35")
        ax.set_title(PANEL_TITLES[key], fontsize=9)
        ax.set_xlim(0, 255)
        ax.set_yticks([])
        ax.set_xlabel("luminance L (0-255)", fontsize=8)
    fig.suptitle(f"Luminance histograms: {name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def summary_figure(rows, out_path):
    """Bar charts of the mean PSNR and SSIM per method (averaged over images).

    rows: list of metric dictionaries (as written to the CSV).
    """
    order = ["baseline", "stretch_luminance", "hist_eq_lab", "gamma_lab"]
    labels = ["Baseline", "Stretch", "Hist. eq.", "Gamma"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, col, ttl in zip(axes, ["psnr_db", "ssim"], ["Mean PSNR (dB)", "Mean SSIM"]):
        vals = [np.mean([r[col] for r in rows if r["method"] == m]) for m in order]
        ax.bar(labels, vals, color=["0.6", "tab:blue", "tab:orange", "tab:green"])
        ax.set_title(ttl)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
