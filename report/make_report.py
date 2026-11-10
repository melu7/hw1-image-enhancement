"""Build report/report.pdf from the generated results.

Run AFTER `python run_experiment.py`:   python report/make_report.py
Edit AUTHOR and REPO_URL below before submitting.
"""
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Table,
                                TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

# ----------------------------- EDIT THESE ------------------------------------
AUTHOR = "Melat Habtemariyam Ekubay, VR545975"
REPO_URL = "https://github.com/melu7/hw1-image-enhancement"
# -----------------------------------------------------------------------------

OUT = cfg.ROOT / "report" / "report.pdf"
FIG_DIR = cfg.ROOT / "report" / "figs"
FIG_DIR.mkdir(exist_ok=True)
IMAGES = ["astronaut", "chelsea", "coffee", "rocket"]
MAIN = ["baseline", "stretch_luminance", "hist_eq_lab", "gamma_lab"]
COLS = ["original", "degraded", "stretch_luminance", "hist_eq_lab", "gamma_lab"]
TITLES = ["Original", "Degraded", "Linear stretch", "Hist. equal.", "Gamma"]

# ---- load metrics --------------------------------------------------------------
with open(cfg.METRICS_CSV, newline="", encoding="utf-8") as f:
    ROWS = list(csv.DictReader(f))


def g(img, method, col):
    for r in ROWS:
        if r["image"] == img and r["method"] == method:
            return float(r[col])
    raise KeyError((img, method, col))


def mean(method, col):
    return sum(g(i, method, col) for i in IMAGES) / len(IMAGES)


def load(img, key):
    if key == "original":
        p = cfg.ORIGINAL_DIR / f"{img}.png"
    elif key == "degraded":
        p = cfg.DEGRADED_DIR / f"{img}_degraded.png"
    else:
        p = cfg.ENHANCED_DIR / f"{img}_{key}.png"
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


# ---- figures ---------------------------------------------------------------------
def fig_overview(path):
    fig, axes = plt.subplots(4, 5, figsize=(11, 6.9))
    for r, img in enumerate(IMAGES):
        for c, key in enumerate(COLS):
            ax = axes[r, c]
            ax.imshow(load(img, key))
            ax.axis("off")
            if r == 0:
                ax.set_title(TITLES[c], fontsize=11)
        axes[r, 0].text(-0.04, 0.5, img, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=10)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_zoom(path, names):
    fig, axes = plt.subplots(len(names), 5, figsize=(11, 2.35 * len(names) + 0.4))
    for r, img in enumerate(names):
        x, y, w, h = cfg.CROP_BOXES[img]
        for c, key in enumerate(COLS):
            ax = axes[r, c]
            ax.imshow(load(img, key)[y:y + h, x:x + w], interpolation="nearest")
            ax.axis("off")
            t = TITLES[c]
            if key in ("stretch_luminance", "hist_eq_lab", "gamma_lab", "degraded"):
                m = "baseline" if key == "degraded" else key
                t += f"\nPSNR {g(img, m, 'psnr_db'):.2f} | SSIM {g(img, m, 'ssim'):.3f}"
            ax.set_title(t, fontsize=9)
        axes[r, 0].text(-0.04, 0.5, img, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=10)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=140)
    plt.close(fig)


fig_overview(FIG_DIR / "overview.png")
fig_zoom(FIG_DIR / "zoom.png", ["rocket", "chelsea"])

# ---- styles ------------------------------------------------------------------------
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=8.8, leading=11.2,
                      alignment=TA_JUSTIFY, spaceAfter=3.5)
H1 = ParagraphStyle("h1", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11,
                    spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1a3a5c"))
TITLE = ParagraphStyle("title", parent=ss["Title"], fontSize=15, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=BODY, alignment=1, fontSize=9, spaceAfter=6)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.8, leading=9.6, textColor=colors.HexColor("#333333"))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.6, leading=9, alignment=0, spaceAfter=0)
CELLC = ParagraphStyle("cellc", parent=CELL, alignment=1)


def P(t, s=BODY):
    return Paragraph(t, s)


def table(data, widths, header_rows=1, extra=None):
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
             ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e6edf5")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    t.setStyle(TableStyle(style + (extra or [])))
    return t


# ---- numbers used in the text -------------------------------------------------------
gam = {i: g(i, "gamma_lab", "gamma_used") for i in IMAGES}
sd_o = {i: g(i, "original_reference", "contrast_std_L") for i in IMAGES}
sd_d = {i: g(i, "baseline", "contrast_std_L") for i in IMAGES}
sd_s = {i: g(i, "stretch_luminance", "contrast_std_L") for i in IMAGES}
sd_h = {i: g(i, "hist_eq_lab", "contrast_std_L") for i in IMAGES}

story = []
story += [P("Homework 1: Image Enhancement with Classical Methods", TITLE),
          P(f"Computer Vision, Fall 2026 &nbsp;|&nbsp; {AUTHOR}", SUB)]

# 1 Objective
story += [P("1. Objective", H1),
          P("The goal is to build and compare a small pipeline that enhances colour images affected by low contrast, "
            "non-uniform illumination and noise, using only classical image processing (OpenCV and NumPy; no neural "
            "networks). The aim is not to reconstruct the original scene perfectly, but to justify the choices, observe "
            "the effect of each method and discuss the trade-offs between detail visibility, noise and natural "
            "appearance. <b>Important:</b> the original photographs are used as <i>references</i> only because the "
            "degradations are synthetic. In real-world scenarios a clean reference is usually not available, so PSNR "
            "and SSIM could not be computed and the evaluation would rely on visual judgement.")]

# 2 Images
src = [[P("<b>Image</b>", CELL), P("<b>Content and challenge</b>", CELL), P("<b>Source</b>", CELL),
        P("<b>License</b>", CELL)],
       [P("astronaut", CELL), P("Portrait; dark suit and helmet next to bright background (512x512)", CELL),
        P("NASA Great Images database, via scikit-image", CELL), P("Public domain", CELL)],
       [P("chelsea", CELL), P("Cat close-up; fine fur texture, whiskers, soft mid-tones (300x451)", CELL),
        P("Photo: S. van der Walt, via scikit-image", CELL), P("CC0", CELL)],
       [P("coffee", CELL), P("Espresso cup; dark shadow areas and wood grain (400x600)", CELL),
        P("Photo: R. Michetti (Pikolo Espresso Bar), via scikit-image", CELL), P("CC0", CELL)],
       [P("rocket", CELL), P("Launch pad at dusk; strongly non-uniform illumination, dark sky (427x640)", CELL),
        P("SpaceX Photos (Flickr), via scikit-image", CELL), P("Public domain (released by SpaceX)", CELL)]]
story += [P("2. Images, sources and licenses", H1),
          P("Four real photographs were used (Table 1). Licenses are those stated in the scikit-image documentation; "
            "the images are included in the repository (<font face='Courier'>data/original/</font>), together with "
            "<font face='Courier'>data/SOURCES_AND_LICENSES.md</font>. For each image a degraded version was generated "
            "(Section 3) and both versions are kept.", BODY),
          table(src, [2.0 * cm, 7.2 * cm, 5.2 * cm, 3.0 * cm]),
          P("Table 1. Image set.", CAP)]

# 3 Methods
story += [P("3. Methods and parameters", H1),
          P(f"<b>Degradation (fixed for all methods).</b> Contrast reduction <i>out = a(in - 128) + 128</i> with "
            f"a = {cfg.CONTRAST_ALPHA}, followed by additive Gaussian noise with sigma = {cfg.NOISE_SIGMA:g} grey "
            f"levels (0-255 scale), clipping to [0, 255] and rounding to 8 bits. The random generator uses the fixed "
            f"seed {cfg.SEED}.", BODY),
          P("<b>Colour handling.</b> Methods 2-4 convert the BGR image to CIE-Lab, modify only the luminance channel L "
            "and keep the colour channels a, b unchanged, so the enhancement itself does not mix colours.", BODY),
          P("<b>M1 Baseline:</b> the degraded image, unprocessed. "
            f"<b>M2 Linear contrast stretching:</b> on L, the 1st and 99th percentiles are mapped to 0 and 255 "
            f"(out = 255 (L - p1) / (p99 - p1), clipped to [0, 255]; percentile clipping avoids control by a few extreme pixels). A supplementary variant stretches each "
            f"B, G, R channel separately, only to discuss colour shifts. "
            "<b>M3 Histogram equalization:</b> global equalization (<font face='Courier'>cv2.equalizeHist</font>) of L: each level is mapped through the normalised cumulative histogram, out = 255 (CDF(L) - CDFmin) / (N - CDFmin). "
            f"<b>M4 Gamma correction:</b> on L, <i>out = 255(in/255)^gamma</i>, where gamma is estimated from the "
            f"degraded image alone so that its mean luminance becomes 0.5 (resulting gamma: "
            + ", ".join(f"{i} {gam[i]:.2f}" for i in IMAGES) + ").", BODY),
          P("<b>Metrics</b> (always against the original): PSNR (own NumPy implementation, checked against <font face='Courier'>cv2.PSNR</font> in <font face='Courier'>tests/sanity_checks.py</font>), SSIM (scikit-image, colour, data range 255) and mean absolute "
            "error (MAE, in the CSV). We also report the luminance standard deviation (contrast) and the mean "
            "Lab a/b difference to the original (colour shift).", BODY)]

# 4 Results
best = {i: max(MAIN, key=lambda m: g(i, m, "psnr_db")) for i in IMAGES}
head = [P("<b>Image</b>", CELLC), P("<b>M1 Baseline</b>", CELLC), P("<b>M2 Stretch (L)</b>", CELLC),
        P("<b>M3 Hist. eq. (L)</b>", CELLC), P("<b>M4 Gamma (L)</b>", CELLC),
        P("<b>Suppl.: stretch per channel</b>", CELLC)]
data = [head]
for i in IMAGES:
    row = [P(i, CELL)]
    for m in MAIN + ["stretch_per_channel"]:
        txt = f"{g(i, m, 'psnr_db'):.2f} / {g(i, m, 'ssim'):.3f}"
        if m == best[i]:
            txt = f"<b>{txt}</b>"
        row.append(P(txt, CELLC))
    data.append(row)
data.append([P("<b>Mean</b>", CELL)] + [P(f"{mean(m, 'psnr_db'):.2f} / {mean(m, 'ssim'):.3f}", CELLC)
                                         for m in MAIN + ["stretch_per_channel"]])
res_tbl = table(data, [2.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.4 * cm],
                extra=[("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1), colors.HexColor("#f3f3f3"))])

story += [KeepTogether([P("4. Results", H1),
          Image(str(FIG_DIR / "overview.png"), width=17.2 * cm, height=17.2 * cm * 6.9 / 11),
          P("Figure 1. Original, degraded and enhanced results for the four images (same size and scale for all panels). "
            "Larger per-image figures with zoom and histograms are in <font face='Courier'>results/figures/</font>.", CAP)]),
          res_tbl,
          P("Table 2. PSNR (dB) / SSIM with respect to the original; bold = highest PSNR among M1-M4. All values are "
            "computed by the code (<font face='Courier'>results/metrics.csv</font>, which also contains MAE).", CAP)]

# 5 Discussion
story += [P("5. Discussion", H1),
          P(f"<b>Overall.</b> Linear stretching of the luminance gives the highest PSNR on three of the four images and "
            f"the best mean PSNR ({mean('stretch_luminance', 'psnr_db'):.2f} dB versus {mean('baseline', 'psnr_db'):.2f} "
            f"dB for the baseline). Histogram equalization has the lowest mean scores "
            f"({mean('hist_eq_lab', 'psnr_db'):.2f} dB, SSIM {mean('hist_eq_lab', 'ssim'):.3f}). Gamma correction is "
            f"almost the identity because the contrast reduction pulls the mean luminance towards the middle grey: the "
            f"estimated gamma is close to 1 for three images, so it cannot fix low contrast. It only changes the result "
            f"clearly for the dark <i>rocket</i> image (gamma {gam['rocket']:.2f}).", BODY),
          P(f"<b>Case 1: rocket (non-uniform illumination).</b> The gantry towers and the rocket body become much easier "
            f"to see after linear stretching (luminance contrast {sd_d['rocket']:.1f} -> {sd_s['rocket']:.1f}; the "
            f"original has {sd_o['rocket']:.1f}), and PSNR rises from {g('rocket', 'baseline', 'psnr_db'):.2f} to "
            f"{g('rocket', 'stretch_luminance', 'psnr_db'):.2f} dB. However SSIM <i>decreases</i> "
            f"({g('rocket', 'baseline', 'ssim'):.3f} -> {g('rocket', 'stretch_luminance', 'ssim'):.3f}), because the "
            f"noise in the large flat sky is amplified and the blue dusk sky turns grey: the degradation compressed the "
            f"colour channels too, and luminance-only stretching does not restore them. Histogram equalization performs "
            f"worst here ({g('rocket', 'hist_eq_lab', 'psnr_db'):.2f} dB, SSIM {g('rocket', 'hist_eq_lab', 'ssim'):.3f}): "
            f"the sky covers most pixels, so its grey levels are spread over half of the range, producing a bright, "
            f"strongly noisy sky and saturated highlights. Gamma (brightening) lowers PSNR to "
            f"{g('rocket', 'gamma_lab', 'psnr_db'):.2f} dB because the original dusk scene is genuinely dark. "
            f"<b>Preferred method: linear stretch</b>, as it reveals structure with the most natural result, but the "
            f"amplified noise is clearly visible (Figure 2).", BODY),
          P(f"<b>Case 2: chelsea (numbers and visual quality disagree).</b> Visually, stretching and equalization make the "
            f"fur texture, whiskers and the structure of the eye much clearer than the washed-out degraded image. "
            f"Yet both have <i>lower</i> PSNR/SSIM than the unprocessed baseline "
            f"({g('chelsea', 'stretch_luminance', 'psnr_db'):.2f} and {g('chelsea', 'hist_eq_lab', 'psnr_db'):.2f} dB "
            f"versus {g('chelsea', 'baseline', 'psnr_db'):.2f} dB). The reason is that the original is itself a "
            f"low-contrast photograph (luminance std {sd_o['chelsea']:.1f}); the methods push it to "
            f"{sd_s['chelsea']:.1f} (stretch) and {sd_h['chelsea']:.1f} (equalization), going <i>beyond</i> the "
            f"reference, and they amplify noise, and both effects count as error. Conversely, gamma has the highest "
            f"numbers on this image (PSNR {g('chelsea', 'gamma_lab', 'psnr_db'):.2f} dB, SSIM "
            f"{g('chelsea', 'gamma_lab', 'ssim'):.3f}) only because gamma = {gam['chelsea']:.2f} leaves the image almost "
            f"unchanged and flat. So a numerical improvement did not mean a more pleasing image, and a more pleasing "
            f"image did not mean a better score. <b>Preferred method: linear stretch</b> (equalization looks harsh and "
            f"noisy in the fur and background).", BODY),
          KeepTogether([Image(str(FIG_DIR / "zoom.png"), width=17.2 * cm, height=17.2 * cm * (2.35 * 2 + 0.4) / 11),
          P("Figure 2. Zoomed regions (boxes marked in <font face='Courier'>results/figures/*_comparison.png</font>) "
            "for the two analysed cases, <i>rocket</i> and <i>chelsea</i>. Titles give PSNR (dB) | SSIM.", CAP)]),
          P(f"<b>Noise, colours and detail.</b> Every method that expands contrast expands noise by a similar factor "
            f"(for linear stretching the luminance contrast is multiplied by "
            f"{min(sd_s[i]/sd_d[i] for i in IMAGES):.1f}-{max(sd_s[i]/sd_d[i] for i in IMAGES):.1f}, depending on the "
            f"image, and the noise is amplified accordingly), and equalization does this most in flat regions. Because M2-M4 leave the a/b channels untouched, hues are preserved but the colours stay as "
            f"desaturated as in the degraded input (colour shift equal to the baseline, e.g. astronaut "
            f"{g('astronaut', 'baseline', 'chroma_shift_ab'):.2f} vs {g('astronaut', 'stretch_luminance', 'chroma_shift_ab'):.2f}). "
            f"The supplementary per-channel stretch restores saturation on <i>astronaut</i> and <i>coffee</i> "
            f"(highest PSNR there: {g('astronaut', 'stretch_per_channel', 'psnr_db'):.2f} and "
            f"{g('coffee', 'stretch_per_channel', 'psnr_db'):.2f} dB) but introduces a strong blue cast on <i>rocket</i> "
            f"(colour shift {g('rocket', 'baseline', 'chroma_shift_ab'):.2f} -> "
            f"{g('rocket', 'stretch_per_channel', 'chroma_shift_ab'):.2f}), so luminance-only processing is safer. "
            f"Percentile clipping and equalization saturate some highlights (cup rim, launch lights, sky), losing "
            f"detail there; none of the methods can recover detail already hidden by noise, and none removes noise.", BODY)]

# 6 Limitations
story += [P("6. Limitations and trade-offs", H1),
          P("(i) Only four public photographs were used and they were not captured under extreme lighting; low "
            "contrast is mostly introduced synthetically. (ii) The degradation is linear, so linear stretching is "
            "almost its exact inverse, which favours M2 in the numbers; real shadows are not removed by a global "
            "linear mapping. (iii) Single noise seed and single parameter set, so no confidence intervals. (iv) "
            "Global equalization was used; local methods (CLAHE) or denoising combined with stretching (e.g. a "
            "bilateral filter) could reduce the noise amplification but were not part of this study. (v) PSNR and "
            "SSIM are global measures and do not fully capture perceived quality, so no method was chosen on a "
            "single metric alone.", BODY)]

# 7 Reproducibility
story += [P("7. Repository and reproducibility", H1),
          P(f"Repository: <b>{REPO_URL}</b>. It contains the Python code, original and degraded images, processed "
            f"images, <font face='Courier'>results/metrics.csv</font> and a README with the Python version "
            f"(3.12; 3.9+ supported), dependencies and the command "
            f"<font face='Courier'>python run_experiment.py</font>, which runs the whole pipeline from the original "
            f"images with fixed seed {cfg.SEED} and produces byte-identical outputs on repeated runs.", BODY),
          P("<b>References.</b> G. Bradski, The OpenCV Library, <i>Dr. Dobb's Journal</i>, 2000. &nbsp; "
            "S. van der Walt et al., scikit-image: image processing in Python, <i>PeerJ</i> 2:e453, 2014. &nbsp; "
            "Z. Wang et al., Image quality assessment: from error visibility to structural similarity, "
            "<i>IEEE Trans. Image Process.</i> 13(4):600-612, 2004.", CAP)]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                        title="Homework 1: Image Enhancement", author=AUTHOR)
doc.build(story)
print("written", OUT)
