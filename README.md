# Homework 1 - Image Enhancement (Computer Vision, Fall 2026)

A small, fully reproducible pipeline that enhances low-contrast, noisy colour
images with **classical** methods only (no deep learning) and compares them
quantitatively (PSNR, SSIM, MAE) and qualitatively.

Methods compared (all applied to the same degraded images, parameters fixed):

1. **Baseline** - the degraded image, unprocessed.
2. **Linear contrast stretching** - on the CIE-Lab luminance channel L
   (1st-99th percentile -> 0-255). A per-channel (B, G, R) variant is also
   computed as a *supplementary* method to discuss colour shifts.
3. **Histogram equalization** - global, on the CIE-Lab luminance channel L.
4. **Gamma correction** - on the CIE-Lab luminance channel L; gamma is estimated
   from the degraded image so that its mean luminance becomes 0.5.

The original photographs act as references for the synthetic degradation
(contrast reduction + Gaussian noise). In real situations a clean reference is
usually not available.

## Requirements

* Python 3.9 or newer (developed and tested with Python 3.12)
* Core packages (`requirements.txt`): NumPy, OpenCV, scikit-image (SSIM), Matplotlib (figures)
* Optional (`requirements-report.txt`): ReportLab, only needed to rebuild the PDF report

```bash
pip install -r requirements.txt
```

## Run the whole experiment

From the project root folder:

```bash
python run_experiment.py
```

This reads every image in `data/original/`, creates the degraded versions
(fixed seed 42), applies the methods, computes the metrics, and writes all
results. It takes a few seconds. Running it twice gives byte-identical outputs.

Optional: run the independent sanity checks (re-computes the metrics from the saved
images, checks the algorithms against separate implementations, and compares with the CSV):

```bash
python tests/sanity_checks.py
```

Optional: rebuild the PDF report from the generated results (needs ReportLab):

```bash
pip install -r requirements-report.txt
python report/make_report.py
```

## Degradation parameters (see `src/config.py`)

| Parameter | Value |
|-----------|-------|
| Contrast: `out = a*(in-128)+128` | a = 0.45 |
| Gaussian noise (0-255 scale) | sigma = 5 |
| Random seed (NumPy `default_rng`) | 42 |

## Project structure

```
hw1-image-enhancement/
├── README.md                  this file
├── requirements.txt           core Python dependencies
├── requirements-report.txt    optional: ReportLab for the PDF report
├── run_experiment.py          MAIN SCRIPT: runs the complete pipeline
├── prepare_images.py          (optional) exports the 4 sample photos to data/original/
├── src/
│   ├── config.py              ALL parameters and paths
│   ├── degrade.py             contrast reduction + Gaussian noise
│   ├── enhance.py             stretch, histogram equalization, gamma correction
│   ├── metrics.py             PSNR, SSIM, MAE, contrast and colour-shift measures
│   └── visualize.py           comparison figures, histograms, summary chart
├── tests/
│   └── sanity_checks.py       independent checks of degradation, algorithms, metrics, CSV
├── data/
│   ├── SOURCES_AND_LICENSES.md  where each image comes from + license
│   ├── original/              4 original photographs (inputs, used as references)
│   └── degraded/              degraded versions (generated)
├── results/
│   ├── enhanced/              processed images (generated)
│   ├── figures/               *_comparison.png, *_histograms.png, summary_metrics.png
│   ├── metrics.csv            all numerical results (generated)
│   └── parameters.json        parameters actually used (generated)
└── report/
    ├── make_report.py         builds the PDF from the results
    └── report.pdf             final report
```

## Using your own images

Copy `.png` / `.jpg` files into `data/original/`, add a zoom box for each in
`CROP_BOXES` in `src/config.py` (optional; a default box is used otherwise),
and run `python run_experiment.py` again.

## Metrics columns in `results/metrics.csv`

`psnr_db`, `ssim`, `mae` are computed against the original image.
The row `original_reference` only reports the contrast of the clean original (no metrics).
`contrast_std_L` is the standard deviation of luminance (a simple contrast
measure). `chroma_shift_ab` is the mean absolute difference of the Lab a/b
colour channels with respect to the original. `gamma_used` is the estimated gamma.
