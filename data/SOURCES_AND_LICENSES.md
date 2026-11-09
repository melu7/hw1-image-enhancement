# Image sources and licenses

All four images are real photographs distributed with the open-source
**scikit-image** library (`skimage.data`). The license/attribution below is
taken from the scikit-image documentation of each image (docstrings in
`skimage/data/_fetchers.py`). Each was exported unchanged to `data/original/`
with `prepare_images.py`.

| File | Content | Source | License / status |
|------|---------|--------|------------------|
| `astronaut.png` | Portrait of astronaut Eileen Collins (512x512) | NASA Great Images database (https://flic.kr/p/r9qvLn) | No known copyright restrictions, released into the public domain |
| `chelsea.png` | Close-up of a cat, "Chelsea" (300x451) | Photographer: Stefan van der Walt | CC0 (no copyright restrictions) |
| `coffee.png` | Espresso cup on a wooden table (400x600) | Photographer: Rachel Michetti, courtesy of Pikolo Espresso Bar | CC0 (no copyright restrictions) |
| `rocket.png` | Falcon 9 / DSCOVR launch pad at dusk (427x640) | SpaceX Photos (https://www.flickr.com/photos/spacexphotos/16511594820/) | Captured by SpaceX and released into the public domain |

The degraded versions in `data/degraded/` are generated from these originals by
`run_experiment.py` (parameters in `src/config.py` and `results/parameters.json`).
