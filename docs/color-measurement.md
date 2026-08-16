# How the dot colors are measured

Every dot on the [Seasons page](https://noahkarsky.github.io/texas-mushrooms/#/seasons) is painted
the dominant color of the mushroom in that photograph. This is how that color is derived, and how
well it holds up.

## The method

A whole-frame color histogram returns the color of the *scene* — soil, leaf litter, shadow —
because on a forest-floor photograph the mushroom is a small minority of the pixels.
`src/texas_mushrooms/pipeline/color.py` therefore isolates the subject before measuring it: it
builds a background color model from the four border bands, weights every pixel by how unlike that
background it is, times local **smoothness** (leaf litter is the most textured surface in frame;
mushroom flesh is among the least) times a mild center prior, keeps the top 20% of pixels, and runs
a weighted k-means in OKLab. No model, no network, no LLM — just Pillow and numpy, deterministic
and seeded.

## How well it works

Photographers state the color in ~1,890 captions ("Light yellow resupinate fungus on a log…"),
which is free ground truth. `scripts/eval_photo_colors.py` scores each extractor by
nearest-class-prototype accuracy under 5-fold CV. Against the previous whole-frame octree
extractor:

| | old | new |
| --- | --- | --- |
| CV accuracy (chance ≈ 0.04) | 0.152 | **0.232** |
| genus color coherence (lower is better) | 0.932 | **0.819** |
| recall: yellow / orange / black | 0.045 / 0.011 / 0.026 | **0.358 / 0.236 / 0.569** |
| recall: white / red | 0.152 / 0.287 | **0.266 / 0.452** |

The metric is nearest-class-**prototype** accuracy, never nearest hand-authored color anchor — an
anchor bakes in an exposure assumption (a white cap in forest shade is OKLab L≈0.62 against a
nominal white of ≈0.95) and ranks the extractors backwards.

## Known limits

Brown and green recall *drop* (0.378 → 0.040, 0.244 → 0.024). That is not a regression in disguise:
the old extractor scored well on brown by predicting brown for everything — it answered "brown" 490
times for 347 truly brown photos, at 0.267 precision, and answered "blue" 175 times at 0.000
precision. The new extractor's precision on the colors that matter is 0.34–0.40 versus 0.01–0.27.

Genuinely drab species remain hard by construction: when the mushroom really is the color of the
litter, "distance from the background" is the wrong signal, and those photos fall back to the
whole-frame color. The caption-labeled set is also biased — photographers name a color mainly for
distinctive specimens.

This is a measurement with known failure modes, not a classifier.
