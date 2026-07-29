"""Measure how well an extractor recovers the *mushroom's* color, not the scene's.

Photographers state the color in their own captions ("Light yellow resupinate
fungus on a log..."), which gives ~1,900 photos with a free ground-truth label.
This script scores each extractor variant against those labels, runs ablations
over the priors, sweeps the fallback thresholds, and writes a visual contact
sheet so the numbers can be checked by eye.

Run:
    python scripts/eval_photo_colors.py                     # v1 vs v2
    python scripts/eval_photo_colors.py --ablate            # + prior ablations
    python scripts/eval_photo_colors.py --sample 400        # quick iteration

Metric note: absolute accuracy is meaningless here. Only the delta versus v1 and
versus the permutation-chance floor is interpretable -- see ``_METRIC_NOTE``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from texas_mushrooms.pipeline.caption_color import (  # noqa: E402
    single_color_label,
)
from texas_mushrooms.pipeline.color import (  # noqa: E402
    ColorExtractionError,
    ColorParams,
    extract_legacy_color,
    extract_photo_color,
    hex_to_oklab,
)
from texas_mushrooms.pipeline.photo_assets import (  # noqa: E402
    genus_of,
    index_local_images,
    match_local_file,
    photo_id,
)

logger = logging.getLogger("eval_photo_colors")

DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "raw" / "images"
PHOTOS_CLEANED_CSV = DATA_DIR / "processed" / "photos_cleaned.csv"
OUTPUT_DIR = DATA_DIR / "outputs"

MIN_CLASS_SIZE = 8
PERMUTATIONS = 15

_METRIC_NOTE = (
    "Absolute accuracy is NOT interpretable: it depends on the class mix and on "
    "how many colors the lexicon distinguishes. Read only (a) the delta versus "
    "v1 and (b) the margin over the permutation floor. The labeled subset is "
    "also biased -- photographers name a color mainly for distinctive specimens."
)

#: Ablations. Each is a set of ColorParams overrides applied to the defaults.
VARIANTS: dict[str, dict[str, object]] = {
    "v1": {},  # handled specially: the legacy octree extractor
    "v2": {},
    "v2-noborder": {"use_background": False},
    "v2-nosmooth": {"use_smoothness": False},
    "v2-nocenter": {"use_center": False},
    "v2-nocompact": {"use_compactness": False},
    "v2-nogate": {"weight_quantile": 1.0},
    "v2-sharpness": {"invert_smoothness": True},
}
DEFAULT_VARIANTS = ("v1", "v2")


# --------------------------------------------------------------------------- #
# Labeled set
# --------------------------------------------------------------------------- #
def build_labeled_set(sample: int | None, seed: int) -> pd.DataFrame:
    """Photos whose caption states exactly one color and whose image is on disk."""
    df = pd.read_csv(PHOTOS_CLEANED_CSV)
    df["caption"] = df["full_caption"].fillna("").astype(str)
    df["label"] = df["caption"].map(single_color_label)
    df = df.dropna(subset=["label"]).copy()
    logger.info("captions with exactly one stated color: %d", len(df))

    by_date = index_local_images(IMAGES_DIR)
    paths: list[str | None] = []
    for row in df.itertuples(index=False):
        local = match_local_file(
            str(getattr(row, "date", "")), str(getattr(row, "photo_url", "")), by_date
        )
        paths.append(str(local) if local is not None else None)
    df["path"] = paths
    df = df.dropna(subset=["path"]).copy()
    logger.info("of those, with a local image: %d", len(df))

    df["id"] = df["photo_url"].astype(str).map(photo_id)
    df["genus"] = (
        df["first_species"].fillna("").astype(str).map(genus_of).astype("object")
    )

    counts = df["label"].value_counts()
    keep = counts[counts >= MIN_CLASS_SIZE].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        logger.info("dropping classes with n < %d: %s", MIN_CLASS_SIZE, dropped)
    df = df[df["label"].isin(keep)].copy()

    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed).copy()
        logger.info("subsampled to %d photos", len(df))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _extract_one(args: tuple[str, str, str]) -> dict[str, object] | None:
    """Worker: (photo_id, path, variant) -> measured columns."""
    pid, path_str, variant = args
    path = Path(path_str)
    try:
        if variant == "v1":
            color = extract_legacy_color(path)
            return {
                "id": pid,
                "color": color,
                "subject_color": color,
                "frame_color": color,
                "confidence": 1.0,
                "separation": 0.0,
                "background": "",
                "source": "frame",
                "swatches": color,
            }
        params = replace(ColorParams(), **VARIANTS[variant])  # type: ignore[arg-type]
        result = extract_photo_color(path, params)
    except ColorExtractionError as exc:
        logger.warning("skipping %s: %s", path, exc)
        return None

    return {
        "id": pid,
        "color": result.color,
        "subject_color": result.subject_color,
        "frame_color": result.frame_color,
        "confidence": result.confidence,
        "separation": result.separation,
        "background": result.background,
        "source": result.source,
        "swatches": "|".join(result.swatches),
    }


def measure_variant(
    df: pd.DataFrame, variant: str, workers: int | None, refresh: bool
) -> pd.DataFrame:
    """Measure one variant over the labeled set, caching raw results to disk."""
    cache_path = OUTPUT_DIR / f"color_eval_raw_{variant}.csv"
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path)
        if set(df["id"]).issubset(set(cached["id"])):
            logger.info("[%s] reusing cached measurements", variant)
            return cached[cached["id"].isin(df["id"])].copy()

    jobs = [(str(r.id), str(r.path), variant) for r in df.itertuples(index=False)]
    logger.info("[%s] measuring %d photos...", variant, len(jobs))
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_extract_one, jobs, chunksize=16), start=1):
            if row is not None:
                rows.append(row)
            if i % 500 == 0:
                logger.info("[%s]   ...%d/%d", variant, i, len(jobs))

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def _fold_ids(n: int, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    assignment = np.empty(n, dtype=np.intp)
    assignment[order] = np.arange(n) % folds
    return assignment


def prototype_cv(
    colors: Sequence[str],
    labels: Sequence[str],
    *,
    folds: int = 5,
    seed: int = 0,
) -> tuple[float, np.ndarray, list[str]]:
    """Nearest-class-prototype accuracy under k-fold CV, in OKLab.

    The class prototype is the *median* extracted color of the training fold, so
    the metric never assumes an absolute exposure. A hand-authored color anchor
    would: a white cap photographed in forest shade sits at OKLab L~0.62 while a
    nominal "white" anchor is ~0.95, so anchor-matching scores it as grey and
    ranks the extractors backwards.
    """
    lab = np.array([hex_to_oklab(c) for c in colors])
    y = np.asarray(labels)
    classes = sorted(set(y.tolist()))
    class_index = {c: i for i, c in enumerate(classes)}

    fold = _fold_ids(len(y), folds, seed)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)

    for f in range(folds):
        train, test = fold != f, fold == f
        if not test.any():
            continue
        prototypes = []
        present = []
        for cls in classes:
            members = train & (y == cls)
            if members.any():
                prototypes.append(np.median(lab[members], axis=0))
                present.append(cls)
        if not prototypes:
            continue
        proto = np.array(prototypes)
        dist = ((lab[test][:, None, :] - proto[None, :, :]) ** 2).sum(axis=2)
        predicted = [present[i] for i in dist.argmin(axis=1)]
        for truth, guess in zip(y[test], predicted, strict=True):
            confusion[class_index[truth], class_index[guess]] += 1

    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    return (correct / total if total else 0.0), confusion, classes


def permutation_floor(
    colors: Sequence[str], labels: Sequence[str], *, folds: int, seed: int
) -> float:
    """Accuracy when the labels carry no information -- the real chance level."""
    rng = np.random.default_rng(seed)
    scores = []
    shuffled = list(labels)
    for _ in range(PERMUTATIONS):
        rng.shuffle(shuffled)
        scores.append(prototype_cv(colors, shuffled, folds=folds, seed=seed)[0])
    return float(np.mean(scores))


def genus_coherence(colors: Sequence[str], genera: Sequence[str | None]) -> float:
    """Mean within-genus OKLab distance / mean between-genus distance. Lower is better.

    Unsupervised, so unlike the caption metric it covers photos whose captions
    never mention a color -- the large majority of the corpus.
    """
    lab = np.array([hex_to_oklab(c) for c in colors])
    g = np.array([x if isinstance(x, str) and x else "" for x in genera])
    usable = g != ""
    lab, g = lab[usable], g[usable]
    if len(lab) < 2:
        return float("nan")

    counts = pd.Series(g).value_counts()
    keep = np.isin(g, counts[counts >= 15].index)
    lab, g = lab[keep], g[keep]
    if len(lab) < 2:
        return float("nan")

    dist = np.sqrt(((lab[:, None, :] - lab[None, :, :]) ** 2).sum(axis=2))
    same = g[:, None] == g[None, :]
    off_diagonal = ~np.eye(len(lab), dtype=bool)
    within = dist[same & off_diagonal]
    between = dist[~same]
    if within.size == 0 or between.size == 0:
        return float("nan")
    return float(within.mean() / between.mean())


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #
def sweep_thresholds(merged: pd.DataFrame, *, folds: int, seed: int) -> pd.DataFrame:
    """Grid-search the confidence/separation fallback gate.

    Cheap because the raw measurement retains both the subject and frame colors,
    so no image is re-decoded.
    """
    rows: list[dict[str, float]] = []
    for min_conf in (0.0, 0.20, 0.30, 0.40, 0.50, 0.60):
        for min_sep in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
            trustworthy = (merged["confidence"] >= min_conf) & (
                merged["separation"] >= min_sep
            )
            color = np.where(
                trustworthy, merged["subject_color"], merged["frame_color"]
            )
            accuracy, _, _ = prototype_cv(
                list(color), list(merged["label"]), folds=folds, seed=seed
            )
            rows.append(
                {
                    "min_confidence": min_conf,
                    "min_separation": min_sep,
                    "subject_share": float(trustworthy.mean()),
                    "cv_accuracy": accuracy,
                }
            )
    return pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)


# --------------------------------------------------------------------------- #
# Contact sheet
# --------------------------------------------------------------------------- #
def write_contact_sheet(
    merged: pd.DataFrame, v1: pd.DataFrame, n_random: int, n_worst: int, seed: int
) -> Path:
    """A visual A/B: thumbnail, v1 swatch, v2 swatch, caption, stated color."""
    lab = np.array([hex_to_oklab(c) for c in merged["color"]])
    prototypes = {
        cls: np.median(lab[(merged["label"] == cls).to_numpy()], axis=0)
        for cls in sorted(set(merged["label"]))
    }
    error = np.array(
        [
            float(np.sqrt(((lab[i] - prototypes[merged["label"].iloc[i]]) ** 2).sum()))
            for i in range(len(merged))
        ]
    )
    merged = merged.assign(error=error)
    legacy = dict(zip(v1["id"], v1["color"], strict=False))

    worst = merged.nlargest(min(n_worst, len(merged)), "error")
    rest = merged.drop(worst.index)
    picks = rest.sample(n=min(n_random, len(rest)), random_state=seed)

    # Write small thumbnails next to the sheet: the originals average 3 MB, so
    # referencing them directly makes the page unopenable.
    thumb_dir = OUTPUT_DIR / "qa_thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    def thumbnail(source: str, pid: str) -> str:
        out = thumb_dir / f"{pid}.jpg"
        if not out.exists():
            try:
                with Image.open(source) as im:
                    im.draft("RGB", (512, 512))
                    small = im.convert("RGB")
                    small.thumbnail((320, 320), Image.Resampling.LANCZOS)
                    small.save(out, quality=82)
            except OSError as exc:
                logger.warning("thumbnail failed for %s: %s", source, exc)
                return ""
        return f"qa_thumbs/{out.name}"

    def section(title: str, note: str, frame: pd.DataFrame) -> str:
        cards = []
        for row in frame.itertuples(index=False):
            rel = thumbnail(str(row.path), str(row.id))
            cards.append(
                f"""<figure class="card">
  <img loading="lazy" src="{rel}" alt="">
  <div class="bars">
    <div class="bar" style="background:{legacy.get(row.id, "#888")}"><span>v1</span></div>
    <div class="bar" style="background:{row.color}"><span>v2</span></div>
  </div>
  <figcaption><b>{row.label}</b> &middot; {row.source}
    &middot; conf {row.confidence:.2f} &middot; sep {row.separation:.2f}
    <br>{str(row.caption)[:150]}</figcaption>
</figure>"""
            )
        return (
            f"<h2>{title}</h2><p class='note'>{note}</p>"
            f"<div class='grid'>{''.join(cards)}</div>"
        )

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Photo color extraction QA</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; background:#14161a; color:#e6e8ec;
        margin:0; padding:24px; }}
 h1 {{ font-size:20px; }} h2 {{ font-size:16px; margin-top:32px; }}
 .note {{ color:#9aa3b2; max-width:70ch; }}
 .grid {{ display:grid; gap:14px;
          grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); }}
 .card {{ margin:0; background:#1c1f26; border-radius:8px; overflow:hidden; }}
 .card img {{ width:100%; height:150px; object-fit:cover; display:block; }}
 .bars {{ display:flex; height:26px; }}
 .bar {{ flex:1; display:flex; align-items:center; justify-content:center; }}
 .bar span {{ font-size:10px; font-weight:700; color:#000;
              background:rgba(255,255,255,.65); padding:0 4px; border-radius:3px; }}
 figcaption {{ padding:8px; font-size:11px; color:#b9c0cc; }}
</style>
<h1>Photo color extraction &mdash; v1 (whole-frame octree) vs v2 (subject-weighted)</h1>
<p class="note">Left bar is the old extractor, right bar is the new one. The
right bar should look like the mushroom; the left bar usually looks like the
ground. &ldquo;{_METRIC_NOTE}&rdquo;</p>
{section("Worst v2 cases", "Furthest from their class prototype -- inspect these for systematic failure modes.", worst)}
{section("Random sample", "An unbiased look at typical behaviour.", picks)}
"""
    path = OUTPUT_DIR / "color_eval_contact_sheet.html"
    path.write_text(html, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument(
        "--ablate", action="store_true", help="measure every prior ablation"
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached measurements"
    )
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--n-worst", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = list(VARIANTS) if args.ablate else args.variants.split(",")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {unknown}")

    labeled = build_labeled_set(args.sample, args.seed)
    logger.info(
        "evaluation set: %d photos, %d classes", len(labeled), labeled.label.nunique()
    )
    logger.info("class balance: %s", labeled["label"].value_counts().to_dict())

    summary: list[dict[str, object]] = []
    by_class: list[dict[str, object]] = []
    measured: dict[str, pd.DataFrame] = {}

    for variant in variants:
        raw = measure_variant(labeled, variant, args.workers, args.refresh)
        merged = labeled.merge(raw, on="id", how="inner")
        measured[variant] = merged

        accuracy, confusion, classes = prototype_cv(
            list(merged["color"]),
            list(merged["label"]),
            folds=args.folds,
            seed=args.seed,
        )
        floor = permutation_floor(
            list(merged["color"]),
            list(merged["label"]),
            folds=args.folds,
            seed=args.seed,
        )
        majority = float(merged["label"].value_counts().iloc[0] / len(merged))
        coherence = genus_coherence(list(merged["color"]), list(merged["genus"]))

        summary.append(
            {
                "variant": variant,
                "n": len(merged),
                "cv_accuracy": round(accuracy, 4),
                "permutation_floor": round(floor, 4),
                "lift_over_chance": round(accuracy / floor, 2)
                if floor
                else float("nan"),
                "majority_baseline": round(majority, 4),
                "genus_coherence": round(coherence, 4),
                "subject_share": round(
                    float((merged["source"] == "subject").mean()), 4
                ),
            }
        )

        pd.DataFrame(confusion, index=classes, columns=classes).to_csv(
            OUTPUT_DIR / f"color_eval_confusion_{variant}.csv"
        )
        for i, cls in enumerate(classes):
            support = int(confusion[i].sum())
            by_class.append(
                {
                    "variant": variant,
                    "color": cls,
                    "n": support,
                    "recall": round(confusion[i, i] / support, 3) if support else 0.0,
                }
            )
        logger.info(
            "[%s] cv=%.3f  chance=%.3f  majority=%.3f  genus_coherence=%.3f",
            variant,
            accuracy,
            floor,
            majority,
            coherence,
        )

    summary_df = pd.DataFrame(summary).sort_values("cv_accuracy", ascending=False)
    summary_path = OUTPUT_DIR / "color_eval_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {_METRIC_NOTE}\n")
        summary_df.to_csv(fh, index=False)

    class_df = pd.DataFrame(by_class).pivot(
        index="color", columns="variant", values="recall"
    )
    support = pd.DataFrame(by_class).groupby("color")["n"].max()
    class_df.insert(0, "n", support)
    class_df.to_csv(OUTPUT_DIR / "color_eval_by_class.csv")

    print("\n=== summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== per-class recall ===")
    print(class_df.to_string())

    if "v2" in measured:
        sweep = sweep_thresholds(measured["v2"], folds=args.folds, seed=args.seed)
        sweep.to_csv(OUTPUT_DIR / "color_eval_threshold_sweep.csv", index=False)
        print("\n=== fallback threshold sweep (top 8) ===")
        print(sweep.head(8).to_string(index=False))

    if args.contact_sheet and "v2" in measured and "v1" in measured:
        path = write_contact_sheet(
            measured["v2"], measured["v1"], args.n_random, args.n_worst, args.seed
        )
        print(f"\nContact sheet: {path}")

    print(f"\nNOTE: {_METRIC_NOTE}")


if __name__ == "__main__":
    main()
