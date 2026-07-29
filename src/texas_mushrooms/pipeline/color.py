"""Subject-weighted dominant-color extraction for mushroom photographs.

The naive approach -- quantize the whole frame and take the most common bin --
returns the color of the *scene*: soil, leaf litter, shadow. On a forest-floor
photograph the mushroom is a small minority of the pixels, so it never wins a
popularity contest.

This module isolates the subject before measuring its color, using three cheap
priors that need no model and no network:

* **background contrast** -- the outer frame of the photograph is, by
  composition, background. Cluster it into a small color model and down-weight
  every pixel that resembles it. This is what lets a *white* cap win: it is far
  from litter in lightness, not merely in saturation.
* **smoothness** -- leaf litter is the most textured surface in the frame while
  mushroom flesh is smooth, so local Laplacian energy is an inverted subject
  cue. (Using it the other way round, as a "focus" prior, actively hurts.)
* **center bias** -- a mild nudge, never a mandate.

The weights multiply, the top quantile of pixels is kept, and a weighted k-means
in OKLab picks the dominant subject color. Where the subject cannot be separated
from the background with confidence -- genuinely drab grey/brown/black species,
where "distance from background" is definitionally the wrong signal -- the
extractor falls back to the whole-frame color rather than inventing a vivid one.

All colors are handled in OKLab so that distances and means are perceptually
meaningful. Extraction is deterministic: seeded subsampling, seeded k-means++.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: Written into the color cache. Bump on any retune so stale rows are not reused.
ALGO_VERSION = "v2-oklab-1"


class ColorExtractionError(RuntimeError):
    """Raised when an image cannot be decoded or measured."""


# --------------------------------------------------------------------------- #
# sRGB <-> OKLab  (Bjorn Ottosson's transform)
# --------------------------------------------------------------------------- #
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def _srgb_to_linear(c: FloatArray) -> FloatArray:
    safe = np.clip(c, 0.0, 1.0)
    return np.asarray(
        np.where(safe <= 0.04045, safe / 12.92, ((safe + 0.055) / 1.055) ** 2.4),
        dtype=np.float64,
    )


def _linear_to_srgb(c: FloatArray) -> FloatArray:
    safe = np.clip(c, 0.0, 1.0)
    return np.asarray(
        np.where(safe <= 0.0031308, safe * 12.92, 1.055 * safe ** (1.0 / 2.4) - 0.055),
        dtype=np.float64,
    )


def srgb_to_oklab(rgb: FloatArray) -> FloatArray:
    """Convert sRGB in [0, 1] to OKLab. Operates on any ``(..., 3)`` array."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    lms = lin @ _M1.T
    # Clip before the cube root: out-of-gamut inputs would otherwise go complex.
    lms = np.cbrt(np.clip(lms, 0.0, None))
    return np.asarray(lms @ _M2.T, dtype=np.float64)


def oklab_to_srgb(lab: FloatArray) -> FloatArray:
    """Convert OKLab back to sRGB in [0, 1], clipped to gamut."""
    lms = np.asarray(lab, dtype=np.float64) @ _M2_INV.T
    lin = np.clip(lms, 0.0, None) ** 3
    return np.clip(_linear_to_srgb(np.asarray(lin @ _M1_INV.T)), 0.0, 1.0)


def oklab_to_hex(lab: FloatArray) -> str:
    """Render a single OKLab triple as ``#rrggbb``."""
    rgb = oklab_to_srgb(np.asarray(lab, dtype=np.float64).reshape(3))
    r, g, b = (int(round(float(v) * 255.0)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_oklab(value: str) -> FloatArray:
    """Parse ``#rrggbb`` into an OKLab triple."""
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"not a #rrggbb color: {value!r}")
    rgb = np.array([int(text[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)
    return srgb_to_oklab(rgb / 255.0)


# --------------------------------------------------------------------------- #
# Parameters and result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColorParams:
    """Tunable knobs for :func:`extract_photo_color`."""

    target_px: int = 224
    """Longest side after draft-decode + LANCZOS resize."""

    k: int = 6
    """Subject clusters."""

    bg_k: int = 5
    """Centroids in the background color model."""

    border_frac: float = 0.08
    """Fraction of each side treated as guaranteed background."""

    require_multi_edge: bool = True
    """Discard background colors that occur on only one of the four border edges.

    A mushroom running off the frame poisons the edge it touches and would
    otherwise be modelled as background -- and then rejected as background. Real
    background wraps the frame; an intruding subject does not.
    """

    edge_presence: float = 0.15
    """Share of an edge's pixels a color needs to count as present on that edge."""

    bg_sigma: float = 0.06
    """OKLab distance scale for the background-similarity falloff."""

    smooth_pct: float = 90.0
    """Percentile of local Laplacian energy used to normalize the texture map."""

    center_floor: float = 0.40
    center_sigma: float = 0.55

    specular_l: float = 0.97
    """Pixels brighter than this are blown-out highlights; weight goes to zero."""

    sky_l_min: float = 0.72
    sky_penalty: float = 0.20
    """Bright blue pixels are usually sky gaps, which the border prior loves."""

    weight_quantile: float = 0.20
    """Fraction of pixels kept before clustering. 0.10 starves k-means."""

    use_compactness: bool = False
    """Keep only the largest connected blob of retained pixels.

    Off by default: it sounds principled but measurably hurts (subject-color
    accuracy 0.213 -> 0.170 over the caption-labeled set). Real subjects are
    split by stipes, gaps and overlapping caps, so "largest blob" routinely
    discards most of the mushroom.
    """

    max_pixels: int = 12_000
    seed: int = 0

    # --- ablation switches, used by scripts/eval_photo_colors.py --- #
    use_background: bool = True
    use_smoothness: bool = True
    use_center: bool = True
    invert_smoothness: bool = False
    """Reproduce the *sharpness* prior, which measurably selects leaf litter."""

    min_confidence: float = 0.20
    """Below this share of retained weight, fall back to the whole-frame color.

    Swept optimum over the caption-labeled set (0.232, versus 0.213 with no gate
    at all). It fires on <1% of photos -- images where the retained pixels
    fragment into six equally-sized clusters, i.e. no coherent subject was found.
    """

    min_separation: float = 0.0
    """Below this OKLab distance from the background, fall back. Off by default.

    Intuitively this should protect drab species, whose color really is the
    litter's. Empirically it does not pay for itself: every non-zero value costs
    accuracy (0.232 at 0.0, 0.211 at 0.05, 0.217 at 0.10, 0.190 at 0.15) because
    it also discards good measurements of dark and muted mushrooms. Exposed as
    ``--min-separation`` for anyone who wants the conservative behaviour.
    """


@dataclass(frozen=True)
class PhotoColor:
    """Measured colors for one photograph.

    ``color`` is what ships to the web app. ``subject_color`` and ``frame_color``
    are both retained so the evaluation harness can re-apply different fallback
    thresholds without re-decoding 60 GB of JPEGs.
    """

    color: str
    subject_color: str
    frame_color: str
    swatches: tuple[str, ...]
    confidence: float
    separation: float
    background: str
    source: str
    algo: str = ALGO_VERSION


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def load_oklab(path: Path, params: ColorParams) -> FloatArray:
    """Decode an image to an ``(H, W, 3)`` OKLab array at working resolution.

    ``draft()`` runs before ``convert()`` so libjpeg can DCT-downscale during
    decode; on the ~3 MB originals in this dataset that dominates the runtime.
    """
    try:
        with Image.open(path) as im:
            im.draft("RGB", (params.target_px * 2, params.target_px * 2))
            rgb = im.convert("RGB")
            rgb.thumbnail(
                (params.target_px, params.target_px), Image.Resampling.LANCZOS
            )
            arr = np.asarray(rgb, dtype=np.float64) / 255.0
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same to us
        raise ColorExtractionError(f"cannot decode {path}: {exc}") from exc

    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        raise ColorExtractionError(f"unexpected image shape {arr.shape} for {path}")
    return srgb_to_oklab(arr)


# --------------------------------------------------------------------------- #
# Per-pixel subject weights
# --------------------------------------------------------------------------- #
def _smoothness_weight(lab: FloatArray, params: ColorParams) -> FloatArray:
    """High where the image is locally smooth.

    Note the sign: this is a *smoothness* prior, not a sharpness one. Leaf litter
    is the most textured surface in a forest-floor photograph and mushroom flesh
    is among the least, so rewarding texture systematically selects the ground.
    """
    lightness = lab[..., 0]
    energy = np.abs(ndimage.laplace(lightness, mode="nearest"))
    radius = max(3, int(0.03 * max(lightness.shape)))
    local = np.asarray(ndimage.uniform_filter(energy, size=radius, mode="nearest"))
    scale = float(np.percentile(local, params.smooth_pct))
    texture = np.clip(local / (scale + 1e-8), 0.0, 1.0)
    signal = texture if params.invert_smoothness else 1.0 - texture
    return np.asarray(0.1 + 0.9 * signal, dtype=np.float64)


def _border_edges(lab: FloatArray, params: ColorParams) -> list[FloatArray]:
    """The four border bands, each subsampled, as separate pixel arrays."""
    height, width = lab.shape[:2]
    band_y = max(1, int(round(params.border_frac * height)))
    band_x = max(1, int(round(params.border_frac * width)))
    bands = [
        lab[:band_y, :].reshape(-1, 3),
        lab[-band_y:, :].reshape(-1, 3),
        lab[:, :band_x].reshape(-1, 3),
        lab[:, -band_x:].reshape(-1, 3),
    ]
    rng = np.random.default_rng(params.seed)
    edges: list[FloatArray] = []
    for band in bands:
        if band.shape[0] == 0:
            continue
        if band.shape[0] > 750:
            band = band[rng.choice(band.shape[0], size=750, replace=False)]
        edges.append(np.asarray(band, dtype=np.float64))
    return edges


def _background_model(
    lab: FloatArray, params: ColorParams
) -> tuple[FloatArray, FloatArray]:
    """Cluster the border into a background color model.

    Colors confined to a single edge are dropped: real background wraps the
    frame, while a mushroom running off one side does not. Without this, an
    edge-touching subject is modelled as background and then rejected as
    background -- it disappears from its own photograph.
    """
    edges = _border_edges(lab, params)
    if not edges:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)

    border = np.vstack(edges)
    centroids, mass = weighted_kmeans(
        border,
        np.ones(border.shape[0]),
        min(params.bg_k, border.shape[0]),
        seed=params.seed,
    )
    if not params.require_multi_edge or len(edges) < 3 or centroids.shape[0] < 3:
        return centroids, mass

    presence = np.zeros(centroids.shape[0], dtype=np.intp)
    for edge in edges:
        dist = ((edge[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        share = np.bincount(dist.argmin(axis=1), minlength=centroids.shape[0]) / len(
            edge
        )
        presence += (share >= params.edge_presence).astype(np.intp)

    # A single surviving color is a legitimate answer (a uniform background);
    # only an empty result means the test was too strict to be useful.
    keep = (presence >= 2) & (mass > 0)
    if not keep.any():
        return centroids, mass
    return centroids[keep], mass[keep]


def _background_weight(
    lab: FloatArray, params: ColorParams
) -> tuple[FloatArray, FloatArray]:
    """Return per-pixel novelty vs. the frame's background model, and that model's mode."""
    height, width = lab.shape[:2]
    centroids, mass = _background_model(lab, params)
    if centroids.shape[0] == 0:
        return np.ones((height, width), dtype=np.float64), lab.reshape(-1, 3).mean(0)

    flat = lab.reshape(-1, 3)
    dist = np.sqrt(((flat[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    nearest = dist.min(axis=1).reshape(height, width)

    novelty = 1.0 - np.exp(-(nearest**2) / (2.0 * params.bg_sigma**2))
    weight = 0.05 + 0.95 * novelty
    dominant = centroids[int(np.argmax(mass))]
    return np.asarray(weight, dtype=np.float64), np.asarray(dominant, dtype=np.float64)


def _center_weight(shape: tuple[int, int], params: ColorParams) -> FloatArray:
    height, width = shape
    ys = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) / max(
        height / 2.0, 1.0
    )
    xs = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) / max(
        width / 2.0, 1.0
    )
    r2 = ys[:, None] ** 2 + xs[None, :] ** 2
    span = 1.0 - params.center_floor
    falloff = np.exp(-r2 / (2.0 * params.center_sigma**2))
    return np.asarray(params.center_floor + span * falloff, dtype=np.float64)


def _highlight_penalty(lab: FloatArray, params: ColorParams) -> FloatArray:
    """Suppress blown-out highlights and bright sky gaps.

    The background-contrast prior rewards anything unlike leaf litter, and a
    patch of sky between the trees qualifies -- one prototype photo of white
    *Marasmius* came back sky blue. The blue wedge is restricted to ``a < 0.04``
    so genuinely violet species (``a > 0``) are untouched.
    """
    lightness = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    penalty = np.ones(lightness.shape, dtype=np.float64)
    sky = (lightness > params.sky_l_min) & (b < -0.02) & (a < 0.04)
    penalty[sky] = params.sky_penalty
    penalty[lightness > params.specular_l] = 0.0
    return penalty


# --------------------------------------------------------------------------- #
# Weighted k-means
# --------------------------------------------------------------------------- #
def weighted_kmeans(
    x: FloatArray,
    w: FloatArray,
    k: int,
    *,
    iters: int = 25,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Weighted k-means with weighted k-means++ init.

    Returns ``(centroids (k, d), mass (k,))`` where ``mass[j]`` is the total
    weight assigned to cluster ``j``. Empty clusters keep their initial position
    rather than being reseeded -- at ``k = 6`` the reseeding buys nothing and
    costs determinism.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = x.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)

    # --- weighted k-means++ ---
    centroids = np.empty((k, x.shape[1]), dtype=np.float64)
    probs = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    centroids[0] = x[rng.choice(n, p=probs)]
    closest = ((x - centroids[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        score = closest * w
        total = float(score.sum())
        pick = rng.choice(n, p=score / total) if total > 0 else rng.integers(n)
        centroids[j] = x[pick]
        closest = np.minimum(closest, ((x - centroids[j]) ** 2).sum(axis=1))

    # --- Lloyd ---
    labels = np.zeros(n, dtype=np.intp)
    for _ in range(iters):
        dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.asarray(dist.argmin(axis=1), dtype=np.intp)
        moved = centroids.copy()
        for j in range(k):
            members = labels == j
            total = float(w[members].sum())
            if total > 1e-9:
                moved[j] = (x[members] * w[members, None]).sum(axis=0) / total
        if np.allclose(moved, centroids, atol=1e-5):
            centroids = moved
            break
        centroids = moved

    dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.asarray(dist.argmin(axis=1), dtype=np.intp)
    mass = np.array([float(w[labels == j].sum()) for j in range(k)], dtype=np.float64)
    return centroids, mass


def _subsample(
    x: FloatArray, w: FloatArray, limit: int, seed: int
) -> tuple[FloatArray, FloatArray]:
    if x.shape[0] <= limit:
        return x, w
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=limit, replace=False)
    return x[idx], w[idx]


def _rank_swatches(
    centroids: FloatArray, mass: FloatArray, count: int
) -> tuple[str, ...]:
    order = np.argsort(-mass)[:count]
    return tuple(oklab_to_hex(centroids[i]) for i in order if mass[i] > 0)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract_photo_color(path: Path, params: ColorParams | None = None) -> PhotoColor:
    """Measure the dominant color of the photograph's *subject*."""
    params = params or ColorParams()
    lab = load_oklab(path, params)
    return measure_oklab(lab, params, fallback_color=extract_legacy_color(path))


def measure_oklab(
    lab: FloatArray,
    params: ColorParams | None = None,
    *,
    fallback_color: str | None = None,
) -> PhotoColor:
    """Core measurement, split out from decoding so tests can feed arrays directly.

    ``fallback_color`` is used where the subject cannot be separated from the
    background. It must be the *legacy whole-frame octree* color: measured over
    the caption-labeled set it scores 0.134 against a 0.038 chance floor, while a
    whole-frame k-means scores 0.054 -- essentially nothing. The old extractor's
    saturation rule is crude, but it is the better of the two answers to give up
    with, and a bad fallback silently erases the gain from subject isolation.
    """
    params = params or ColorParams()
    height, width = lab.shape[:2]

    bg_w, bg_centroid = _background_weight(lab, params)
    weight = _highlight_penalty(lab, params)
    if params.use_background:
        weight = weight * bg_w
    if params.use_smoothness:
        weight = weight * _smoothness_weight(lab, params)
    if params.use_center:
        weight = weight * _center_weight((height, width), params)

    keep = _gate(weight, params)
    kept_x = lab[keep]
    kept_w = weight[keep]
    background = oklab_to_hex(bg_centroid)

    if kept_x.shape[0] < params.k or kept_w.sum() <= 0:
        # Nothing separable (uniform or degenerate image).
        frame = fallback_color or background
        return PhotoColor(
            color=frame,
            subject_color=frame,
            frame_color=frame,
            swatches=(frame,),
            confidence=0.0,
            separation=0.0,
            background=background,
            source="frame",
        )

    sub_x, sub_w = _subsample(kept_x, kept_w, params.max_pixels, params.seed)
    centroids, mass = weighted_kmeans(sub_x, sub_w, params.k, seed=params.seed)
    winner = int(np.argmax(mass))
    subject_lab = centroids[winner]
    subject_color = oklab_to_hex(subject_lab)

    total = float(mass.sum())
    confidence = float(mass[winner] / total) if total > 0 else 0.0
    separation = float(np.sqrt(((subject_lab - bg_centroid) ** 2).sum()))
    frame = fallback_color or background

    trustworthy = (
        confidence >= params.min_confidence and separation >= params.min_separation
    )
    return PhotoColor(
        color=subject_color if trustworthy else frame,
        subject_color=subject_color,
        frame_color=frame,
        swatches=_rank_swatches(centroids, mass, 3),
        confidence=confidence,
        separation=separation,
        background=background,
        source="subject" if trustworthy else "frame",
    )


def _gate(weight: FloatArray, params: ColorParams) -> BoolArray:
    """Keep the top-``weight_quantile`` pixels, optionally the largest blob only.

    The hard gate is not polish: background outnumbers subject 20-50x, so soft
    weighting alone lets background mass win the k-means and simply re-derives
    the scene color.
    """
    threshold = float(np.quantile(weight, 1.0 - params.weight_quantile))
    keep: BoolArray = weight >= threshold
    if not params.use_compactness or not keep.any():
        return keep

    labelled, count = ndimage.label(keep)
    if count <= 1:
        return keep
    sizes = np.bincount(np.asarray(labelled).ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return np.asarray(labelled == largest, dtype=bool)


def apply_fallback(result: PhotoColor, params: ColorParams) -> PhotoColor:
    """Re-apply the confidence gate to an already-measured photo.

    Lets the evaluation harness sweep ``min_confidence`` / ``min_separation``
    without re-decoding the images.
    """
    trustworthy = (
        result.confidence >= params.min_confidence
        and result.separation >= params.min_separation
    )
    color = result.subject_color if trustworthy else result.frame_color
    source = "subject" if trustworthy else "frame"
    return replace(result, color=color, source=source)


def extract_legacy_color(path: Path) -> str:
    """The pre-v2 whole-frame octree color: 64px thumbnail, 3-bin fast octree,
    most common swatch unless it is drab, in which case the most saturated.

    Still used as the fallback when the subject cannot be separated, and as the
    A/B baseline in ``scripts/eval_photo_colors.py``. Its saturation rule makes
    white and grey unreachable by construction, which is exactly what v2 fixes.
    """
    import colorsys

    try:
        with Image.open(path) as im:
            im.draft("RGB", (128, 128))
            rgb = im.convert("RGB")
            rgb.thumbnail((64, 64), Image.Resampling.LANCZOS)
            quantized = rgb.quantize(colors=3, method=Image.Quantize.FASTOCTREE)
    except Exception as exc:  # noqa: BLE001
        raise ColorExtractionError(f"cannot decode {path}: {exc}") from exc

    palette = quantized.getpalette() or []
    counts: list[tuple[int, Any]] = sorted(quantized.getcolors() or [], reverse=True)
    swatches: list[tuple[int, int, int]] = []
    for _, idx in counts[:3]:
        base = int(idx) * 3
        swatches.append((palette[base], palette[base + 1], palette[base + 2]))
    if not swatches:
        return "#888888"

    def saturation(c: tuple[int, int, int]) -> float:
        return colorsys.rgb_to_hsv(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)[1]

    chosen = (
        swatches[0]
        if saturation(swatches[0]) >= 0.12
        else max(swatches, key=saturation)
    )
    return "#%02x%02x%02x" % chosen
