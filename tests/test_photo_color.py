"""Tests for subject-weighted photo color extraction.

Every fixture is synthesised in-process -- no sample images, no network. The
scenarios encode the failure modes of the whole-frame octree extractor this
replaced, so a regression in the priors shows up as a specific failing test
rather than a vague drift in the visualization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from texas_mushrooms.pipeline.color import (
    ColorExtractionError,
    ColorParams,
    PhotoColor,
    extract_legacy_color,
    extract_photo_color,
    hex_to_oklab,
    oklab_to_hex,
    oklab_to_srgb,
    srgb_to_oklab,
    weighted_kmeans,
)

LITTER = (0x6B, 0x5A, 0x45)
DARK_LITTER = (0x3A, 0x32, 0x28)
ORANGE = (0xD9, 0x72, 0x1E)
WHITE = (0xF2, 0xF0, 0xEA)
PALE = (0xD9, 0xD3, 0xC8)


def _disc(
    size: int,
    background: tuple[int, int, int],
    subject: tuple[int, int, int],
    *,
    center: tuple[float, float] = (0.5, 0.5),
    radius_frac: float = 0.22,
    noise: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """A solid disc of ``subject`` on a field of ``background``."""
    arr = np.zeros((size, size, 3), dtype=np.float64)
    arr[:, :] = background
    if noise:
        rng = np.random.default_rng(seed)
        arr += rng.uniform(-noise, noise, size=arr.shape)

    ys = np.arange(size)[:, None]
    xs = np.arange(size)[None, :]
    cy, cx = center[0] * size, center[1] * size
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= (radius_frac * size) ** 2
    arr[mask] = subject
    return np.clip(arr, 0, 255).astype(np.uint8)


def _write(tmp_path: Path, arr: np.ndarray, name: str = "photo.jpg") -> Path:
    path = tmp_path / name
    Image.fromarray(arr).save(path, quality=95)
    return path


def _measure(tmp_path: Path, arr: np.ndarray, **overrides: object) -> PhotoColor:
    params = ColorParams(**overrides)  # type: ignore[arg-type]
    return extract_photo_color(_write(tmp_path, arr), params)


def _distance(a: str, b: str) -> float:
    return float(np.sqrt(((hex_to_oklab(a) - hex_to_oklab(b)) ** 2).sum()))


# --------------------------------------------------------------------------- #
# Color space
# --------------------------------------------------------------------------- #
def test_oklab_roundtrip_is_lossless_to_8_bit() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.random((4096, 3))
    back = oklab_to_srgb(srgb_to_oklab(rgb))
    np.testing.assert_allclose(back, rgb, atol=1.0 / 255.0)


def test_oklab_anchors() -> None:
    white = srgb_to_oklab(np.array([1.0, 1.0, 1.0]))
    assert white[0] == pytest.approx(1.0, abs=1e-6)
    assert abs(white[1]) < 1e-6 and abs(white[2]) < 1e-6

    grey = srgb_to_oklab(np.array([0.5, 0.5, 0.5]))
    assert abs(grey[1]) < 1e-6 and abs(grey[2]) < 1e-6

    assert oklab_to_hex(hex_to_oklab("#d9721e")) == "#d9721e"


def test_oklab_handles_out_of_gamut_without_nan() -> None:
    # Cluster centroids can land outside the sRGB gamut; they must still render.
    wild = np.array([[1.4, 0.9, -0.9], [-0.3, -0.5, 0.6]])
    out = oklab_to_srgb(wild)
    assert np.isfinite(out).all()
    assert ((out >= 0.0) & (out <= 1.0)).all()


# --------------------------------------------------------------------------- #
# Subject isolation
# --------------------------------------------------------------------------- #
def test_saturated_subject_beats_drab_background(tmp_path: Path) -> None:
    result = _measure(tmp_path, _disc(300, LITTER, ORANGE))
    assert result.source == "subject"
    assert _distance(result.color, "#d9721e") < 0.10


def test_white_subject_on_dark_litter(tmp_path: Path) -> None:
    """The regression the old extractor categorically could not pass.

    Its ``MIN_INTERESTING_SATURATION`` floor made white unreachable: a white
    swatch is drab by definition, so it always lost to the litter.
    """
    result = _measure(tmp_path, _disc(300, DARK_LITTER, WHITE))
    assert result.source == "subject"
    assert hex_to_oklab(result.color)[0] > 0.85


def test_smooth_subject_beats_textured_background(tmp_path: Path) -> None:
    """Fails if anyone re-inverts the smoothness prior into a sharpness prior.

    Leaf litter is textured and mushroom flesh is smooth, so rewarding local
    gradient energy selects the ground.
    """
    arr = _disc(300, LITTER, PALE, noise=40)
    result = _measure(tmp_path, arr)
    assert _distance(result.color, "#d9d3c8") < _distance(result.color, "#6b5a45")


def test_subject_touching_frame_edge_is_not_background(tmp_path: Path) -> None:
    """Documents the known-degraded case: the subject contaminates the border sample."""
    arr = _disc(300, LITTER, ORANGE, center=(0.5, 0.85))
    result = _measure(tmp_path, arr)
    assert _distance(result.color, "#6b5a45") > 0.05


def test_uniform_image_is_measured_without_blowing_up(tmp_path: Path) -> None:
    """Guards the divide-by-zero paths: w.sum() == 0 and percentile(energy) == 0.

    On a uniform image the subject, frame and background colors all coincide, so
    which branch answers does not matter -- only that the answer is that color
    and that nothing goes NaN.
    """
    arr = np.full((200, 200, 3), LITTER, dtype=np.uint8)
    result = _measure(tmp_path, arr)
    assert _distance(result.color, "#6b5a45") < 0.08
    assert _distance(result.background, "#6b5a45") < 0.08
    assert np.isfinite(result.confidence) and np.isfinite(result.separation)
    assert result.separation < 0.02  # nothing stands out from the background


def test_low_separation_falls_back_to_frame(tmp_path: Path) -> None:
    """A subject barely distinguishable from the ground should not be forced vivid."""
    near_litter = (0x70, 0x5F, 0x4A)
    arr = _disc(300, LITTER, near_litter)
    result = _measure(tmp_path, arr, min_separation=0.25)
    assert result.source == "frame"
    assert result.color == result.frame_color


def test_blown_highlights_are_suppressed(tmp_path: Path) -> None:
    """A pure-white specular patch must not become the subject color."""
    arr = _disc(300, LITTER, ORANGE)
    arr[0:40, 0:40] = 255  # a clipped highlight in the corner
    result = _measure(tmp_path, arr)
    assert hex_to_oklab(result.color)[0] < 0.95


# --------------------------------------------------------------------------- #
# Contracts relied on elsewhere
# --------------------------------------------------------------------------- #
def test_fallback_is_the_legacy_color_not_a_frame_kmeans(tmp_path: Path) -> None:
    """The fallback must stay the octree color.

    Measured over the caption-labeled set: the legacy octree color scores 0.134
    against a 0.038 chance floor, a whole-frame k-means only 0.054. Swapping in
    the "cleaner" k-means silently erased the entire gain from subject isolation.
    """
    path = _write(tmp_path, _disc(300, LITTER, ORANGE))
    assert extract_photo_color(path).frame_color == extract_legacy_color(path)


def test_compactness_is_off_by_default() -> None:
    """Largest-connected-blob sounds principled but measured 0.213 -> 0.170.

    Real subjects are split by stipes, gaps and overlapping caps, so the largest
    blob routinely discards most of the mushroom.
    """
    assert ColorParams().use_compactness is False


def test_extraction_is_deterministic(tmp_path: Path) -> None:
    path = _write(tmp_path, _disc(300, LITTER, ORANGE))
    assert extract_photo_color(path) == extract_photo_color(path)


def test_swatches_match_the_frontend_contract(tmp_path: Path) -> None:
    """``Seasons.tsx`` renders a gradient chip from >= 2 swatches."""
    result = _measure(tmp_path, _disc(300, LITTER, ORANGE))
    assert len(result.swatches) == 3
    for swatch in result.swatches:
        assert len(swatch) == 7 and swatch[0] == "#"
        assert all(c in "0123456789abcdef" for c in swatch[1:])


def test_corrupt_file_raises_typed_error(tmp_path: Path) -> None:
    bad = tmp_path / "truncated.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    with pytest.raises(ColorExtractionError):
        extract_photo_color(bad)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def test_weighted_kmeans_follows_the_weights() -> None:
    """A small but heavily-weighted group must win over a large light one."""
    x = np.vstack([np.zeros((900, 3)), np.ones((100, 3))])
    w = np.concatenate([np.full(900, 0.01), np.full(100, 5.0)])
    centroids, mass = weighted_kmeans(x, w, k=2, seed=0)
    winner = centroids[int(np.argmax(mass))]
    np.testing.assert_allclose(winner, np.ones(3), atol=1e-6)


def test_weighted_kmeans_handles_degenerate_input() -> None:
    x = np.zeros((5, 3))
    centroids, mass = weighted_kmeans(x, np.zeros(5), k=6, seed=0)
    assert centroids.shape == (5, 3)
    assert np.isfinite(centroids).all()
    assert float(mass.sum()) == 0.0
