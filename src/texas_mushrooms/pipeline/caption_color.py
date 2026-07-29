"""Extract photographer-stated color words from photo captions.

Captions on texasmushrooms.org follow a rigid grammar::

    <subject phrase> on|in|at|from|... <substrate> ... , near <town>. Texas, <date>

Only the *leading clause* -- everything before the first locative preposition --
describes the mushroom itself. Everything after it is substrate and place names,
and that is where the false positives live ("White Oak Bayou", "Black Creek",
"Green Trail"). Splitting on the first preposition removes them by construction,
without needing a dependency parse.

These labels exist only to *score* the color extractor in
``scripts/eval_photo_colors.py``. They never set a photo's displayed color.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #
# canonical class -> surface forms. Each form is matched with word boundaries
# and tolerates an "-ish" suffix, so "yellowish" counts as "yellow" and
# "cinnabar-red" counts as "red" (the hyphen is a word boundary).
_SURFACE_FORMS: dict[str, tuple[str, ...]] = {
    "white": ("white", "whitish", "ivory", "cream", "creamy", "milky-white"),
    "grey": ("grey", "gray", "greyish", "grayish", "silver", "silvery"),
    "black": ("black", "blackish"),
    "brown": ("brown", "brownish", "chocolate"),
    "tan": ("tan", "beige", "buff", "fawn"),
    "red": ("red", "reddish", "crimson", "scarlet", "vermilion", "cinnabar"),
    "orange": ("orange", "orangish", "orangeish", "amber", "salmon", "apricot"),
    "yellow": ("yellow", "yellowish", "lemon", "sulphur", "sulfur"),
    "golden": ("golden", "gold"),
    "green": ("green", "greenish"),
    "olive": ("olive", "olivaceous"),
    "blue": ("blue", "bluish", "azure"),
    "violet": ("violet", "purple", "purplish", "lilac", "lavender", "violaceous"),
    "pink": ("pink", "pinkish", "magenta", "rose", "rosy"),
}

# Surface form -> canonical class.
COLOR_LEXICON: dict[str, str] = {
    form: canonical for canonical, forms in _SURFACE_FORMS.items() for form in forms
}

# Locative prepositions that end the subject phrase. "of" is deliberately absent
# ("Close up of ...", "Underside of ...") -- it introduces the subject, not the
# location.
_PREPOSITIONS = (
    "on",
    "in",
    "at",
    "under",
    "near",
    "from",
    "atop",
    "over",
    "along",
    "beside",
    "among",
    "growing",
    "amongst",
    "beneath",
    "next",
)
_LEAD_SPLIT = re.compile(rf"\s+(?:{'|'.join(_PREPOSITIONS)})\s+", re.IGNORECASE)

# A parenthetical describing the spore print states the print's color, not the
# mushroom's visible color: "(spore print light brown)". Drop those outright.
_SPORE_PRINT_PAREN = re.compile(r"\([^()]*spore\s+print[^()]*\)", re.IGNORECASE)

# A color word immediately followed by one of these is naming a tree, a place or
# a landform, not the fungus: "white oak", "black gum", "Green Trail".
_SUBSTRATE_NOUNS = (
    "oak",
    "oaks",
    "pine",
    "pines",
    "gum",
    "bay",
    "cedar",
    "elm",
    "birch",
    "ash",
    "hickory",
    "maple",
    "willow",
    "cypress",
    "magnolia",
    "sweetgum",
    "spruce",
    "fir",
    "walnut",
    "cherry",
    "beech",
    "creek",
    "bayou",
    "trail",
    "lake",
    "river",
    "mountain",
    "road",
    "hill",
    "county",
    "park",
    "forest",
    "springs",
    "water",
    "sand",
    "sands",
)
_SUBSTRATE_FOLLOWER = re.compile(
    rf"\s+(?:{'|'.join(_SUBSTRATE_NOUNS)})\b", re.IGNORECASE
)

# "rust fungus" is a plant disease, not a color. Only the explicitly adjectival
# forms count.
_RUST_COLOR = re.compile(r"\brust(?:y|-colou?red|\s+colou?red)\b", re.IGNORECASE)

_FORM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{re.escape(form)}\b", re.IGNORECASE), canonical)
    for form, canonical in COLOR_LEXICON.items()
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def leading_clause(caption: str) -> str:
    """Return the subject phrase: everything before the first locative preposition.

    ``"Small orange cup mushrooms Anthracobia maurilabra on a bonfire site on
    Richards Loop Trail..."`` -> ``"Small orange cup mushrooms Anthracobia
    maurilabra"``.
    """
    text = _SPORE_PRINT_PAREN.sub(" ", caption or "")
    return _LEAD_SPLIT.split(text, maxsplit=1)[0].strip()


def caption_color_labels(caption: str) -> frozenset[str]:
    """Return the canonical color classes stated in a caption's leading clause."""
    lead = leading_clause(caption)
    if not lead:
        return frozenset()

    found: set[str] = set()
    for pattern, canonical in _FORM_PATTERNS:
        for match in pattern.finditer(lead):
            # Skip "white oak", "Green Trail", ... -- the color names the substrate.
            if _SUBSTRATE_FOLLOWER.match(lead, match.end()):
                continue
            found.add(canonical)
            break

    if _RUST_COLOR.search(lead):
        found.add("orange")

    return frozenset(found)


def single_color_label(caption: str) -> str | None:
    """Return the color class when a caption states exactly one, else ``None``.

    Captions naming two or more colors ("silver-blue", "red and white") are
    ambiguous as single-label ground truth and are dropped from the eval set.
    """
    labels = caption_color_labels(caption)
    if len(labels) != 1:
        return None
    return next(iter(labels))


def label_counts(captions: Iterable[str]) -> dict[str, int]:
    """Count single-label captions per color class. Convenience for the eval script."""
    counts: dict[str, int] = {}
    for caption in captions:
        label = single_color_label(caption)
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
