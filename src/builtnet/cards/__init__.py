"""Construction cards: machine-readable descriptions of register-network construction rules.

A card is a TOML file with three tables:

  [card]        name, the network it describes, its sources, what the preset leaves out, and optionally
                `base`, another card it modifies;
  [rules]       fields of `builtnet.rulesets.RuleSet`, one to one; caps may be a number, "se_k" (the Swedish
                link function as published), "se_k_code" (with the constants of the builders' released scripts)
                or "none";
  [provenance]  for each field, its status (documented, inferred or assumed) and source.

    from builtnet import cards
    rs = cards.load("nl-2018")                 # a RuleSet
    cards.available()                          # names of the bundled cards
    cards.provenance("se")                     # field -> {status, source}, including inherited fields
"""

from __future__ import annotations

import pathlib
import tomllib

from ..rulesets import RuleSet, se_k, se_k_code

HERE = pathlib.Path(__file__).resolve().parent
CAPS = {"se_k": se_k, "se_k_code": se_k_code, "none": None}
TUPLES = {"layers", "kin_types", "kin_parent_gap"}


def available() -> list[str]:
    return sorted(p.stem for p in HERE.glob("*.toml"))


def _read(name_or_path: str | pathlib.Path) -> dict:
    p = pathlib.Path(name_or_path)
    if not p.suffix:
        p = HERE / f"{name_or_path}.toml"
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def _chain(name_or_path) -> list[dict]:
    out, seen = [], set()
    card = _read(name_or_path)
    while True:
        out.append(card)
        base = card.get("card", {}).get("base")
        if not base:
            break
        if base in seen:
            raise ValueError(f"cycle in card bases at {base}")
        seen.add(base)
        card = _read(base)
    return out[::-1]  # base first


def _value(key: str, v):
    if isinstance(v, str) and v in CAPS:
        return CAPS[v]
    if key in TUPLES and isinstance(v, list):
        return tuple(v)
    return v


def rules(name_or_path) -> dict:
    """The merged [rules] table of a card and its bases, with caps resolved."""
    merged: dict = {}
    for card in _chain(name_or_path):
        merged.update(card.get("rules", {}))
    return {k: _value(k, v) for k, v in merged.items()}


def load(name_or_path) -> RuleSet:
    """A RuleSet from a bundled card name (for example "nl-2018") or a path to a TOML card."""
    chain = _chain(name_or_path)
    fields = set(RuleSet.__dataclass_fields__) - {"name"}
    r = rules(name_or_path)
    unknown = set(r) - fields
    if unknown:
        raise ValueError(f"unknown rule fields: {sorted(unknown)}")
    return RuleSet(chain[-1]["card"]["name"], **r)


def provenance(name_or_path) -> dict:
    merged: dict = {}
    for card in _chain(name_or_path):
        merged.update(card.get("provenance", {}))
    return merged


def describe(name_or_path) -> dict:
    """The [card] table of a card, with its bases' sources added."""
    chain = _chain(name_or_path)
    info = dict(chain[-1]["card"])
    info["all_sources"] = sorted({s for c in chain for s in c.get("card", {}).get("sources", [])})
    return info
