"""Experiment metadata for tech tree tracking.

The ``[meta]`` TOML section stores experiment lineage, status, and tags.
It does **not** affect pipeline execution — it exists purely for
discoverability and the experiment tree viewer (``kromcanon.tree``).

Schema
------
::

    [meta]
    id      = "full"                     # unique id (defaults to filename stem)
    title   = "Production baseline"      # human-readable title
    status  = "baseline"                 # wip | promising | dead_end | baseline | …
    parents = ["quick"]                  # parent experiment ids (lineage edges)
    tags    = ["all-arch", "seed-42"]    # free-form tags for filtering
    date    = 2026-03-10                 # ISO date (TOML native or string)
    notes   = "Base for all comparisons."
    justification = "Why: base config for all Phase 1 comparisons."
    comments = [
        "2026-03-14: Re-run with Canon-ABCD init fix",
        "2026-03-14: Eval loss ordering K < C < V",
    ]

Every field except ``id`` is optional.  When ``id`` is omitted the TOML
filename stem is used.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

VALID_STATUSES: frozenset[str] = frozenset({
    "wip",
    "running",
    "promising",
    "dead_end",
    "baseline",
    "superseded",
    "archived",
    "parked",
})


@dataclass(frozen=True, slots=True)
class MetaConfig:
    """Experiment metadata — does not affect pipeline execution.

    Attributes:
        id: Unique experiment identifier.
        title: Human-readable experiment title.
        status: Experiment status (see ``VALID_STATUSES``).
        parents: Parent experiment ids for lineage tracking.
        tags: Free-form tags for filtering.
        notes: Multi-line notes (markdown-compatible).
        date: ISO 8601 date string.
        justification: Why this experiment exists — what question it answers.
        comments: Timestamped observations, one string per comment.
    """

    id: str
    title: str = ""
    status: str = "wip"
    parents: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    date: str = ""
    justification: str = ""
    comments: list[str] = field(default_factory=list)


def parse_meta(
    raw: dict[str, object],
    *,
    fallback_id: str = "",
) -> MetaConfig | None:
    """Parse a ``[meta]`` section from a raw TOML dict.

    Args:
        raw: The full parsed TOML dict (top-level keys).
        fallback_id: Used as ``id`` when the section exists but ``id``
            is omitted.  Typically the TOML filename stem.

    Returns:
        ``MetaConfig`` if a ``[meta]`` section is present, else ``None``.

    Raises:
        ValueError: On invalid field values (bad status, empty strings in
            lists, etc.).
        TypeError: On wrong field types.
    """
    section = raw.get("meta")
    if section is None:
        return None
    if not isinstance(section, dict):
        msg = f"[meta] must be a table, got {type(section).__name__}"
        raise TypeError(msg)

    # --- id ---
    meta_id = section.get("id", fallback_id)
    if not isinstance(meta_id, str):
        msg = f"meta.id must be a string, got {type(meta_id).__name__}"
        raise TypeError(msg)
    if not meta_id:
        msg = "meta.id must be non-empty (provide id or a fallback_id)"
        raise ValueError(msg)

    # --- title ---
    title = section.get("title", "")
    if not isinstance(title, str):
        msg = f"meta.title must be a string, got {type(title).__name__}"
        raise TypeError(msg)

    # --- status ---
    status = section.get("status", "wip")
    if not isinstance(status, str):
        msg = f"meta.status must be a string, got {type(status).__name__}"
        raise TypeError(msg)
    if status not in VALID_STATUSES:
        msg = f"meta.status must be one of {sorted(VALID_STATUSES)}, got {status!r}"
        raise ValueError(msg)

    # --- parents ---
    parents = section.get("parents", [])
    if not isinstance(parents, list):
        msg = f"meta.parents must be a list, got {type(parents).__name__}"
        raise TypeError(msg)
    for i, p in enumerate(parents):
        if not isinstance(p, str) or not p.strip():
            msg = f"meta.parents[{i}] must be a non-empty string, got {p!r}"
            raise ValueError(msg)

    # --- tags ---
    tags = section.get("tags", [])
    if not isinstance(tags, list):
        msg = f"meta.tags must be a list, got {type(tags).__name__}"
        raise TypeError(msg)
    for i, t in enumerate(tags):
        if not isinstance(t, str) or not t.strip():
            msg = f"meta.tags[{i}] must be a non-empty string, got {t!r}"
            raise ValueError(msg)

    # --- notes ---
    notes = section.get("notes", "")
    if not isinstance(notes, str):
        msg = f"meta.notes must be a string, got {type(notes).__name__}"
        raise TypeError(msg)

    # --- date ---
    raw_date = section.get("date", "")
    if isinstance(raw_date, datetime.date):
        date_str = raw_date.isoformat()
    elif isinstance(raw_date, str):
        date_str = raw_date
    else:
        msg = f"meta.date must be a string or date, got {type(raw_date).__name__}"
        raise TypeError(msg)

    # --- justification ---
    justification = section.get("justification", "")
    if not isinstance(justification, str):
        msg = (
            "meta.justification must be a string, "
            f"got {type(justification).__name__}"
        )
        raise TypeError(msg)

    # --- comments ---
    comments = section.get("comments", [])
    if not isinstance(comments, list):
        msg = f"meta.comments must be a list, got {type(comments).__name__}"
        raise TypeError(msg)
    for i, c in enumerate(comments):
        if not isinstance(c, str):
            msg = (
                f"meta.comments[{i}] must be a string, got {c!r}"
            )
            raise ValueError(msg)

    return MetaConfig(
        id=meta_id,
        title=title,
        status=status,
        parents=list(parents),
        tags=list(tags),
        notes=notes.strip(),
        date=date_str,
        justification=justification.strip(),
        comments=list(comments),
    )
