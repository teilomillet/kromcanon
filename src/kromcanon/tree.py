"""Experiment lineage tree viewer.

Discovers all TOML experiment configs in a directory, parses their
``[meta]`` sections, and renders the experiment DAG as an ASCII tree
or Mermaid flowchart.

Usage::

    python -m kromcanon.tree                              # default: experiments/
    python -m kromcanon.tree experiments/
    python -m kromcanon.tree experiments/ --format mermaid
    python -m kromcanon.tree experiments/ --status promising
    python -m kromcanon.tree experiments/ --tag bias-sweep

The tree is built from ``parents`` edges declared in each config's
``[meta]`` section.  Experiments with no parents (or whose parents
are all missing) become roots.  Status badges show progress at a glance::

    [BASE] Production baseline (seed=42)
    ├── [WIP] Gating: KromHC bias_res_init=-2
    │   ├── [WIP] Bias sweep: bias_res_init=-1
    │   └── [WIP] Bias sweep: bias_res_init=0
    └── [WIP] Ablation: minimal SFT budget
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from kromcanon.meta import MetaConfig, parse_meta

# ─────────────────────────────────────────────────────────────────────
# Status badges
# ─────────────────────────────────────────────────────────────────────

_BADGES: dict[str, str] = {
    "wip": "[WIP]",
    "running": "[RUN]",
    "promising": "[OK+]",
    "dead_end": "[X]",
    "baseline": "[BASE]",
    "superseded": "[OLD]",
    "archived": "[ARC]",
}

_MERMAID_STYLES: dict[str, str] = {
    "wip": "fill:#fff3cd,stroke:#ffc107",
    "running": "fill:#d1ecf1,stroke:#17a2b8",
    "promising": "fill:#d4edda,stroke:#28a745",
    "dead_end": "fill:#f8d7da,stroke:#dc3545",
    "baseline": "fill:#cce5ff,stroke:#007bff",
    "superseded": "fill:#e2e3e5,stroke:#6c757d",
    "archived": "fill:#e2e3e5,stroke:#6c757d",
}


# ─────────────────────────────────────────────────────────────────────
# Experiment node
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExperimentNode:
    """A single experiment in the lineage graph.

    Attributes:
        id: Unique experiment identifier.
        title: Human-readable title.
        status: One of the ``VALID_STATUSES``.
        parents: Parent experiment ids.
        tags: Free-form tags.
        path: Filesystem path to the TOML config.
        date: ISO date string (from TOML ``[meta].date``).
        started_at: ISO timestamp from results (auto-set by runner).
        completed_at: ISO timestamp from results (auto-set by runner).
        outcome: Last run outcome (``"success"``, ``"failed"``, or ``""``).
        run_count: Total number of runs (current + archived).
    """

    id: str
    title: str
    status: str
    parents: list[str]
    tags: list[str]
    path: Path
    date: str = ""
    started_at: str = ""
    completed_at: str = ""
    outcome: str = ""
    run_count: int = 0

    @property
    def badge(self) -> str:
        """Status badge for display."""
        return _BADGES.get(self.status, f"[{self.status.upper()}]")

    @property
    def timestamp(self) -> str:
        """Best available date+time for display (started > completed > meta date).

        Uses ``started_at`` for chronological ordering (when the experiment
        began), falls back to ``completed_at`` then manual ``date``.
        Shows date+time (``YYYY-MM-DD HH:MM``) for run timestamps,
        date-only for manual ``[meta].date``.
        """
        for ts in (self.started_at, self.completed_at):
            if ts:
                # ISO format: 2026-03-14T08:13:15+00:00 → 2026-03-14 08:13
                return ts[:16].replace("T", " ")
        if self.date:
            return self.date[:10]
        return ""

    @property
    def display_label(self) -> str:
        """Badge + timestamp + title (or id if no title).

        Appends retry count if the experiment has been run more than once,
        and a ``(FAILED)`` marker if the last run failed.
        """
        label = self.title if self.title else self.id
        ts = self.timestamp
        parts: list[str] = [self.badge]
        if ts:
            parts.append(ts)
        parts.append(label)
        suffix = ""
        if self.outcome == "failed":
            suffix += " (FAILED)"
        if self.run_count > 1:
            suffix += f" (run {self.run_count})"
        return " ".join(parts) + suffix


# ─────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────


def discover_experiments(directory: Path) -> list[ExperimentNode]:
    """Find all TOML files with ``[meta]`` sections in *directory*.

    Files that fail to parse are silently skipped.  Files without a
    ``[meta]`` section get a default node (id = filename stem, status = wip).

    Args:
        directory: Root directory to search (non-recursive: only top-level).

    Returns:
        List of ``ExperimentNode`` instances.
    """
    nodes: list[ExperimentNode] = []
    for path in sorted(directory.glob("*.toml")):
        try:
            raw = tomllib.loads(path.read_text())
        except Exception:  # noqa: BLE001 — skip unparseable TOML
            continue

        meta = parse_meta(raw, fallback_id=path.stem)
        if meta is None:
            # No [meta] section — create a minimal stub
            meta = MetaConfig(id=path.stem)

        nodes.append(ExperimentNode(
            id=meta.id,
            title=meta.title,
            status=meta.status,
            parents=meta.parents,
            tags=meta.tags,
            path=path,
            date=meta.date,
        ))
    return nodes


@dataclass(frozen=True, slots=True)
class _ResultInfo:
    """Run metadata extracted from a results ``config.json``."""

    started_at: str
    completed_at: str
    outcome: str
    run_count: int


def _read_result_info(config_path: Path) -> _ResultInfo | None:
    """Read run info from a single results config.json.

    Args:
        config_path: Path to ``results/<run>/config.json``.

    Returns:
        ``_ResultInfo`` or ``None`` on error.
    """
    try:
        data = json.loads(config_path.read_text())
    except Exception:  # noqa: BLE001
        return None
    prev_runs: list[object] = data.get("runs", [])
    return _ResultInfo(
        started_at=data.get("started_at", ""),
        completed_at=data.get("completed_at", ""),
        outcome=data.get("outcome", ""),
        run_count=len(prev_runs) + (1 if data.get("started_at") else 0),
    )


def enrich_from_results(
    nodes: list[ExperimentNode],
    results_dir: Path,
) -> list[ExperimentNode]:
    """Enrich nodes with run timestamps from results ``config.json`` files.

    For each node, looks for ``results_dir/<run_name>/config.json`` and
    reads ``started_at``, ``completed_at``, ``outcome``, and ``runs``
    history written by the experiment runner.

    Args:
        nodes: Discovered experiment nodes.
        results_dir: Path to the results directory (e.g. ``results/``).

    Returns:
        New list of nodes with timestamps and run info filled in.
    """
    if not results_dir.is_dir():
        return nodes

    # Index config.json by run_name and meta.id
    index: dict[str, _ResultInfo] = {}
    for config_path in results_dir.glob("*/config.json"):
        info = _read_result_info(config_path)
        if info is None:
            continue
        try:
            data = json.loads(config_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        run_name = data.get("run_name", "")
        if run_name:
            index[run_name] = info
        meta_id = data.get("meta", {}).get("id", "")
        if meta_id and meta_id not in index:
            index[meta_id] = info

    enriched: list[ExperimentNode] = []
    for node in nodes:
        # Try matching by id first, then by TOML run_name
        info = index.get(node.id)
        if info is None:
            try:
                raw = tomllib.loads(node.path.read_text())
                run_name = raw.get("experiment", {}).get("run_name", "")
                if run_name:
                    info = index.get(run_name)
            except Exception:  # noqa: BLE001
                pass

        if info is not None:
            enriched.append(ExperimentNode(
                id=node.id,
                title=node.title,
                status=node.status,
                parents=node.parents,
                tags=node.tags,
                path=node.path,
                date=node.date,
                started_at=info.started_at,
                completed_at=info.completed_at,
                outcome=info.outcome,
                run_count=info.run_count,
            ))
        else:
            enriched.append(node)

    return enriched


# ─────────────────────────────────────────────────────────────────────
# Graph building
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ExperimentGraph:
    """DAG of experiments built from parent edges.

    Attributes:
        nodes: All experiment nodes, keyed by id.
        children: Adjacency list (parent → list of children).
        roots: Nodes with no parents (or all parents missing).
        warnings: Validation warnings (duplicates, dangling refs, cycles).
    """

    nodes: dict[str, ExperimentNode] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    roots: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_graph(nodes: list[ExperimentNode]) -> ExperimentGraph:
    """Build a validated experiment DAG.

    Detects duplicate ids, dangling parent references, and cycles.
    All issues are recorded as warnings — the graph is always usable.

    Args:
        nodes: Discovered experiment nodes.

    Returns:
        Validated ``ExperimentGraph``.
    """
    g = ExperimentGraph()

    # Index by id, detect duplicates
    for node in nodes:
        if node.id in g.nodes:
            g.warnings.append(
                f"Duplicate id {node.id!r}: {g.nodes[node.id].path} and {node.path}"
            )
            continue
        g.nodes[node.id] = node

    # Build children map, detect dangling parents
    for node in g.nodes.values():
        has_valid_parent = False
        for parent_id in node.parents:
            if parent_id not in g.nodes:
                g.warnings.append(
                    f"{node.id!r} references missing parent {parent_id!r}"
                )
            else:
                g.children[parent_id].append(node.id)
                has_valid_parent = True
        if not has_valid_parent:
            g.roots.append(node.id)

    # Sort children and roots for deterministic output
    for parent_id in g.children:
        g.children[parent_id].sort(key=lambda cid: g.nodes[cid].title or cid)
    g.roots.sort(key=lambda rid: g.nodes[rid].title or rid)

    # Cycle detection (DFS)
    visited: set[str] = set()
    in_stack: set[str] = set()

    def _dfs(node_id: str) -> None:
        if node_id in in_stack:
            g.warnings.append(f"Cycle detected involving {node_id!r}")
            return
        if node_id in visited:
            return
        visited.add(node_id)
        in_stack.add(node_id)
        for child_id in g.children.get(node_id, []):
            _dfs(child_id)
        in_stack.discard(node_id)

    for root_id in g.roots:
        _dfs(root_id)

    # Nodes not reachable from any root are part of a cycle
    for node_id in g.nodes:
        if node_id not in visited:
            g.warnings.append(f"Cycle detected involving {node_id!r}")

    return g


# ─────────────────────────────────────────────────────────────────────
# Text tree rendering
# ─────────────────────────────────────────────────────────────────────


def render_text_tree(graph: ExperimentGraph) -> str:
    """Render the experiment DAG as an ASCII tree.

    Args:
        graph: Validated experiment graph.

    Returns:
        Multi-line string with tree structure, tag summary, and warnings.
    """
    lines: list[str] = []
    rendered: set[str] = set()

    def _render(node_id: str, prefix: str, is_last: bool) -> None:
        if node_id in rendered:
            return
        rendered.add(node_id)

        node = graph.nodes[node_id]
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{node.display_label}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        children = graph.children.get(node_id, [])
        for i, child_id in enumerate(children):
            _render(child_id, child_prefix, is_last=(i == len(children) - 1))

    # Render each root as a top-level tree
    for i, root_id in enumerate(graph.roots):
        if root_id in rendered:
            continue
        node = graph.nodes[root_id]
        # Roots without children — render inline
        children = graph.children.get(root_id, [])
        lines.append(node.display_label)
        for j, child_id in enumerate(children):
            _render(child_id, "", is_last=(j == len(children) - 1))
        if i < len(graph.roots) - 1 and children:
            lines.append("")

    # Tag summary
    tag_counts: dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        for tag in node.tags:
            tag_counts[tag] += 1
    if tag_counts:
        tag_parts = [f"{t}({c})" for t, c in sorted(tag_counts.items())]
        lines.append("")
        lines.append(f"Tags: {', '.join(tag_parts)}")

    # Warnings
    if graph.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in graph.warnings:
            lines.append(f"  ! {w}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Mermaid rendering
# ─────────────────────────────────────────────────────────────────────


def _sanitize_mermaid_id(node_id: str) -> str:
    """Replace non-alphanumeric characters with underscores.

    Args:
        node_id: Raw experiment id.

    Returns:
        Mermaid-safe identifier.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", node_id)


def render_mermaid(graph: ExperimentGraph) -> str:
    """Render the experiment DAG as a Mermaid flowchart.

    Args:
        graph: Validated experiment graph.

    Returns:
        Mermaid flowchart string (paste into any Mermaid renderer).
    """
    lines: list[str] = ["flowchart TD"]

    # Node definitions
    for node in graph.nodes.values():
        safe_id = _sanitize_mermaid_id(node.id)
        label = node.title if node.title else node.id
        # Escape quotes in label
        label = label.replace('"', "'")
        lines.append(f'    {safe_id}["{node.badge} {label}"]')

    lines.append("")

    # Edges (parent → child)
    for parent_id, child_ids in sorted(graph.children.items()):
        safe_parent = _sanitize_mermaid_id(parent_id)
        for child_id in child_ids:
            safe_child = _sanitize_mermaid_id(child_id)
            lines.append(f"    {safe_parent} --> {safe_child}")

    # Status-based styling
    lines.append("")
    status_groups: dict[str, list[str]] = defaultdict(list)
    for node in graph.nodes.values():
        status_groups[node.status].append(_sanitize_mermaid_id(node.id))
    for status, node_ids in sorted(status_groups.items()):
        style = _MERMAID_STYLES.get(status, "")
        if style:
            for nid in node_ids:
                lines.append(f"    style {nid} {style}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────


def filter_nodes(
    nodes: list[ExperimentNode],
    *,
    status: str | None = None,
    tag: str | None = None,
) -> list[ExperimentNode]:
    """Filter experiment nodes by status and/or tag.

    Args:
        nodes: All discovered nodes.
        status: Keep only nodes with this status.
        tag: Keep only nodes with this tag.

    Returns:
        Filtered list.
    """
    result = nodes
    if status is not None:
        result = [n for n in result if n.status == status]
    if tag is not None:
        result = [n for n in result if tag in n.tags]
    return result


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the experiment tree viewer.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        prog="python -m kromcanon.tree",
        description="Render the experiment lineage tree from [meta] sections.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="experiments",
        help="Directory containing experiment TOML files (default: experiments/)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "mermaid"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="Filter by status (wip, promising, dead_end, baseline, superseded, archived)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Filter by tag",
    )
    parser.add_argument(
        "--results",
        default="results",
        help="Results directory for run timestamps (default: results/)",
    )
    args = parser.parse_args(argv)

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    nodes = discover_experiments(directory)
    if not nodes:
        print(f"No TOML files found in {directory}", file=sys.stderr)
        sys.exit(1)

    # Enrich with run timestamps from results directory
    results_dir = Path(args.results)
    nodes = enrich_from_results(nodes, results_dir)

    nodes = filter_nodes(nodes, status=args.status, tag=args.tag)
    if not nodes:
        print("No experiments match the given filters.", file=sys.stderr)
        sys.exit(1)

    graph = build_graph(nodes)

    if args.format == "mermaid":
        print(render_mermaid(graph))
    else:
        print(render_text_tree(graph))


if __name__ == "__main__":
    main()
