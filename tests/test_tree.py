"""Tests for kromcanon.tree — experiment lineage tree viewer."""

from __future__ import annotations

from pathlib import Path

from kromcanon.tree import (
    ExperimentNode,
    build_graph,
    discover_experiments,
    filter_nodes,
    render_mermaid,
    render_text_tree,
)


def _node(
    id: str,
    *,
    title: str = "",
    status: str = "wip",
    parents: list[str] | None = None,
    tags: list[str] | None = None,
) -> ExperimentNode:
    """Helper to create test nodes."""
    return ExperimentNode(
        id=id,
        title=title or id,
        status=status,
        parents=parents or [],
        tags=tags or [],
        path=Path(f"experiments/{id}.toml"),
    )


class TestBuildGraph:
    """Graph construction and validation."""

    def test_linear_chain(self) -> None:
        nodes = [
            _node("quick", status="baseline"),
            _node("full", parents=["quick"], status="baseline"),
            _node("seed2", parents=["full"]),
        ]
        g = build_graph(nodes)
        assert g.roots == ["quick"]
        assert "full" in g.children["quick"]
        assert "seed2" in g.children["full"]
        assert g.warnings == []

    def test_dangling_parent_warning(self) -> None:
        nodes = [_node("child", parents=["missing"])]
        g = build_graph(nodes)
        assert "child" in g.roots  # becomes root since parent is missing
        assert any("missing" in w for w in g.warnings)

    def test_duplicate_id_warning(self) -> None:
        nodes = [
            _node("dup"),
            ExperimentNode(
                id="dup",
                title="duplicate",
                status="wip",
                parents=[],
                tags=[],
                path=Path("experiments/dup2.toml"),
            ),
        ]
        g = build_graph(nodes)
        assert any("Duplicate" in w for w in g.warnings)

    def test_cycle_warning(self) -> None:
        # a → b → c → a
        nodes = [
            _node("a", parents=["c"]),
            _node("b", parents=["a"]),
            _node("c", parents=["b"]),
        ]
        g = build_graph(nodes)
        assert any("Cycle" in w or "missing" in w for w in g.warnings)

    def test_multi_parent_dag(self) -> None:
        nodes = [
            _node("root"),
            _node("a", parents=["root"]),
            _node("b", parents=["root"]),
            _node("merge", parents=["a", "b"]),
        ]
        g = build_graph(nodes)
        assert "root" in g.roots
        assert "merge" in g.children["a"]
        assert "merge" in g.children["b"]

    def test_empty_graph(self) -> None:
        g = build_graph([])
        assert g.nodes == {}
        assert g.roots == []


class TestRenderTextTree:
    """Text tree output."""

    def test_simple_tree(self) -> None:
        nodes = [
            _node("quick", status="baseline", title="Pipeline validation"),
            _node("full", parents=["quick"], status="baseline", title="Production baseline"),
            _node("seed2", parents=["full"], title="Seed 2 replication"),
        ]
        g = build_graph(nodes)
        output = render_text_tree(g)
        assert "[BASE] Pipeline validation" in output
        assert "├── " in output or "└── " in output
        assert "[BASE] Production baseline" in output
        assert "Seed 2 replication" in output

    def test_tag_summary(self) -> None:
        nodes = [
            _node("a", tags=["gcg", "infix"]),
            _node("b", tags=["gcg"]),
        ]
        g = build_graph(nodes)
        output = render_text_tree(g)
        assert "Tags:" in output
        assert "gcg(2)" in output
        assert "infix(1)" in output

    def test_warnings_displayed(self) -> None:
        nodes = [_node("child", parents=["missing"])]
        g = build_graph(nodes)
        output = render_text_tree(g)
        assert "Warnings:" in output
        assert "missing" in output


class TestRenderMermaid:
    """Mermaid flowchart output."""

    def test_mermaid_header(self) -> None:
        nodes = [_node("root"), _node("child", parents=["root"])]
        g = build_graph(nodes)
        output = render_mermaid(g)
        assert output.startswith("flowchart TD")

    def test_mermaid_edges(self) -> None:
        nodes = [_node("root"), _node("child", parents=["root"])]
        g = build_graph(nodes)
        output = render_mermaid(g)
        assert "root --> child" in output

    def test_mermaid_id_sanitization(self) -> None:
        nodes = [_node("bias-sweep.v2", title="Bias sweep")]
        g = build_graph(nodes)
        output = render_mermaid(g)
        assert "bias_sweep_v2" in output
        # No raw hyphens/dots in node ids
        assert 'bias-sweep.v2[' not in output


class TestFilterNodes:
    """Filtering by status and tag."""

    def test_filter_by_status(self) -> None:
        nodes = [
            _node("a", status="wip"),
            _node("b", status="baseline"),
            _node("c", status="wip"),
        ]
        result = filter_nodes(nodes, status="wip")
        assert len(result) == 2
        assert all(n.status == "wip" for n in result)

    def test_filter_by_tag(self) -> None:
        nodes = [
            _node("a", tags=["gcg", "infix"]),
            _node("b", tags=["egd"]),
        ]
        result = filter_nodes(nodes, tag="infix")
        assert len(result) == 1
        assert result[0].id == "a"

    def test_filter_combined(self) -> None:
        nodes = [
            _node("a", status="wip", tags=["gcg"]),
            _node("b", status="baseline", tags=["gcg"]),
            _node("c", status="wip", tags=["egd"]),
        ]
        result = filter_nodes(nodes, status="wip", tag="gcg")
        assert len(result) == 1
        assert result[0].id == "a"


class TestDiscoverExperiments:
    """Discovery from filesystem."""

    def test_discover_from_directory(self, tmp_path: Path) -> None:
        toml = tmp_path / "test.toml"
        toml.write_text("""
[meta]
title = "Test experiment"
status = "wip"
tags = ["test"]

[experiment]
run_name = "test"
""")
        nodes = discover_experiments(tmp_path)
        assert len(nodes) == 1
        assert nodes[0].id == "test"
        assert nodes[0].title == "Test experiment"

    def test_discover_without_meta(self, tmp_path: Path) -> None:
        toml = tmp_path / "bare.toml"
        toml.write_text('[experiment]\nrun_name = "bare"\n')
        nodes = discover_experiments(tmp_path)
        assert len(nodes) == 1
        assert nodes[0].id == "bare"
        assert nodes[0].status == "wip"

    def test_bad_toml_skipped(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.toml"
        bad.write_text("this is not valid toml {{{")
        nodes = discover_experiments(tmp_path)
        assert len(nodes) == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        nodes = discover_experiments(tmp_path)
        assert len(nodes) == 0
