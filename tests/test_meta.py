"""Tests for kromcanon.meta — [meta] TOML section parsing."""

from __future__ import annotations

import datetime

import pytest

from kromcanon.meta import VALID_STATUSES, MetaConfig, parse_meta


class TestParseMetaBasic:
    """Basic parsing and defaults."""

    def test_missing_section_returns_none(self) -> None:
        assert parse_meta({}) is None
        assert parse_meta({"experiment": {"run_name": "foo"}}) is None

    def test_empty_section_uses_fallback_id(self) -> None:
        meta = parse_meta({"meta": {}}, fallback_id="quick")
        assert meta is not None
        assert meta.id == "quick"
        assert meta.title == ""
        assert meta.status == "wip"
        assert meta.parents == []
        assert meta.tags == []

    def test_explicit_id_overrides_fallback(self) -> None:
        meta = parse_meta({"meta": {"id": "custom"}}, fallback_id="quick")
        assert meta is not None
        assert meta.id == "custom"

    def test_full_section(self) -> None:
        raw = {
            "meta": {
                "id": "full",
                "title": "Production baseline",
                "status": "baseline",
                "parents": ["quick"],
                "tags": ["all-arch", "seed-42"],
                "notes": "Base for all comparisons.",
                "date": "2026-03-10",
            },
        }
        meta = parse_meta(raw)
        assert meta == MetaConfig(
            id="full",
            title="Production baseline",
            status="baseline",
            parents=["quick"],
            tags=["all-arch", "seed-42"],
            notes="Base for all comparisons.",
            date="2026-03-10",
        )


class TestParseMetaDate:
    """Date field handling."""

    def test_string_date(self) -> None:
        meta = parse_meta({"meta": {"date": "2026-03-10"}}, fallback_id="x")
        assert meta is not None
        assert meta.date == "2026-03-10"

    def test_native_toml_date(self) -> None:
        meta = parse_meta(
            {"meta": {"date": datetime.date(2026, 3, 10)}},
            fallback_id="x",
        )
        assert meta is not None
        assert meta.date == "2026-03-10"


class TestParseMetaValidation:
    """Validation errors."""

    def test_bad_status_raises(self) -> None:
        with pytest.raises(ValueError, match="meta.status"):
            parse_meta({"meta": {"status": "invalid"}}, fallback_id="x")

    def test_all_valid_statuses_accepted(self) -> None:
        for status in VALID_STATUSES:
            meta = parse_meta(
                {"meta": {"status": status}}, fallback_id="x"
            )
            assert meta is not None
            assert meta.status == status

    def test_empty_id_no_fallback_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            parse_meta({"meta": {}})

    def test_non_string_id_raises(self) -> None:
        with pytest.raises(TypeError, match="meta.id"):
            parse_meta({"meta": {"id": 42}}, fallback_id="x")

    def test_non_list_parents_raises(self) -> None:
        with pytest.raises(TypeError, match="meta.parents"):
            parse_meta({"meta": {"parents": "quick"}}, fallback_id="x")

    def test_empty_string_in_parents_raises(self) -> None:
        with pytest.raises(ValueError, match="meta.parents"):
            parse_meta({"meta": {"parents": [""]}}, fallback_id="x")

    def test_non_list_tags_raises(self) -> None:
        with pytest.raises(TypeError, match="meta.tags"):
            parse_meta({"meta": {"tags": "foo"}}, fallback_id="x")

    def test_non_table_section_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            parse_meta({"meta": "not a table"}, fallback_id="x")

    def test_bad_date_type_raises(self) -> None:
        with pytest.raises(TypeError, match="meta.date"):
            parse_meta({"meta": {"date": 42}}, fallback_id="x")


class TestMetaConfigFrozen:
    """MetaConfig is immutable."""

    def test_frozen(self) -> None:
        meta = MetaConfig(id="test")
        with pytest.raises(AttributeError):
            meta.id = "changed"  # type: ignore[misc]
