"""Tests for visualization module."""

import numpy as np
import pytest

from kromcanon.interp.viz import (
    ARCH_COLORS,
    fig1_training_curves,
    fig2_direction_norms,
    fig3_cosine_heatmap,
    fig4_stream_distribution,
    fig5_alpha_sweep,
    fig6_abliteration_bars,
    fig7_stream_cosines,
    fig8_method_comparison,
    generate_all_figures,
)

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")


@pytest.fixture
def output_dir(tmp_path: object) -> object:
    """Temporary output directory for figures."""
    return tmp_path / "figures"


class TestFigureRendering:
    """Verify each figure renders without error and returns a Figure."""

    def test_fig1_training_curves(self, output_dir: object) -> None:
        """Training curves render with mock data."""
        logs = {
            "vanilla": [{"step": i, "loss": 5.0 - i * 0.1} for i in range(20)],
            "canon": [{"step": i, "loss": 5.1 - i * 0.09} for i in range(20)],
            "kromcanon": [{"step": i, "loss": 5.2 - i * 0.08} for i in range(20)],
        }
        fig = fig1_training_curves(logs, output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert (output_dir / "fig1_training_curves.pdf").exists()

    def test_fig2_direction_norms(self, output_dir: object) -> None:
        """Direction norms render with mock data."""
        norms = {
            "vanilla": np.random.rand(4),
            "canon": np.random.rand(4),
            "kromcanon": np.random.rand(4),
        }
        fig = fig2_direction_norms(norms, output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert (output_dir / "fig2_direction_norms.pdf").exists()

    def test_fig3_cosine_heatmap(self, output_dir: object) -> None:
        """Cosine heatmap renders with mock data."""
        matrix = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
        fig = fig3_cosine_heatmap(matrix, ["vanilla", "canon", "kromcanon"], output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert (output_dir / "fig3_cosine_heatmap.pdf").exists()

    def test_fig3_cosine_heatmap_3d(self, output_dir: object) -> None:
        """Cosine heatmap with per-layer data (3D) is averaged."""
        matrix = np.random.rand(3, 3, 4)
        fig = fig3_cosine_heatmap(matrix, ["vanilla", "canon", "kromcanon"], output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_fig4_stream_distribution(self, output_dir: object) -> None:
        """Stream distribution renders with mock data."""
        norm_ratios = np.random.dirichlet([1, 1, 1, 1], size=4)
        concentration = np.random.rand(4)
        fig = fig4_stream_distribution(norm_ratios, concentration, output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert (output_dir / "fig4_stream_distribution.pdf").exists()

    def test_fig5_alpha_sweep(self, output_dir: object) -> None:
        """Alpha sweep renders with mock data."""
        alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]
        sweep_data = {
            "vanilla": (alphas, np.array([0.5, 0.2, 0.0, 0.3, 0.6])),
            "canon": (alphas, np.array([0.4, 0.15, 0.0, 0.25, 0.5])),
            "kromcanon": (alphas, np.array([0.3, 0.1, 0.0, 0.15, 0.35])),
        }
        fig = fig5_alpha_sweep(sweep_data, output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert (output_dir / "fig5_alpha_sweep.pdf").exists()

    def test_fig6_abliteration_bars(self, output_dir: object) -> None:
        """Abliteration bars render with mock data."""
        rates = {
            "vanilla": {"before": 0.8, "after": 0.2},
            "canon": {"before": 0.75, "after": 0.25},
            "kromcanon": {"before": 0.7, "after": 0.3},
        }
        fig = fig6_abliteration_bars(rates, output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert (output_dir / "fig6_abliteration_bars.pdf").exists()

    def test_fig7_stream_cosines(self, output_dir: object) -> None:
        """Stream cosine matrix renders with mock data."""
        matrix = np.eye(4) + 0.1 * np.random.rand(4, 4)
        matrix = (matrix + matrix.T) / 2  # symmetrize
        fig = fig7_stream_cosines(matrix, output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert (output_dir / "fig7_stream_cosines.pdf").exists()

    def test_fig8_method_comparison(self, output_dir: object) -> None:
        """Method comparison renders with mock data."""
        mean_norms = {
            "vanilla": np.random.rand(4),
            "canon": np.random.rand(4),
        }
        svd_norms = {
            "vanilla": np.random.rand(4),
            "canon": np.random.rand(4),
        }
        fig = fig8_method_comparison(mean_norms, svd_norms, output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert (output_dir / "fig8_method_comparison.pdf").exists()

    def test_fig8_empty_data(self, output_dir: object) -> None:
        """Method comparison handles empty data gracefully."""
        fig = fig8_method_comparison({}, {}, output_dir)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestGenerateAll:
    """Test the generate_all_figures convenience function."""

    def test_generates_all_pdfs(self, output_dir: object) -> None:
        """All requested figures are generated as PDFs."""
        paths = generate_all_figures(
            training_logs={
                "vanilla": [{"step": i, "loss": 5.0 - i * 0.1} for i in range(10)],
            },
            direction_norms={"vanilla": np.random.rand(4)},
            cosine_matrix=np.eye(1),
            arch_names=["vanilla"],
            stream_norm_ratios=np.random.dirichlet([1, 1, 1, 1], size=4),
            stream_concentration=np.random.rand(4),
            alpha_sweep_data={
                "vanilla": ([-1.0, 0.0, 1.0], np.array([0.3, 0.0, 0.3])),
            },
            refusal_rates={"vanilla": {"before": 0.8, "after": 0.2}},
            stream_cosine_matrix=np.eye(4),
            mean_diff_norms={"vanilla": np.random.rand(4)},
            svd_norms={"vanilla": np.random.rand(4)},
            output_dir=output_dir,
        )
        assert len(paths) == 8
        for p in paths:
            assert p.exists(), f"Missing: {p}"

    def test_skips_optional_figures(self, output_dir: object) -> None:
        """Figures requiring optional data are skipped when data is None."""
        paths = generate_all_figures(
            training_logs={"vanilla": [{"step": 1, "loss": 3.0}]},
            direction_norms={"vanilla": np.array([1.0])},
            cosine_matrix=np.eye(1),
            arch_names=["vanilla"],
            stream_norm_ratios=None,
            stream_concentration=None,
            alpha_sweep_data={},
            refusal_rates={"vanilla": {"before": 0.5, "after": 0.1}},
            stream_cosine_matrix=None,
            mean_diff_norms={"vanilla": np.array([1.0])},
            svd_norms={"vanilla": np.array([0.9])},
            output_dir=output_dir,
        )
        # Should have 6 figures (no fig4 stream dist, no fig7 stream cosines)
        assert len(paths) == 6


class TestColorPalette:
    """Verify the architecture color mapping."""

    def test_all_archs_have_colors(self) -> None:
        """All three architectures are in the color palette."""
        for arch in ["vanilla", "canon", "kromcanon"]:
            assert arch in ARCH_COLORS
