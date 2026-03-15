"""Publication-quality visualization for KromCanon experiments.

Generates 8 figures comparing direction extraction, steering, and abliteration
across Vanilla, Canon, and KromCanon architectures.

Requires: matplotlib>=3.9.0, seaborn>=0.13.0
Install via: uv pip install -e ".[viz]"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

# Architecture color palette
ARCH_COLORS: dict[str, str] = {
    "vanilla": "#4285f4",   # blue
    "canon": "#f4a742",     # orange
    "kromcanon": "#34a853",  # green
}

ARCH_ORDER: list[str] = ["vanilla", "canon", "kromcanon"]


def _setup_style() -> None:
    """Configure matplotlib/seaborn for publication-quality figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _get_arch_color(arch: str) -> str:
    """Get color for an architecture, with fallback.

    Args:
        arch: Architecture name.

    Returns:
        Hex color string.
    """
    return ARCH_COLORS.get(arch, "#888888")


def _save_fig(
    fig: matplotlib.figure.Figure, output_dir: Path, name: str
) -> Path:
    """Save figure to PDF.

    Args:
        fig: Matplotlib figure.
        output_dir: Directory to save to.
        name: File name (without extension).

    Returns:
        Path to saved PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.pdf"
    fig.savefig(str(path))
    return path


def fig1_training_curves(
    logs: dict[str, list[dict[str, float]]],
    output_dir: Path,
) -> matplotlib.figure.Figure:
    """Plot training loss curves for all architectures.

    Args:
        logs: Dict mapping arch name → list of log dicts with "step" and "loss".
        output_dir: Directory to save the figure.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for arch in ARCH_ORDER:
        if arch not in logs or not logs[arch]:
            continue
        steps = [entry["step"] for entry in logs[arch]]
        losses = [entry["loss"] for entry in logs[arch]]
        ax.plot(steps, losses, label=arch, color=_get_arch_color(arch), linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8")
    ax.set_yscale("log")
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig1_training_curves")
    plt.close(fig)
    return fig


def fig2_direction_norms(
    norms: dict[str, np.ndarray],
    output_dir: Path,
) -> matplotlib.figure.Figure:
    """Plot per-layer direction magnitude profiles.

    Args:
        norms: Dict mapping arch → array of shape (n_layers,).
        output_dir: Directory to save the figure.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for arch in ARCH_ORDER:
        if arch not in norms:
            continue
        layers = np.arange(len(norms[arch]))
        ax.plot(
            layers, norms[arch],
            label=arch, color=_get_arch_color(arch),
            linewidth=2, marker="o", markersize=5,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Direction Norm")
    ax.set_title("Refusal Direction Magnitude by Layer")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8")
    ax.set_xticks(layers if len(norms) > 0 else [])
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig2_direction_norms")
    plt.close(fig)
    return fig


def fig3_cosine_heatmap(
    cosine_matrix: np.ndarray,
    arch_names: list[str],
    output_dir: Path,
) -> matplotlib.figure.Figure:
    """Plot cross-architecture cosine similarity heatmap.

    Args:
        cosine_matrix: Pairwise cosine sim matrix, shape (n_archs, n_archs, n_layers)
            or averaged to (n_archs, n_archs).
        arch_names: Architecture names for axis labels.
        output_dir: Directory to save the figure.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    # If 3D (per-layer), average over layers
    plot_data = cosine_matrix.mean(axis=-1) if cosine_matrix.ndim == 3 else cosine_matrix

    sns.heatmap(
        plot_data,
        annot=True, fmt=".3f",
        xticklabels=arch_names, yticklabels=arch_names,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        ax=ax, square=True,
        cbar_kws={"label": "Cosine Similarity"},
    )
    ax.set_title("Cross-Architecture Direction Similarity")
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig3_cosine_heatmap")
    plt.close(fig)
    return fig


def fig4_stream_distribution(
    norm_ratios: np.ndarray,
    concentration: np.ndarray,
    output_dir: Path,
) -> matplotlib.figure.Figure:
    """Plot KromCanon stream distribution analysis.

    Args:
        norm_ratios: Shape (n_layers, n_streams) — fraction of norm per stream.
        concentration: Shape (n_layers,) — 0=uniform, 1=single stream.
        output_dir: Directory to save the figure.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_layers, n_streams = norm_ratios.shape
    layers = np.arange(n_layers)

    # Left: stacked bar of norm ratios per stream
    bottom = np.zeros(n_layers)
    stream_colors = plt.cm.Greens(np.linspace(0.3, 0.9, n_streams))
    for s in range(n_streams):
        ax1.bar(
            layers, norm_ratios[:, s], bottom=bottom,
            label=f"Stream {s}", color=stream_colors[s], width=0.7,
        )
        bottom += norm_ratios[:, s]
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Norm Ratio")
    ax1.set_title("Direction Distribution Across Streams")
    ax1.legend(frameon=True, fancybox=False, edgecolor="0.8")
    ax1.set_xticks(layers)

    # Right: concentration per layer
    ax2.plot(
        layers, concentration,
        color=_get_arch_color("kromcanon"),
        linewidth=2, marker="s", markersize=6,
    )
    ax2.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5, label="Uniform")
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Single stream")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Concentration (0=uniform, 1=single)")
    ax2.set_title("Stream Concentration per Layer")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(layers)
    ax2.legend(frameon=True, fancybox=False, edgecolor="0.8")

    fig.tight_layout()
    _save_fig(fig, output_dir, "fig4_stream_distribution")
    plt.close(fig)
    return fig


def fig5_alpha_sweep(
    sweep_data: dict[str, tuple[list[float], np.ndarray]],
    output_dir: Path,
) -> matplotlib.figure.Figure:
    """Plot steering dose-response curves.

    For each architecture, shows how a steering metric (e.g. max logit shift
    or KL divergence from baseline) changes with alpha.

    Args:
        sweep_data: Dict mapping arch → (alphas, metric_values).
            alphas: list of alpha values.
            metric_values: array of shape (n_alphas,) — e.g. KL from baseline.
        output_dir: Directory to save the figure.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for arch in ARCH_ORDER:
        if arch not in sweep_data:
            continue
        alphas, metrics = sweep_data[arch]
        ax.plot(
            alphas, metrics,
            label=arch, color=_get_arch_color(arch),
            linewidth=2, marker="o", markersize=4,
        )

    ax.set_xlabel("Steering Alpha")
    ax.set_ylabel("Logit Shift (KL from baseline)")
    ax.set_title("Steering Dose-Response")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8")
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig5_alpha_sweep")
    plt.close(fig)
    return fig


def fig6_abliteration_bars(
    refusal_rates: dict[str, dict[str, float]],
    output_dir: Path,
) -> matplotlib.figure.Figure:
    """Plot before/after abliteration refusal rates.

    Args:
        refusal_rates: Dict mapping arch → {"before": rate, "after": rate}.
        output_dir: Directory to save the figure.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    archs = [a for a in ARCH_ORDER if a in refusal_rates]
    x = np.arange(len(archs))
    width = 0.35

    before_vals = [refusal_rates[a]["before"] for a in archs]
    after_vals = [refusal_rates[a]["after"] for a in archs]
    colors = [_get_arch_color(a) for a in archs]

    bars_before = ax.bar(
        x - width / 2, before_vals, width,
        label="Before abliteration", color=colors, alpha=0.6, edgecolor="0.3",
    )
    bars_after = ax.bar(
        x + width / 2, after_vals, width,
        label="After abliteration", color=colors, alpha=1.0, edgecolor="0.3",
    )

    # Add value labels on bars
    for bar in [*bars_before, *bars_after]:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Architecture")
    ax.set_ylabel("Refusal Rate")
    ax.set_title("Abliteration: Before vs After Refusal Rates")
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8")
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig6_abliteration_bars")
    plt.close(fig)
    return fig


def fig7_stream_cosines(
    stream_cosine_matrix: np.ndarray,
    output_dir: Path,
) -> matplotlib.figure.Figure:
    """Plot pairwise stream direction cosine similarity matrix.

    Args:
        stream_cosine_matrix: Shape (n_streams, n_streams) — averaged over layers.
        output_dir: Directory to save the figure.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_style()
    n_streams = stream_cosine_matrix.shape[0]
    labels = [f"Stream {i}" for i in range(n_streams)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        stream_cosine_matrix,
        annot=True, fmt=".3f",
        xticklabels=labels, yticklabels=labels,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        ax=ax, square=True,
        cbar_kws={"label": "Cosine Similarity"},
    )
    ax.set_title("KromCanon: Pairwise Stream Direction Similarity")
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig7_stream_cosines")
    plt.close(fig)
    return fig


def fig8_method_comparison(
    mean_diff_norms: dict[str, np.ndarray],
    svd_norms: dict[str, np.ndarray],
    output_dir: Path,
) -> matplotlib.figure.Figure:
    """Plot mean-diff vs SVD direction agreement.

    Shows per-layer norm profiles for both methods to check if the refusal
    direction is truly rank-1 in each architecture.

    Args:
        mean_diff_norms: Dict mapping arch → array of shape (n_layers,).
        svd_norms: Dict mapping arch → array of shape (n_layers,) — top singular value.
        output_dir: Directory to save the figure.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    archs = [a for a in ARCH_ORDER if a in mean_diff_norms and a in svd_norms]
    n_archs = len(archs)
    if n_archs == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _save_fig(fig, output_dir, "fig8_method_comparison")
        plt.close(fig)
        return fig

    fig, axes = plt.subplots(1, n_archs, figsize=(5 * n_archs, 5), squeeze=False)

    for i, arch in enumerate(archs):
        ax = axes[0, i]
        n_layers = len(mean_diff_norms[arch])
        layers = np.arange(n_layers)

        ax.plot(
            layers, mean_diff_norms[arch],
            label="Mean-diff", color=_get_arch_color(arch),
            linewidth=2, marker="o", markersize=5,
        )
        ax.plot(
            layers, svd_norms[arch],
            label="SVD (top-1)", color=_get_arch_color(arch),
            linewidth=2, marker="^", markersize=5, linestyle="--",
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Direction Norm / Singular Value")
        ax.set_title(f"{arch}")
        ax.legend(frameon=True, fancybox=False, edgecolor="0.8")
        ax.set_xticks(layers)

    fig.suptitle("Mean-Diff vs SVD: Is Refusal Rank-1?", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig8_method_comparison")
    plt.close(fig)
    return fig


def generate_all_figures(
    training_logs: dict[str, list[dict[str, float]]],
    direction_norms: dict[str, np.ndarray],
    cosine_matrix: np.ndarray,
    arch_names: list[str],
    stream_norm_ratios: np.ndarray | None,
    stream_concentration: np.ndarray | None,
    alpha_sweep_data: dict[str, tuple[list[float], np.ndarray]],
    refusal_rates: dict[str, dict[str, float]],
    stream_cosine_matrix: np.ndarray | None,
    mean_diff_norms: dict[str, np.ndarray],
    svd_norms: dict[str, np.ndarray],
    output_dir: Path,
) -> list[Path]:
    """Generate all 8 figures.

    Args:
        training_logs: Per-architecture training logs.
        direction_norms: Per-architecture direction norm profiles.
        cosine_matrix: Cross-architecture cosine similarity matrix.
        arch_names: Architecture names for heatmap labels.
        stream_norm_ratios: KromCanon norm ratios (n_layers, n_streams) or None.
        stream_concentration: KromCanon concentration (n_layers,) or None.
        alpha_sweep_data: Per-architecture steering sweep data.
        refusal_rates: Per-architecture before/after refusal rates.
        stream_cosine_matrix: KromCanon stream cosines (n_streams, n_streams) or None.
        mean_diff_norms: Per-architecture mean-diff direction norms.
        svd_norms: Per-architecture SVD top singular values.
        output_dir: Directory to save all figures.

    Returns:
        List of paths to generated PDF files.
    """
    paths: list[Path] = []

    fig1_training_curves(training_logs, output_dir)
    paths.append(output_dir / "fig1_training_curves.pdf")

    fig2_direction_norms(direction_norms, output_dir)
    paths.append(output_dir / "fig2_direction_norms.pdf")

    fig3_cosine_heatmap(cosine_matrix, arch_names, output_dir)
    paths.append(output_dir / "fig3_cosine_heatmap.pdf")

    if stream_norm_ratios is not None and stream_concentration is not None:
        fig4_stream_distribution(stream_norm_ratios, stream_concentration, output_dir)
        paths.append(output_dir / "fig4_stream_distribution.pdf")

    fig5_alpha_sweep(alpha_sweep_data, output_dir)
    paths.append(output_dir / "fig5_alpha_sweep.pdf")

    fig6_abliteration_bars(refusal_rates, output_dir)
    paths.append(output_dir / "fig6_abliteration_bars.pdf")

    if stream_cosine_matrix is not None:
        fig7_stream_cosines(stream_cosine_matrix, output_dir)
        paths.append(output_dir / "fig7_stream_cosines.pdf")

    fig8_method_comparison(mean_diff_norms, svd_norms, output_dir)
    paths.append(output_dir / "fig8_method_comparison.pdf")

    return paths
