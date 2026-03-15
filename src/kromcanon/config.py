"""Configuration dataclasses for Vanilla, Canon, and KromCanon architectures."""

from dataclasses import dataclass, field


@dataclass
class CanonConfig:
    """Configuration for Canon layers (1-D causal convolutions)."""

    enabled: bool = False
    canon_set: str = "ABCD"  # Which canon layers: "", "A", "AB", "ABCD"
    kernel_size: int = 4
    bias: bool = False
    residual: bool = True  # Add canon output to input


@dataclass
class KromHCConfig:
    """Configuration for KromHC residual connections."""

    enabled: bool = False
    n_streams: int = 4  # Number of residual streams
    kronecker_factors: list[int] = field(default_factory=lambda: [2, 2])  # n=4 → 2x2
    dynamic: bool = True  # Input-dependent coefficients
    alpha_init: float = 0.01  # Initial scale for dynamic params
    bias_res_init: float = -8.0  # Init for res bias → softmax ≈ [1, 0] → identity
    freeze_hres: bool = False  # If True, H^res is hard-coded to identity (ablation)


@dataclass
class ModelConfig:
    """Full model configuration."""

    # Architecture
    arch: str = "vanilla"  # "vanilla", "canon", "kromhc", "kromcanon"
    vocab_size: int = 50304  # GPT-2: 50257 tokens, padded to nearest 64 for kernel alignment
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072  # 4 * d_model
    max_seq_len: int = 2048
    dropout: float = 0.0
    norm_eps: float = 1e-5

    # Sub-configs
    canon: CanonConfig = field(default_factory=CanonConfig)
    kromhc: KromHCConfig = field(default_factory=KromHCConfig)

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        """Set sub-config enabled flags based on arch."""
        if self.arch == "canon":
            self.canon = CanonConfig(enabled=True)
        elif self.arch == "kromhc":
            self.kromhc = KromHCConfig(enabled=True)
        elif self.arch == "kromcanon":
            self.canon = CanonConfig(enabled=True)
            self.kromhc = KromHCConfig(enabled=True)
        elif self.arch != "vanilla":
            msg = (
                f"Unknown architecture: {self.arch!r}. "
                "Must be 'vanilla', 'canon', 'kromhc', or 'kromcanon'."
            )
            raise ValueError(msg)


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    dataset: str = "HuggingFaceFW/fineweb-edu"
    seq_len: int = 2048
    batch_size: int = 64  # Sequences per batch

    # Optimization
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 5000
    grad_clip: float = 1.0

    # Muon optimizer (for 2D weight matrices — attention/FFN projections)
    use_muon: bool = True
    muon_lr: float = 0.02

    # KromHC-specific optimizer params (AdamW for HC params)
    hc_lr: float = 5e-3
    hc_betas: tuple[float, float] = (0.8, 0.95)
    hc_weight_decay: float = 0.2

    # Logging
    log_interval: int = 10
    eval_interval: int = 250
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"


# Size presets: (n_heads, d_model, d_ff, max_seq_len)
# Aligned with Physics of LLMs Part 4.1 (Allen-Zhu, NeurIPS 2025)
SIZE_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "micro": (4, 256, 1024, 256),    # Pipeline validation — trains in seconds
    "small": (8, 512, 2048, 512),    # 8L512D from Physics of LLMs 4.1
    "medium": (12, 768, 3072, 2048), # 12L768D — GPT-2 small scale
}


def make_config(
    arch: str = "vanilla",
    depth: int = 12,
    size: str = "medium",
) -> ModelConfig:
    """Create a model config for the given architecture, depth, and size preset.

    Args:
        arch: Architecture variant — "vanilla", "canon", or "kromcanon".
        depth: Number of transformer layers.
        size: Size preset — "micro", "small", or "medium".

    Returns:
        Configured ModelConfig instance.
    """
    if size not in SIZE_PRESETS:
        msg = f"Unknown size preset: {size!r}. Must be one of {list(SIZE_PRESETS.keys())}."
        raise ValueError(msg)
    n_heads, d_model, d_ff, max_seq_len = SIZE_PRESETS[size]
    return ModelConfig(
        arch=arch,
        n_layers=depth,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
    )
