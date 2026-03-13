"""Configuration dataclasses for Vanilla, Canon, and KromCanon architectures."""

from dataclasses import dataclass, field


@dataclass
class CanonConfig:
    """Configuration for Canon layers (1-D causal convolutions)."""

    enabled: bool = False
    canon_set: str = "AB"  # Which canon layers: "", "A", "AB", "ABCD"
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


@dataclass
class ModelConfig:
    """Full model configuration."""

    # Architecture
    arch: str = "vanilla"  # "vanilla", "canon", "kromcanon"
    vocab_size: int = 32768
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
        elif self.arch == "kromcanon":
            self.canon = CanonConfig(enabled=True)
            self.kromhc = KromHCConfig(enabled=True)
        elif self.arch != "vanilla":
            msg = f"Unknown architecture: {self.arch!r}. Must be 'vanilla', 'canon', or 'kromcanon'."
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

    # KromHC-specific optimizer params (AdamW for HC params)
    hc_lr: float = 5e-3
    hc_betas: tuple[float, float] = (0.8, 0.95)
    hc_weight_decay: float = 0.2

    # Logging
    log_interval: int = 10
    eval_interval: int = 250
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"


def make_config(arch: str = "vanilla", depth: int = 12) -> ModelConfig:
    """Create a model config for the given architecture and depth.

    Args:
        arch: Architecture variant — "vanilla", "canon", or "kromcanon".
        depth: Number of transformer layers.

    Returns:
        Configured ModelConfig instance.
    """
    return ModelConfig(arch=arch, n_layers=depth)
