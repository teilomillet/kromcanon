# KromCanon

Studying how steering vectors and linear direction extraction behave in non-standard transformer architectures.

## Research Question

> Do abliteration-style techniques generalize to architectures with multi-stream residual coupling (KromHC) and local token mixing (Canon layers)?

## The Experiment

Train three GPT-2 124M variants from scratch on Apple Silicon, with identical data and hyperparameters:

1. **Vanilla** — standard GPT-2
2. **Canon** — GPT-2 + Canon layers (1-D causal conv, kernel=4)
3. **KromCanon** — GPT-2 + Canon layers + KromHC residual connections (n=4 streams)

Then safety fine-tune all three and compare interpretability properties:
- Are refusal directions still linear?
- Does multi-stream coupling spread or concentrate directions?
- Does local convolution affect direction extraction?

## References

- [Canon Layers](https://arxiv.org/abs/2512.17351) — Allen-Zhu (Physics of LMs Part 4.1)
- [KromHC](https://arxiv.org/abs/2601.21579) — Wang et al.
- [Hyper-Connections](https://arxiv.org/abs/2409.19606) — foundation for KromHC
- [Abliteration](https://arxiv.org/abs/2406.11717) — Arditi et al.

## Infrastructure

- Training: [nanochat-mlx](https://github.com/scasella/nanochat-mlx) (pure MLX)
- Hardware: Apple M4 Pro, 24 GB
- Training time: ~30-40 min per variant
