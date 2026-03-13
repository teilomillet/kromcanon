"""KromHC: Kronecker-product doubly stochastic multi-stream residual connections.

MLX port of Wang et al. (2025) — arxiv.org/abs/2601.21579.

Key idea: Replace standard residual connections (x = x + F(x)) with multi-stream
mixing using doubly stochastic matrices built via Kronecker products of small factors.

For n=4 streams: H^res = U_1 ⊗ U_2, where each U_k is a 2x2 doubly stochastic matrix
parametrized as a convex combination of permutation matrices (identity and swap).
"""

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from kromcanon.config import KromHCConfig


class KromHCLayer(nn.Module):
    """KromHC residual connection for a single transformer layer.

    Replaces `x = x + F(x)` with multi-stream mixing:
        X_{l+1} = H^res · X_l + H^post · F(H^pre · X_l)

    Where H^res is a doubly stochastic matrix built as a Kronecker product
    of learned 2x2 factors.

    Args:
        d_model: Model hidden dimension.
        n_streams: Number of residual streams (default 4).
        layer_index: Index of this layer (for initialization).
        config: KromHC configuration.
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int,
        layer_index: int,
        config: KromHCConfig,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_streams = n_streams
        self.layer_index = layer_index
        self.config = config
        self.kronecker_factors = config.kronecker_factors  # e.g. [2, 2] for n=4

        # Validate factors multiply to n_streams
        product = 1
        for f in self.kronecker_factors:
            product *= f
        if product != n_streams:
            msg = (
                f"Kronecker factors {self.kronecker_factors} product "
                f"{product} != n_streams {n_streams}"
            )
            raise ValueError(msg)

        # --- H^pre coefficients: select which stream(s) to feed the branch ---
        # Shape: (n_streams,) — sigmoid → weights for combining streams into branch input
        self._init_pre_params()

        # --- H^post coefficients: distribute branch output back to streams ---
        # Shape: (n_streams,) — sigmoid×2 → weights for distributing branch output
        self._init_post_params()

        # --- H^res: Kronecker factors for residual mixing matrix ---
        self._init_res_params()

        # Scale parameters for dynamic coefficients
        self.alpha_pre = mx.array(config.alpha_init)
        self.alpha_post = mx.array(config.alpha_init)
        self.alpha_res = mx.array(config.alpha_init)

        # Dynamic projection weights (initialized to zero → starts as static)
        if config.dynamic:
            self.norm = nn.RMSNorm(n_streams * d_model)
            self.w_pre = mx.zeros((n_streams * d_model, n_streams))
            self.w_post = mx.zeros((n_streams * d_model, n_streams))
            # One projection per Kronecker factor
            self.w_res: list[mx.array] = []
            for factor_size in self.kronecker_factors:
                n_perms = _factorial(factor_size)
                self.w_res.append(mx.zeros((n_streams * d_model, n_perms)))

    def _init_pre_params(self) -> None:
        """Initialize H^pre bias: select stream at (layer_index % n_streams)."""
        # b^pre = [-1, ..., -1, 1, -1, ..., -1] with 1 at layer_index % n
        b_pre = mx.full((self.n_streams,), -1.0)
        idx = self.layer_index % self.n_streams
        b_pre = b_pre.at[idx].add(2.0)  # -1 + 2 = 1
        self.b_pre = b_pre

    def _init_post_params(self) -> None:
        """Initialize H^post bias: write to stream at (layer_index % n_streams)."""
        b_post = mx.full((self.n_streams,), -1.0)
        idx = self.layer_index % self.n_streams
        b_post = b_post.at[idx].add(2.0)
        self.b_post = b_post

    def _init_res_params(self) -> None:
        """Initialize Kronecker factor biases for H^res ≈ Identity at init.

        For 2x2 factors: b^res = [0, -8] → softmax ≈ [1, 0] → U ≈ Identity.
        """
        self.b_res: list[mx.array] = []
        for factor_size in self.kronecker_factors:
            n_perms = _factorial(factor_size)
            # First permutation is identity, rest get large negative bias
            b = mx.full((n_perms,), self.config.bias_res_init)
            b = b.at[0].add(-self.config.bias_res_init)  # b[0] = 0
            self.b_res.append(b)

    def _compute_hpre(self, x_flat: mx.array | None) -> mx.array:
        """Compute H^pre coefficients (stream → branch input weights).

        Args:
            x_flat: Flattened residual streams for dynamic computation,
                    shape (batch, seq_len, n_streams * d_model). None if static.

        Returns:
            H^pre weights, shape (batch, seq_len, n_streams) or (n_streams,).
        """
        coeffs = self.b_pre
        if self.config.dynamic and x_flat is not None:
            normed = self.norm(x_flat)
            dynamic = self.alpha_pre * mx.tanh(normed @ self.w_pre)
            coeffs = dynamic + coeffs
        return mx.sigmoid(coeffs)

    def _compute_hpost(self, x_flat: mx.array | None) -> mx.array:
        """Compute H^post coefficients (branch output → stream distribution weights).

        Args:
            x_flat: Flattened residual streams for dynamic computation.

        Returns:
            H^post weights, shape (batch, seq_len, n_streams) or (n_streams,).
        """
        coeffs = self.b_post
        if self.config.dynamic and x_flat is not None:
            normed = self.norm(x_flat)
            dynamic = self.alpha_post * mx.tanh(normed @ self.w_post)
            coeffs = dynamic + coeffs
        return 2.0 * mx.sigmoid(coeffs)

    def _build_kronecker_hres(self, x_flat: mx.array | None) -> mx.array:
        """Build H^res as Kronecker product of doubly stochastic factors.

        Each factor U_k is a convex combination of permutation matrices for that factor size.
        For 2x2: U = p * I + (1-p) * [[0,1],[1,0]], where p = softmax(coeffs)[0].

        Args:
            x_flat: Flattened residual streams for dynamic computation.

        Returns:
            H^res matrix, shape (batch, seq_len, n_streams, n_streams) or (n_streams, n_streams).
        """
        hres: mx.array | None = None

        for k, factor_size in enumerate(self.kronecker_factors):
            # Compute factor coefficients
            coeffs = self.b_res[k]
            if self.config.dynamic and x_flat is not None:
                normed = self.norm(x_flat)
                dynamic = self.alpha_res * mx.tanh(normed @ self.w_res[k])
                coeffs = dynamic + coeffs

            # Softmax to get convex combination weights
            weights = mx.softmax(coeffs, axis=-1)

            # Build factor matrix as weighted sum of permutation matrices
            factor = _build_doubly_stochastic_factor(weights, factor_size)

            # Kronecker product with accumulated result
            hres = factor if hres is None else _kronecker_product(hres, factor)

        assert hres is not None
        return hres

    def width_connection(self, residuals: mx.array) -> tuple[mx.array, mx.array]:
        """Mix streams to produce branch input and updated residuals.

        Takes the multi-stream residuals and produces:
        1. branch_input: weighted combination of streams → single stream for the branch (attn/ffn)
        2. updated_residuals: residual streams mixed via H^res

        Args:
            residuals: Multi-stream residuals, shape (batch, n_streams, seq_len, d_model).

        Returns:
            Tuple of:
                - branch_input: shape (batch, seq_len, d_model)
                - residuals_mixed: shape (batch, n_streams, seq_len, d_model)
        """
        b, n, t, d = residuals.shape

        # Flatten streams for dynamic coefficient computation
        x_flat: mx.array | None = None
        if self.config.dynamic:
            # (batch, n_streams, seq_len, d_model) → (batch, seq_len, n_streams * d_model)
            x_flat = residuals.transpose(0, 2, 1, 3).reshape(b, t, n * d)

        # H^pre: (batch, seq_len, n_streams) or (n_streams,)
        h_pre = self._compute_hpre(x_flat)

        # Branch input: weighted sum of streams
        # h_pre: (..., n_streams) → (..., n_streams, 1) for broadcasting
        if h_pre.ndim == 1:
            # Static: broadcast over batch and seq
            branch_input = (residuals * h_pre[None, :, None, None]).sum(axis=1)
        else:
            # Dynamic: h_pre is (batch, seq_len, n_streams)
            # residuals: (batch, n_streams, seq_len, d_model)
            h_pre_expanded = h_pre.transpose(0, 2, 1)[..., None]  # (batch, n_streams, seq_len, 1)
            branch_input = (residuals * h_pre_expanded).sum(axis=1)  # (batch, seq_len, d_model)

        # H^res: mix residual streams
        h_res = self._build_kronecker_hres(x_flat)

        if h_res.ndim == 2:
            # Static: (n_streams, n_streams) @ (batch, n_streams, seq_len, d_model)
            # einsum: 'ij, bjsd -> bisd'
            residuals_mixed = mx.einsum("ij,bjsd->bisd", h_res, residuals)
        else:
            # Dynamic: h_res is (batch, seq_len, n_streams, n_streams)
            # residuals: (batch, n_streams, seq_len, d_model)
            # → rearrange for per-token mixing
            # residuals_t: (batch, seq_len, n_streams, d_model)
            residuals_t = residuals.transpose(0, 2, 1, 3)
            # h_res: (batch, seq_len, n_streams, n_streams)
            # out: (batch, seq_len, n_streams, d_model)
            residuals_mixed_t = mx.einsum("bsij,bsjd->bsid", h_res, residuals_t)
            # → (batch, n_streams, seq_len, d_model)
            residuals_mixed = residuals_mixed_t.transpose(0, 2, 1, 3)

        return branch_input, residuals_mixed

    def depth_connection(
        self, branch_output: mx.array, residuals: mx.array
    ) -> mx.array:
        """Combine branch output back into multi-stream residuals.

        Args:
            branch_output: Output from branch (attn/ffn), shape (batch, seq_len, d_model).
            residuals: Mixed residual streams from width_connection,
                      shape (batch, n_streams, seq_len, d_model).

        Returns:
            Updated residuals, shape (batch, n_streams, seq_len, d_model).
        """
        b, n, t, d = residuals.shape

        # Compute H^post
        x_flat: mx.array | None = None
        if self.config.dynamic:
            x_flat = residuals.transpose(0, 2, 1, 3).reshape(b, t, n * d)

        h_post = self._compute_hpost(x_flat)

        # Distribute branch output to streams via H^post
        if h_post.ndim == 1:
            # Static: (n_streams,) → (1, n_streams, 1, 1)
            distributed = branch_output[:, None, :, :] * h_post[None, :, None, None]
        else:
            # Dynamic: h_post is (batch, seq_len, n_streams)
            # → (batch, n_streams, seq_len, 1)
            h_post_expanded = h_post.transpose(0, 2, 1)[..., None]
            distributed = branch_output[:, None, :, :] * h_post_expanded

        return residuals + distributed

    def __call__(
        self, residuals: mx.array, branch_fn: Callable[[mx.array], mx.array]
    ) -> mx.array:
        """Full forward pass: width connection → branch → depth connection.

        Args:
            residuals: Multi-stream residuals, shape (batch, n_streams, seq_len, d_model).
            branch_fn: The transformer branch function (attention or FFN).

        Returns:
            Updated residuals, shape (batch, n_streams, seq_len, d_model).
        """
        branch_input, residuals_mixed = self.width_connection(residuals)
        branch_output = branch_fn(branch_input)
        return self.depth_connection(branch_output, residuals_mixed)


class KromHCInit(nn.Module):
    """Initialize multi-stream residuals from single-stream input.

    Expands (batch, seq_len, d_model) → (batch, n_streams, seq_len, d_model)
    by replicating the input across all streams.

    Args:
        n_streams: Number of residual streams.
    """

    def __init__(self, n_streams: int) -> None:
        super().__init__()
        self.n_streams = n_streams

    def __call__(self, x: mx.array) -> mx.array:
        """Expand single stream to multi-stream.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Multi-stream tensor, shape (batch, n_streams, seq_len, d_model).
        """
        # (batch, seq_len, d_model) → (batch, 1, seq_len, d_model) → repeat
        return mx.repeat(x[:, None, :, :], repeats=self.n_streams, axis=1)


class KromHCReduce(nn.Module):
    """Reduce multi-stream residuals back to single stream.

    Averages across streams: (batch, n_streams, seq_len, d_model) → (batch, seq_len, d_model).

    Args:
        n_streams: Number of residual streams.
    """

    def __init__(self, n_streams: int) -> None:
        super().__init__()
        self.n_streams = n_streams

    def __call__(self, residuals: mx.array) -> mx.array:
        """Reduce multi-stream to single stream via mean.

        Args:
            residuals: Multi-stream tensor, shape (batch, n_streams, seq_len, d_model).

        Returns:
            Single-stream tensor, shape (batch, seq_len, d_model).
        """
        return residuals.mean(axis=1)


# --- Helper functions ---


def _factorial(n: int) -> int:
    """Compute n! for small n (used for permutation count)."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _build_doubly_stochastic_factor(weights: mx.array, factor_size: int) -> mx.array:
    """Build a doubly stochastic matrix as convex combination of permutation matrices.

    For factor_size=2 (optimized path):
        U = p * I + (1-p) * swap = [[p, 1-p], [1-p, p]]
        where p = weights[..., 0]

    Args:
        weights: Softmax weights over permutations, shape (..., n_perms).
        factor_size: Size of the factor matrix.

    Returns:
        Doubly stochastic matrix, shape (..., factor_size, factor_size).
    """
    if factor_size == 2:
        return _build_2x2_factor(weights)
    return _build_general_factor(weights, factor_size)


def _build_2x2_factor(weights: mx.array) -> mx.array:
    """Optimized path for 2x2 doubly stochastic factors.

    U = [[p, 1-p], [1-p, p]] where p = weights[..., 0].

    Args:
        weights: Softmax weights, shape (..., 2).

    Returns:
        Factor matrix, shape (..., 2, 2).
    """
    p = weights[..., 0:1]  # (..., 1)
    one_minus_p = weights[..., 1:2]  # (..., 1)
    # Build rows: row0 = [p, 1-p], row1 = [1-p, p]
    row0 = mx.concatenate([p, one_minus_p], axis=-1)  # (..., 2)
    row1 = mx.concatenate([one_minus_p, p], axis=-1)  # (..., 2)
    return mx.stack([row0, row1], axis=-2)  # (..., 2, 2)


def _build_general_factor(weights: mx.array, factor_size: int) -> mx.array:
    """General path for n×n doubly stochastic factors via Birkhoff-von Neumann.

    Builds all permutation matrices of size factor_size and takes their
    weighted combination.

    Args:
        weights: Softmax weights over permutations, shape (..., n!).
        factor_size: Size of the factor matrix.

    Returns:
        Factor matrix, shape (..., factor_size, factor_size).
    """
    perms = _all_permutation_matrices(factor_size)  # (n!, factor_size, factor_size)
    # weights: (..., n!) → (..., n!, 1, 1)
    w = weights[..., None, None]
    # perms: (n!, factor_size, factor_size) broadcast with w
    return (w * perms).sum(axis=-3)


def _all_permutation_matrices(n: int) -> mx.array:
    """Generate all n! permutation matrices of size n×n.

    Args:
        n: Matrix size.

    Returns:
        Tensor of shape (n!, n, n) containing all permutation matrices.
    """
    from itertools import permutations

    perms_list: list[list[list[float]]] = []
    for perm in permutations(range(n)):
        mat = [[1.0 if j == perm[i] else 0.0 for j in range(n)] for i in range(n)]
        perms_list.append(mat)
    return mx.array(perms_list)


def _kronecker_product(a: mx.array, b: mx.array) -> mx.array:
    """Compute the Kronecker product of two matrices (with batch dimensions).

    For A of shape (..., m, m) and B of shape (..., n, n),
    produces (..., m*n, m*n).

    Args:
        a: First matrix, shape (..., m, m).
        b: Second matrix, shape (..., n, n).

    Returns:
        Kronecker product, shape (..., m*n, m*n).
    """
    # Get the last two dimensions
    m = a.shape[-1]
    n = b.shape[-1]
    batch_shape = a.shape[:-2]

    # a: (..., m, 1, m, 1) * b: (..., 1, n, 1, n) → (..., m, n, m, n)
    a_expanded = a[..., :, None, :, None]
    b_expanded = b[..., None, :, None, :]
    result = a_expanded * b_expanded

    # Reshape (..., m, n, m, n) → (..., m*n, m*n)
    return result.reshape(*batch_shape, m * n, m * n)
