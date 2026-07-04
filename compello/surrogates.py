"""Differentiable tabular surrogates for classical ML (Section 7.5).

Two named architectures with a genuinely closer inductive bias to tree ensembles
than a plain MLP:

  * NODE  -- Neural Oblivious Decision Ensembles: a differentiable analogue of an
    oblivious decision-tree ensemble (soft threshold splits shared per depth).
  * FT-Transformer -- Feature Tokenizer + Transformer over per-feature tokens.

Both ship with:
  * a pure-numpy **reference forward** (``NodeReference`` / ``FTTransformerReference``)
    that runs with no DL framework -- exact same math, usable for CPU inference of
    a distilled student and, crucially, testable here for shape/finiteness; and
  * guarded **builders** (``build_node`` / ``build_ft_transformer``) that return a
    real ``torch.nn.Module`` when torch is installed.

The reference forwards are deterministic given a seed and are what let the
classical-ML path be validated without pulling in torch/tf/jax.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    _HAS_NUMPY = False


def _sigmoid(x):
    return 1.0 / (1.0 + _np.exp(-x))


def _softmax(x, axis=-1):
    x = x - _np.max(x, axis=axis, keepdims=True)
    e = _np.exp(x)
    return e / _np.sum(e, axis=axis, keepdims=True)


class NodeReference:
    """Pure-numpy forward of one Neural Oblivious Decision Ensemble layer (7.5).

    Each of ``n_trees`` oblivious trees has ``depth`` soft threshold splits; a
    split at a given depth applies the *same* learned feature-selector and
    threshold to every node at that depth (the "oblivious" property CatBoost
    uses). Leaf membership is the outer product of the per-depth soft decisions,
    and the output is the leaf-response-weighted sum, averaged over trees.
    """

    def __init__(self, in_features: int, out_features: int = 1, n_trees: int = 8,
                 depth: int = 3, steepness: float = 1.0, seed: int = 0):
        if not _HAS_NUMPY:  # pragma: no cover
            raise RuntimeError("NodeReference requires numpy")
        self.in_features = in_features
        self.out_features = out_features
        self.n_trees = n_trees
        self.depth = depth
        self.steepness = steepness
        rng = _np.random.default_rng(seed)
        self.feature_logits = rng.normal(0, 1, (n_trees, depth, in_features))
        self.thresholds = rng.normal(0, 1, (n_trees, depth))
        self.leaf_response = rng.normal(0, 1, (n_trees, 2 ** depth, out_features))

    def forward(self, x):
        x = _np.asarray(x, dtype=_np.float64)
        if x.ndim == 1:
            x = x[None, :]
        b = x.shape[0]
        fsel = _softmax(self.feature_logits, axis=-1)      # (T, D, F)
        # projected feature value per (tree, depth): (b, T, D)
        proj = _np.einsum("bf,tdf->btd", x, fsel)
        decision = _sigmoid(self.steepness * (proj - self.thresholds[None]))  # (b,T,D)

        # leaf membership = product over depths of chosen branch prob
        leaf = _np.ones((b, self.n_trees, 1))
        for d in range(self.depth):
            dd = decision[:, :, d][:, :, None]             # (b,T,1)
            go_right = dd
            go_left = 1.0 - dd
            leaf = _np.concatenate([leaf * go_left, leaf * go_right], axis=-1)
        # leaf: (b, T, 2^depth); response: (T, 2^depth, out)
        out = _np.einsum("btl,tlo->bto", leaf, self.leaf_response)
        return out.mean(axis=1)                            # (b, out)


class FTTransformerReference:
    """Pure-numpy forward of a minimal FT-Transformer block (7.5).

    Tokenizes each numeric feature into a ``d_token`` embedding, prepends a CLS
    token, runs one multi-head self-attention + FFN block (pre-norm, residual),
    and reads the CLS token through a linear head.
    """

    def __init__(self, in_features: int, out_features: int = 1, d_token: int = 16,
                 n_heads: int = 2, seed: int = 0):
        if not _HAS_NUMPY:  # pragma: no cover
            raise RuntimeError("FTTransformerReference requires numpy")
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"
        self.in_features = in_features
        self.out_features = out_features
        self.d = d_token
        self.h = n_heads
        rng = _np.random.default_rng(seed)
        self.feat_w = rng.normal(0, 0.1, (in_features, d_token))
        self.feat_b = rng.normal(0, 0.1, (in_features, d_token))
        self.cls = rng.normal(0, 0.1, (1, d_token))
        self.Wq = rng.normal(0, 0.1, (d_token, d_token))
        self.Wk = rng.normal(0, 0.1, (d_token, d_token))
        self.Wv = rng.normal(0, 0.1, (d_token, d_token))
        self.Wo = rng.normal(0, 0.1, (d_token, d_token))
        self.ff1 = rng.normal(0, 0.1, (d_token, d_token * 2))
        self.ff2 = rng.normal(0, 0.1, (d_token * 2, d_token))
        self.head = rng.normal(0, 0.1, (d_token, out_features))

    def _ln(self, x):
        m = x.mean(-1, keepdims=True)
        v = x.var(-1, keepdims=True)
        return (x - m) / _np.sqrt(v + 1e-5)

    def forward(self, x):
        x = _np.asarray(x, dtype=_np.float64)
        if x.ndim == 1:
            x = x[None, :]
        b = x.shape[0]
        # per-feature token embeddings: (b, F, d)
        tokens = x[:, :, None] * self.feat_w[None] + self.feat_b[None]
        cls = _np.broadcast_to(self.cls, (b, 1, self.d))
        seq = _np.concatenate([cls, tokens], axis=1)       # (b, F+1, d)

        # multi-head self-attention (pre-norm + residual)
        z = self._ln(seq)
        q = z @ self.Wq; k = z @ self.Wk; v = z @ self.Wv
        b_, n, d = q.shape
        hd = d // self.h

        def split(t):
            return t.reshape(b_, n, self.h, hd).transpose(0, 2, 1, 3)  # (b,h,n,hd)

        qh, kh, vh = split(q), split(k), split(v)
        scores = qh @ kh.transpose(0, 1, 3, 2) / _np.sqrt(hd)          # (b,h,n,n)
        attn = _softmax(scores, axis=-1)
        ctx = attn @ vh                                                # (b,h,n,hd)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(b_, n, d)
        seq = seq + ctx @ self.Wo

        # feed-forward (pre-norm + residual)
        z2 = self._ln(seq)
        ff = _np.maximum(z2 @ self.ff1, 0.0) @ self.ff2
        seq = seq + ff

        return seq[:, 0, :] @ self.head                                # CLS -> (b, out)


# -- guarded framework builders --------------------------------------------

def build_node(in_features: int, out_features: int = 1, *, backend: str = "numpy", **kw):
    """Return a NODE surrogate for the requested backend.

    ``backend="numpy"`` returns the reference forward (no framework needed);
    ``backend="torch"`` returns a real ``torch.nn.Module``.
    """
    if backend == "numpy":
        return NodeReference(in_features, out_features, **kw)
    if backend == "torch":
        return _build_node_torch(in_features, out_features, **kw)
    raise ValueError(f"unsupported backend {backend!r} for NODE")


def build_ft_transformer(in_features: int, out_features: int = 1, *, backend: str = "numpy", **kw):
    if backend == "numpy":
        return FTTransformerReference(in_features, out_features, **kw)
    if backend == "torch":
        return _build_ft_transformer_torch(in_features, out_features, **kw)
    raise ValueError(f"unsupported backend {backend!r} for FT-Transformer")


def _build_node_torch(in_features, out_features=1, n_trees=8, depth=3, steepness=1.0, **_):
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("build_node(backend='torch') requires torch") from exc

    class NODELayer(nn.Module):  # pragma: no cover - requires torch
        def __init__(self):
            super().__init__()
            self.feature_logits = nn.Parameter(torch.randn(n_trees, depth, in_features))
            self.thresholds = nn.Parameter(torch.randn(n_trees, depth))
            self.leaf_response = nn.Parameter(torch.randn(n_trees, 2 ** depth, out_features))
            self.steepness = steepness

        def forward(self, x):
            fsel = torch.softmax(self.feature_logits, dim=-1)
            proj = torch.einsum("bf,tdf->btd", x, fsel)
            dec = torch.sigmoid(self.steepness * (proj - self.thresholds[None]))
            leaf = torch.ones(x.shape[0], n_trees, 1, device=x.device)
            for d in range(depth):
                dd = dec[:, :, d].unsqueeze(-1)
                leaf = torch.cat([leaf * (1 - dd), leaf * dd], dim=-1)
            out = torch.einsum("btl,tlo->bto", leaf, self.leaf_response)
            return out.mean(dim=1)

    return NODELayer()


def _build_ft_transformer_torch(in_features, out_features=1, d_token=16, n_heads=2, **_):
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("build_ft_transformer(backend='torch') requires torch") from exc

    class FTTransformer(nn.Module):  # pragma: no cover - requires torch
        def __init__(self):
            super().__init__()
            self.feat_w = nn.Parameter(torch.randn(in_features, d_token) * 0.1)
            self.feat_b = nn.Parameter(torch.randn(in_features, d_token) * 0.1)
            self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.1)
            self.attn = nn.MultiheadAttention(d_token, n_heads, batch_first=True)
            self.ln1 = nn.LayerNorm(d_token)
            self.ln2 = nn.LayerNorm(d_token)
            self.ff = nn.Sequential(nn.Linear(d_token, d_token * 2), nn.ReLU(),
                                    nn.Linear(d_token * 2, d_token))
            self.head = nn.Linear(d_token, out_features)

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.feat_w + self.feat_b
            cls = self.cls.expand(x.shape[0], -1, -1)
            seq = torch.cat([cls, tokens], dim=1)
            z = self.ln1(seq)
            a, _ = self.attn(z, z, z)
            seq = seq + a
            seq = seq + self.ff(self.ln2(seq))
            return self.head(seq[:, 0, :])

    return FTTransformer()
