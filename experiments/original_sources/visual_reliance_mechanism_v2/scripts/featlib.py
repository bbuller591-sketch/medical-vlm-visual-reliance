#!/usr/bin/env python
"""Feature loading helpers shared by the probe, TVI, and causal scripts.

Features are stored sharded per (model, condition, option_order) as
`features/<model>/<condition>/<order>/shard_XXXX.npz` with arrays

    decision_hidden        [n_items, T, 1536]  fp16
    visual_pooled          [n_items, T, 1536]  fp16   (NaN where there is no image)
    layer_candidate_logits [n_items, T, n_cand] fp32  (logit lens)
    vision_encoder_pool    [n_items, 1536]     fp16   (NaN rows when no image)
    sample_ids             [n_items]

with T = 29 for Qwen2-VL-2B.  The tuple was decoded EMPIRICALLY, not assumed:

    row 0    == embed_tokens(last token) to 0.00000 (bitwise)
    row 28   == the state the LM head consumes: head(row28) reproduces the model's own
                logits to 0.031 in logit units (bf16), while head(norm(row28)) is off by
                12.0
    rows 1..27 are the 27 states in between.

So the labelling is `embed`, `L1`..`L27`, `L28+final_norm`.  Anything that patches a
layer must respect the +-1: hooking `layers[k]` (0-indexed) produces row k+1.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import common_v2 as V

FEAT = V.ROOT / "features"
CAND = "ABCDZ"


def layer_label(t_index: int, T: int) -> str:
    """Human label for a hidden-state row, matching the empirical decode above."""
    if t_index == 0:
        return "embed"
    if t_index == T - 1:
        return "L28+norm"
    return f"L{t_index}"


def _shards(model: str, cond: str, order: str):
    d = FEAT / model / cond / order
    return sorted(d.glob("shard_*.npz")) if d.is_dir() else []


def has(model: str, cond: str, order: str = "O0") -> bool:
    return bool(_shards(model, cond, order))


def load_meta(model: str, cond: str, order: str = "O0") -> list[dict]:
    d = FEAT / model / cond / order
    out = []
    for p in sorted(d.glob("shard_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    out.append(r)
    return out


def load_arrays(model: str, cond: str, order: str = "O0",
                keys=("decision_hidden", "visual_pooled", "layer_candidate_logits",
                      "vision_encoder_pool")) -> dict:
    """Concatenate all shards; returns dict of arrays plus sample_ids + meta."""
    parts: dict = defaultdict(list)
    metas: list[dict] = []
    for p in _shards(model, cond, order):
        z = np.load(p)
        for k in keys:
            if k in z.files:
                parts[k].append(z[k])
        d = p.with_suffix(".jsonl")
        if d.is_file():
            metas += [json.loads(l) for l in d.read_text(encoding="utf-8").splitlines()
                      if l.strip()]
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    out["sample_ids"] = np.array([m["sample_id"] for m in metas])
    out["meta"] = metas
    return out


def align(model: str, cond_a: str, cond_b: str, order: str = "O0"):
    """Load two conditions and return rows matched by sample_id (sorted)."""
    A = load_arrays(model, cond_a, order)
    B = load_arrays(model, cond_b, order)
    ia = {s: i for i, s in enumerate(A["sample_ids"])}
    ib = {s: i for i, s in enumerate(B["sample_ids"])}
    common = sorted(set(ia) & set(ib))
    idx_a = np.array([ia[s] for s in common])
    idx_b = np.array([ib[s] for s in common])
    return common, {k: v[idx_a] for k, v in A.items()
                    if isinstance(v, np.ndarray) and v.shape[0] == len(A["sample_ids"])}, \
        {k: v[idx_b] for k, v in B.items()
         if isinstance(v, np.ndarray) and v.shape[0] == len(B["sample_ids"])}


def semantic_probs_from_logits(lens_row: np.ndarray, n_options: int,
                               permutation: list[int]) -> np.ndarray:
    """Map display-letter logit-lens values back to semantic option identity.

    `lens_row` is [n_cand] in display order A,B,C,D(,Z); `permutation` is
    display position -> semantic index.  Only the n_options real options are kept.
    """
    z = lens_row[:n_options]
    z = z - z.max()
    e = np.exp(z)
    p = e / e.sum()
    out = np.zeros(n_options)
    for disp, sem in enumerate(permutation[:n_options]):
        out[sem] = p[disp]
    return out


def gt_margin_by_layer(meta: list[dict], lens: np.ndarray, items: dict) -> np.ndarray:
    """[n_items, T] GT-vs-best-wrong margin from the per-layer logit lens."""
    T = lens.shape[1]
    out = np.empty((len(meta), T), dtype=np.float32)
    for i, m in enumerate(meta):
        it = items[m["sample_id"]]
        n = it["n_options"]
        gt = it["gt_index"]
        perm = m["permutation"]
        for t in range(T):
            p = semantic_probs_from_logits(lens[i, t], n, perm)
            others = np.delete(p, gt)
            out[i, t] = p[gt] - (others.max() if len(others) else 0.0)
    return out


def cosine_distance_by_layer(A_dec: np.ndarray, B_dec: np.ndarray) -> np.ndarray:
    """[T] mean cosine distance between two aligned sets of decision states."""
    a = A_dec.astype(np.float32)
    b = B_dec.astype(np.float32)
    na = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9
    nb = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9
    cos = ((a / na) * (b / nb)).sum(-1)          # [n, T]
    return (1.0 - cos).mean(0)                   # [T]


def vip_tvi(d: np.ndarray, frac: float = 0.5) -> dict:
    """VIP-like change point and TVI-like area, on a per-layer distance curve.

    Named *-like on purpose: the ICLR 2026 VIP/TVI paper defines these on a
    chain-of-embedding with its own aggregation, which we do not reproduce exactly.
    """
    d = np.asarray(d, dtype=float)
    base = float(d[0])
    peak = float(d.max())
    span = max(peak - base, 1e-12)
    cum = np.cumsum(np.clip(d - base, 0, None))
    total = max(cum[-1], 1e-12)
    vip_idx = int(np.searchsorted(cum / total, frac))
    return {
        "tvi_like": float(total / len(d)),          # mean excess distance over layers
        "tvi_like_sum": float(total),
        "vip_like_layer_index": vip_idx,
        "vip_like_layer_label": layer_label(vip_idx, len(d)),
        "frac": frac,
        "base_distance": base, "peak_distance": peak,
    }
