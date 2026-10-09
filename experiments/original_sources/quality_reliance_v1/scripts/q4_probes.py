#!/usr/bin/env python
"""Q4 -- Level 1: layerwise linear probing of degradation and mismatch state.

THE ONE CONTROL THAT MATTERS
----------------------------
For a matched-vs-mismatch probe, the matched class of a training fold shows
{image(i)} and the mismatch class shows {image(donor(i))}.  The derangement is
built inside closed subgroups and the folds are unions of subgroups, so those two
multisets are IDENTICAL on every training fold.  A probe that sees only the image
therefore has nothing to learn: it must report chance, and if it does not, the
construction is broken -- not the model.

That control is `vis_only_balanced`.  The previous round of this project shipped a
version of it that read 0.057 where it had to read 0.500, because the probe's
folds were a fresh GroupKFold rather than the folds the balance was established
in.  Here the balance is a property of the subgroup structure, and the fold
builder uses it, so `learned_direction_norm` collapsing to ~0 is observed rather
than hoped for.

WHAT A PROBE RESULT MEANS
-------------------------
Above chance => the information is LINEARLY DECODABLE from that layer's state.
It never means the model USES that information, and it never means the state is
causally responsible for the answer.  Decodable != integrated != used.

STATE KINDS
-----------
  decision : the hidden state at the answer position, per decoder layer
             (row 0 = input embedding, row k+1 = decoder layer k)
  vision   : the pooled output of the vision tower, before the language model
             sees anything -- used only as the balanced control
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402

RNG_SEED = 20260917
N_FOLDS = 5

# Any probe fit that raised.  Recorded rather than swallowed: a silent
# `continue` inside a layer loop produces a short curve that reads exactly like
# a finished one.  BUG 13 (the transfer block, where every layer >= 1 was
# dropped without a word) was only found by finally printing this list.
PROBE_FAILURES: list[dict] = []

# A fit that hit the iteration cap is reported as its own count.  An unconverged
# logistic fit is not an error and does not raise, so it would otherwise be
# invisible -- and 43-79% of the matched-vs-mismatch fits were hitting a 300
# cap while the resulting curve was being reported as a finished result.
PROBE_UNCONVERGED: list[dict] = []
PROBE_MAX_ITER = 2000


# --------------------------------------------------------------------------- #
def load_features(model: str) -> dict:
    """{arm: {'ids': [...], 'hidden': (n, rows, d), 'vis': (n, d)}}

    Feature files are written per (arm, order, worker) because item shards run in
    parallel and must not write the same path.  They are merged here, and a file
    that fails to decompress is reported rather than silently skipped -- a
    truncated archive previously produced a probe table that looked complete.
    """
    d = Q.ROOT / "features" / model
    out: dict = {}
    if not d.is_dir():
        return out
    parts: dict = {}
    for p in sorted(d.glob("*__*__*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        stem = p.name[:-len(".npz")]
        bits = stem.split("__")
        if len(bits) != 3:
            continue
        arm, order, _worker = bits
        if order != "O0":
            continue
        try:
            z = np.load(p, allow_pickle=False)
            h = z["hidden"].astype(np.float32)
            v = (z["vis_pooled"].astype(np.float32) if "vis_pooled" in z.files
                 else None)
            parts.setdefault(arm, []).append(
                ([str(x) for x in z["sample_ids"]], h, v))
        except Exception as exc:
            print(f"  !! unreadable feature file {p.name}: {type(exc).__name__}: {exc}")
    for arm, chunks in parts.items():
        ids, hids, viss = [], [], []
        for cids, ch, cv in chunks:
            ids += cids
            hids.append(ch)
            viss.append(cv)
        has_vis = all(v is not None for v in viss)
        dvis = max((v.shape[1] for v in viss if v is not None), default=0)
        V = np.full((len(ids), dvis), np.nan, dtype=np.float32)
        k = 0
        for cv in viss:
            if cv is not None:
                V[k:k + cv.shape[0], :cv.shape[1]] = cv
            k += (cv.shape[0] if cv is not None else 0)
        # order the concatenation so ids and rows stay aligned
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        out[arm] = {"ids": [ids[i] for i in order],
                    "hidden": np.concatenate(hids, axis=0)[order],
                    "vis": V[order], "has_vision": has_vis,
                    "n_parts": len(chunks)}
    return out


def folds_from_subgroups(ids: list[str], k: int = 1,
                         n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold id per sample, built as unions of CLOSED SUBGROUPS.

    This is the whole point: because the derangement never crosses a subgroup,
    every union of subgroups has identical matched and mismatch image multisets,
    so the training set of every fold is balanced.

    CALLERS MUST PASS A SET THAT IS A UNION OF WHOLE SUBGROUPS.  A partially
    present subgroup breaks the guarantee: the training fold would then contain
    some members of a subgroup whose donors are the missing members, so the two
    classes stop sharing one image multiset.  The first full run of this script
    passed whatever items happened to exist, and exactly the four controls whose
    item set was incomplete (n = 1436, 1588, 1606, 1716 instead of 1800) stopped
    reading 0.500 -- with fitted ||w|| of 2.4 to 5.5.  `complete_subgroups` below
    is what keeps that from happening.
    """
    sub = Q.load_subgroups(k)
    groups = sorted({sub[s] for s in ids if s in sub})
    if not groups:
        # no derangement information: fall back to a deterministic hash split,
        # which is still group-safe but carries NO balance guarantee
        return np.array([hash(s) % n_folds for s in ids])
    fold_of_group = {g: i % n_folds for i, g in enumerate(groups)}
    return np.array([fold_of_group.get(sub.get(s), 0) for s in ids])


def complete_subgroups(ids: list[str], k: int = 1) -> list[str]:
    """Keep only items whose closed subgroup is fully present in `ids`.

    The complement is what would otherwise leak: with a subgroup half present,
    the matched class shows half of a subgroup's images and the mismatch class
    shows the other half, and a vision-only probe can learn the difference.

    Membership is computed from ALL 900 items, not from `ids`.  Building it from
    `ids` is circular -- a half-present subgroup then looks like a complete
    smaller one, the filter passes everything, and the control silently loses its
    guarantee.  That mistake was made once here and is why this docstring exists.
    """
    sub = Q.load_subgroups(k)
    have = set(ids)
    members: dict = {}
    for s in Q.load_items():
        members.setdefault(sub.get(s), []).append(s)
    keep = {s for s in ids
            if sub.get(s) is not None
            and all(m in have for m in members.get(sub.get(s), []))
            and len(members.get(sub.get(s), [])) > 1}
    return [s for s in ids if s in keep]


# --------------------------------------------------------------------------- #
def probe_layers(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                 binary: bool, seed: int = RNG_SEED,
                 layers: list | None = None) -> dict:
    """Out-of-fold predictions for every layer, plus the fitted weight norm.

    Standardisation statistics come from the training folds only; using the whole
    matrix would leak the test distribution into the probe.

    `max_iter` is a CAP, not a fixed cost.  It was 300, which is enough for the
    separable blocks (severity converges in ~20 iterations) but is NOT enough
    for matched-vs-mismatch: measured on both families, 43-79% of those fits hit
    the cap at 300 and 0% do at 2000.  A curve drawn from mostly-unconverged
    fits is not a converged result, so the cap is 2000 everywhere and the search
    stop is left to lbfgs.  Blocks that converge early cost the same as before.

    The cap is uniform across every layer and every block on purpose: raising it
    only where a layer looks bad would be hand-tuning the probe to raise AUROC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings as _w
    n, rows, dim = X.shape
    oof = np.full((rows, n), np.nan)
    norms = np.full(rows, np.nan)
    want = set(range(rows)) if not layers else set(layers)
    for f in range(int(folds.max()) + 1):
        tr = folds != f
        te = ~tr
        if tr.sum() < 10 or te.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        # Per-LAYER standardisation.  X is (sample, layer, feature), so the
        # reduction axis is 0 alone.  Reducing over (0, 1) pools every layer
        # into one (1, 1, dim) statistic, which still broadcasts and still
        # fits, but then divides each layer by a shared scale -- which is the
        # same as giving every layer a different effective C.  The curve is
        # then comparable across layers only by accident.
        mu = X[tr].mean(axis=0, keepdims=True)
        sd = X[tr].std(axis=0, keepdims=True) + 1e-6
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        for L in range(rows):
            if L not in want:
                continue
            clf = LogisticRegression(max_iter=PROBE_MAX_ITER, tol=1e-3, C=1.0)
            try:
                with _w.catch_warnings(record=True) as ws:
                    _w.simplefilter("always")
                    clf.fit(Xtr[:, L, :], y[tr])
                if any(issubclass(x.category, ConvergenceWarning) for x in ws):
                    PROBE_UNCONVERGED.append({"where": "probe_layers",
                                              "fold": f, "layer": L})
            except Exception as exc:
                # Do not swallow this.  A try/except inside a layer loop that
                # only `continue`s will report a curve that stops early and
                # look like a converged result; that is how BUG 13 hid.
                PROBE_FAILURES.append({
                    "where": "probe_layers", "fold": f, "layer": L,
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_finite": int(np.isfinite(Xtr[:, L, :]).all(axis=1).sum()),
                    "n_train": int(tr.sum())})
                continue
            if clf.coef_.shape[0] == 1:
                oof[L, te] = 1.0 / (1.0 + np.exp(-clf.decision_function(
                    Xte[:, L, :])))
                norms[L] = float(np.linalg.norm(clf.coef_[0]))
            else:                                  # multiclass
                p = clf.predict_proba(Xte[:, L, :])
                oof[L, te] = p @ np.arange(p.shape[1])
                norms[L] = float(np.linalg.norm(clf.coef_, axis=1).mean())
    return {"oof": oof, "norms": norms}


def metrics_binary(oof: np.ndarray, y: np.ndarray, folds: np.ndarray,
                   n_boot: int = 1000, seed: int = RNG_SEED) -> list[dict]:
    from sklearn.metrics import roc_auc_score
    rows = oof.shape[0]
    out = []
    rng = np.random.default_rng(seed)
    uniq = np.unique(folds)
    idx = [np.where(folds == u)[0] for u in uniq]
    for L in range(rows):
        s = oof[L]
        ok = np.isfinite(s)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            out.append({"layer": L, "auroc": None, "acc": None, "ci_lo": None,
                        "ci_hi": None})
            continue
        so, yo = s[ok], y[ok]
        auc = float(roc_auc_score(yo, so))
        acc = float(((so > 0.5).astype(int) == yo).mean())
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            sel = np.concatenate([np.where(folds == uniq[p])[0] for p in pick])
            sel = sel[np.isfinite(oof[L][sel])]
            if len(sel) < 20 or len(np.unique(y[sel])) < 2:
                continue
            boots.append(roc_auc_score(y[sel], oof[L][sel]))
        out.append({"layer": L, "auroc": auc, "acc": acc,
                    "ci_lo": float(np.percentile(boots, 2.5)) if boots else None,
                    "ci_hi": float(np.percentile(boots, 97.5)) if boots else None})
    return out


def metrics_ordinal(oof: np.ndarray, y: np.ndarray) -> list[dict]:
    """Spearman correlation between the probe score and the severity ordinal."""
    from scipy.stats import spearmanr
    out = []
    for L in range(oof.shape[0]):
        s = oof[L]
        ok = np.isfinite(s)
        if ok.sum() < 20:
            out.append({"layer": L, "spearman": None, "acc": None})
            continue
        rho = float(spearmanr(s[ok], y[ok]).statistic)
        acc = float((np.round(s[ok]).astype(int) == y[ok]).mean())
        out.append({"layer": L, "spearman": rho, "acc": acc})
    return out


def probe_vision(vis: np.ndarray, y: np.ndarray, folds: np.ndarray,
                 n_boot: int = 1000, seed: int = RNG_SEED) -> dict:
    """One probe on the pooled vision representation -- the balanced control."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ok_rows = np.isfinite(vis).all(axis=1)
    oof = np.full(len(y), np.nan)
    norm = float("nan")
    for f in range(int(folds.max()) + 1):
        tr = (folds != f) & ok_rows
        te = (folds == f) & ok_rows
        if tr.sum() < 10 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        mu = vis[tr].mean(axis=0, keepdims=True)
        sd = vis[tr].std(axis=0, keepdims=True) + 1e-6
        clf = LogisticRegression(max_iter=2000, tol=1e-4, C=1.0)
        clf.fit((vis[tr] - mu) / sd, y[tr])
        oof[te] = 1.0 / (1.0 + np.exp(-clf.decision_function((vis[te] - mu) / sd)))
        norm = float(np.linalg.norm(clf.coef_[0]))
    ok = np.isfinite(oof)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return {"auroc": None, "acc": None, "ci_lo": None, "ci_hi": None,
                "weight_norm": norm, "n": int(ok.sum()),
                "n_vision_features_used": int(ok_rows.sum())}
    auc = float(roc_auc_score(y[ok], oof[ok]))
    acc = float(((oof[ok] > 0.5).astype(int) == y[ok]).mean())
    rng = np.random.default_rng(seed)
    uniq = np.unique(folds)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([np.where((folds == uniq[p]) & ok)[0] for p in pick])
        if len(sel) < 20 or len(np.unique(y[sel])) < 2:
            continue
        boots.append(roc_auc_score(y[sel], oof[sel]))
    return {"auroc": auc, "acc": acc,
            "ci_lo": float(np.percentile(boots, 2.5)) if boots else None,
            "ci_hi": float(np.percentile(boots, 97.5)) if boots else None,
            "weight_norm": norm, "n": int(ok.sum()),
            "n_vision_features_used": int(ok_rows.sum())}


# --------------------------------------------------------------------------- #
def align(feat: dict, arm_a: str, arm_b: str | None) -> tuple:
    """Common sample ids and stacked states for one or two arms."""
    if arm_b is None:
        a = feat[arm_a]
        return a["ids"], a["hidden"], a["vis"], [0] * len(a["ids"])
    a, b = feat[arm_a], feat[arm_b]
    common = sorted(set(a["ids"]) & set(b["ids"]))
    ia = {s: i for i, s in enumerate(a["ids"])}
    ib = {s: i for i, s in enumerate(b["ids"])}
    X = np.concatenate([a["hidden"][[ia[s] for s in common]],
                        b["hidden"][[ib[s] for s in common]]], axis=0)
    V = np.concatenate([a["vis"][[ia[s] for s in common]],
                        b["vis"][[ib[s] for s in common]]], axis=0)
    y = np.array([0] * len(common) + [1] * len(common))
    return common + common, X, V, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2vl,medvlmr1")
    ap.add_argument("--stage", default="main")
    ap.add_argument("--shuffle-control", action="store_true", default=True)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--fast", action="store_true",
                    help="subset scan for a first deliverable: donor k=1 only, "
                         "every third row")
    ap.add_argument("--donors", default="", help="comma list of donor k")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    donor_ks = ([int(x) for x in args.donors.split(",") if x] if args.donors
                else ([1] if args.fast else list(range(1, Q.N_DONORS + 1))))
    layer_stride = 3 if args.fast else 1
    out_dir = Q.ROOT / "RESULTS"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Q.ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in models:
        feat = load_features(model)
        # An arm whose file predates the vision hook is dropped rather than used
        # without its control: the control is the point of this script.
        stale = [a for a, f in feat.items() if not f.get("has_vision")]
        if stale:
            print(f"[{model}] dropping {len(stale)} arms without vision features "
                  f"(will be re-extracted): {stale}")
            for a in stale:
                feat.pop(a, None)
        print(f"[{model}] arms with features: {sorted(feat)}")
        if not feat:
            continue
        ref_ids = sorted(set(feat[next(iter(feat))]["ids"]))
        folds_map = folds_from_subgroups(ref_ids, k=1)

        def F(ids):
            m = {s: f for s, f in zip(ref_ids, folds_map)}
            return np.array([m.get(s, 0) for s in ids])

        def emit(task, layer, **kw):
            rows.append({"model": model, "task": task, "layer": layer, **kw})

        # ---------------- A. severity encoding (ordinal) ----------------
        # The four-class ordinal fit is by far the most expensive block in this
        # script (multinomial logistic on 3600x1536, five folds, every row).  In
        # --fast mode it is skipped: the informative severity statement is the
        # binary clean-vs-severe contrast below, and the ordinal curve is a
        # secondary refinement that can be added without invalidating anything.
        sev_arms = [Q.MATCHED_ARM[q] for q in Q.QUALITY_LEVELS]
        if all(a in feat for a in sev_arms) and not args.fast:
            ids = sorted(set.intersection(*[set(feat[a]["ids"]) for a in sev_arms]))
            idx = {a: {s: i for i, s in enumerate(feat[a]["ids"])} for a in sev_arms}
            X = np.concatenate([feat[a]["hidden"][[idx[a][s] for s in ids]]
                                for a in sev_arms], axis=0)
            V = np.concatenate([feat[a]["vis"][[idx[a][s] for s in ids]]
                                for a in sev_arms], axis=0)
            y = np.array([i for i in range(len(sev_arms))
                          for _ in range(len(ids))])
            f = F(ids * len(sev_arms))
            # The four-class ordinal fit is by far the most expensive block in this
            # script (multinomial logistic on 3600x1536, five folds, every row).  In
            # --fast mode it is skipped: the informative severity statement is the
            # binary clean-vs-severe contrast below, and the ordinal curve is a
            # secondary refinement that can be added without invalidating anything.
            res = probe_layers(X, y, f, binary=False,
                               layers=list(range(0, X.shape[1], layer_stride)))
            for m in metrics_ordinal(res["oof"], y):
                if m["spearman"] is None or not np.isfinite(m["spearman"]):
                    # row 0 is the input embedding of a text token: identical across
                    # the four severity arms, so the correlation is undefined, not zero.
                    emit("severity_ordinal", m["layer"], spearman=None,
                         acc=m["acc"], control="degenerate_row", kind="decision")
                    continue
                emit("severity_ordinal", m["layer"], spearman=m["spearman"],
                     acc=m["acc"], weight_norm=(None if not np.isfinite(
                         res["norms"][m["layer"]]) else float(res["norms"][m["layer"]])),
                     n=int(len(ids) * len(sev_arms)), n_features=X.shape[2],
                     control="none", kind="decision")
            # severe vs clean, binary, with a shuffled-label control
            ids2 = sorted(set(feat["clean"]["ids"]) & set(feat["blur_severe"]["ids"]))
            ids_, X_, V_, y_ = align({k: feat[k] for k in ("clean", "blur_severe")},
                                     "clean", "blur_severe")
            f2 = F(ids_)
            r2 = probe_layers(X_, y_, f2, binary=True,
                              layers=list(range(0, X_.shape[1], layer_stride)))
            for m in metrics_binary(r2["oof"], y_, f2):
                if m["auroc"] is None:
                    continue
                emit("severity_clean_vs_severe", m["layer"], auroc=m["auroc"],
                     acc=m["acc"], ci_lo=m["ci_lo"], ci_hi=m["ci_hi"],
                     weight_norm=float(r2["norms"][m["layer"]]),
                     n=len(y_), n_features=X_.shape[2], control="none",
                     kind="decision")
            v2 = probe_vision(V_, y_, f2)
            emit("severity_clean_vs_severe", -1, auroc=v2["auroc"], acc=v2["acc"],
                 ci_lo=v2["ci_lo"], ci_hi=v2["ci_hi"],
                 weight_norm=v2["weight_norm"], n=v2["n"], n_features=None,
                 control="vision_tower_pooled", kind="vision")
            if args.shuffle_control:
                # The shuffled-label null is established, not measured layer by
                # layer: running it on all 29 rows triples the cost of the whole
                # job to reproduce the same ~0.5 twenty nine times.
                rng = np.random.default_rng(RNG_SEED)
                ys = y_.copy(); rng.shuffle(ys)
                rs = probe_layers(X_, ys, f2, binary=True,
                                  layers=list(range(0, X_.shape[1], 3)))
                for m in metrics_binary(rs["oof"], ys, f2):
                    if m["auroc"] is None:
                        continue
                        emit("severity_clean_vs_severe", m["layer"], auroc=m["auroc"],
                             acc=m["acc"], ci_lo=m["ci_lo"], ci_hi=m["ci_hi"],
                             weight_norm=float(rs["norms"][m["layer"]]),
                             n=len(ys), n_features=X_.shape[2],
                             control="shuffled_labels", kind="decision")

        # ---------------- B. cross-corruption frozen transfer ----------------
        # Train once on clean-vs-BLURRED, freeze the probe, evaluate on
        # clean-vs-NOISE.  If it transfers, the representation carries something
        # more general than the low-level statistics of one corruption.
        if all(a in feat for a in ("clean", "blur_severe",
                                   "modality_specific_severe")):
            ids_tr, Xtr, Vtr, ytr = align(
                {k: feat[k] for k in ("clean", "blur_severe")},
                "clean", "blur_severe")
            ids_te, Xte, Vte, yte = align(
                {k: feat[k] for k in ("clean", "modality_specific_severe")},
                "clean", "modality_specific_severe")
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            f_tr = F(ids_tr)
            # per-layer again: axis 0 is the sample axis for a (n, layers, dim)
            # array.  Reducing over (0, 1) collapsed all 29 layers into one
            # statistic, which then indexed out of bounds at L >= 1 and, via the
            # bare `continue` below, deleted the whole transfer curve (BUG 13).
            mu = Xtr.mean(axis=0, keepdims=True)
            sd = Xtr.std(axis=0, keepdims=True) + 1e-6
            for L in range(Xtr.shape[1]):
                if args.max_layers and L > args.max_layers:
                    break
                if args.fast and L % 3:
                    continue
                clf = LogisticRegression(max_iter=1000, tol=1e-3, C=1.0)
                try:
                    clf.fit(((Xtr[:, L, :] - mu[:, L, :]) / sd[:, L, :]), ytr)
                except Exception as exc:
                    PROBE_FAILURES.append({
                        "where": "crosscorruption", "layer": L,
                        "error": f"{type(exc).__name__}: {exc}"})
                    continue
                s_te = clf.decision_function((Xte[:, L, :] - mu[:, L, :])
                                             / sd[:, L, :])
                auc = float(roc_auc_score(yte, s_te))
                emit("crosscorruption_blur_to_noise", L, auroc=auc,
                     acc=float(((s_te > 0).astype(int) == yte).mean()),
                     weight_norm=float(np.linalg.norm(clf.coef_[0])),
                     n_train=len(ytr), n_test=len(yte), n_features=Xtr.shape[2],
                     train_condition="clean_vs_blur_severe",
                     test_condition="clean_vs_modality_specific_severe",
                     control="none", kind="decision")
            # the same transfer on the pooled vision representation, for contrast
            clf = LogisticRegression(max_iter=2000, tol=1e-4, C=1.0)
            ok_tr = np.isfinite(Vtr).all(axis=1)
            ok_te = np.isfinite(Vte).all(axis=1)
            if ok_tr.sum() > 20 and ok_te.sum() > 20:
                muV = Vtr[ok_tr].mean(axis=0, keepdims=True)
                sdV = Vtr[ok_tr].std(axis=0, keepdims=True) + 1e-6
                clf.fit((Vtr[ok_tr] - muV) / sdV, ytr[ok_tr])
                s = clf.decision_function((Vte[ok_te] - muV) / sdV)
                emit("crosscorruption_blur_to_noise", -1,
                     auroc=float(roc_auc_score(yte[ok_te], s)),
                     acc=float(((s > 0).astype(int) == yte[ok_te]).mean()),
                     weight_norm=float(np.linalg.norm(clf.coef_[0])),
                     n_train=int(ok_tr.sum()), n_test=int(ok_te.sum()),
                     train_condition="clean_vs_blur_severe_via_vision_tower",
                     test_condition="clean_vs_modality_specific_severe",
                     control="vision_tower_pooled", kind="vision")

        # ---------------- C. mismatch encoding + the balanced control ----------------
            for q in Q.QUALITY_LEVELS:
                ma = Q.MATCHED_ARM[q]
                for k in donor_ks:
                    pa = Q.mismatch_arm(q, k)
                    if ma not in feat or pa not in feat:
                        continue
                    ids_, X_, V_, y_ = align(feat, ma, pa)
                    if len(y_) < 50:
                        continue
                    # Keep only items in complete closed subgroups; see the note on
                    # folds_from_subgroups for why a partial subgroup silently
                    # breaks the balanced control.
                    keep = set(complete_subgroups(ids_[:len(ids_) // 2], k=k))
                    mask = np.array([s in keep for s in ids_])
                    if mask.sum() < 50:
                        continue
                    ids_, X_, V_, y_ = ( [s for s, m in zip(ids_, mask) if m],
                                        X_[mask], V_[mask], y_[mask] )
                    n_half = len(y_) // 2
                    f_ = folds_from_subgroups(ids_[:n_half], k=k)
                    fmap = {s: f for s, f in zip(ids_[:n_half], f_)}
                    f_all = np.array([fmap[s] for s in ids_])
                    scan = list(range(0, X_.shape[1], layer_stride))
                    r = probe_layers(X_, y_, f_all, binary=True, layers=scan)
                    for m in metrics_binary(r["oof"], y_, f_all):
                        if m["auroc"] is None:
                            continue
                        emit(f"mismatch_{q}_k{k}", m["layer"], auroc=m["auroc"],
                             acc=m["acc"], ci_lo=m["ci_lo"], ci_hi=m["ci_hi"],
                             weight_norm=float(r["norms"][m["layer"]]),
                             n=len(y_), n_features=X_.shape[2], control="none",
                             kind="decision")
                    # THE CONTROL: pooled vision tower only.  Both classes show the
                    # same image multiset on every fold, so this must sit at chance.
                    v = probe_vision(V_, y_, f_all)
                    emit(f"mismatch_{q}_k{k}", -1, auroc=v["auroc"], acc=v["acc"],
                         ci_lo=v["ci_lo"], ci_hi=v["ci_hi"],
                         weight_norm=v["weight_norm"], n=v["n"], n_features=None,
                         control="vis_only_balanced", kind="vision")
                    if args.shuffle_control:
                        rng = np.random.default_rng(RNG_SEED)
                        ys = y_.copy(); rng.shuffle(ys)
                        rs = probe_layers(X_, ys, f_all, binary=True, layers=scan)
                        for m in metrics_binary(rs["oof"], ys, f_all):
                            if m["auroc"] is None:
                                continue
                            emit(f"mismatch_{q}_k{k}", m["layer"], auroc=m["auroc"],
                                 acc=m["acc"], ci_lo=m["ci_lo"], ci_hi=m["ci_hi"],
                                 weight_norm=float(rs["norms"][m["layer"]]),
                                 n=len(ys), n_features=X_.shape[2],
                                 control="shuffled_labels", kind="decision")

    if not rows:
        print("no probe results produced -- features missing?")
        return
    keys = sorted({k for r in rows for k in r})
    name = ("LEVEL1_PROBE_RESULTS.csv" if args.stage == "main"
            else f"LEVEL1_PROBE_RESULTS_{args.stage}.csv")
    with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", name, len(rows), "rows")

    # ---------------- the control verdict, stated plainly ----------------
    ctrl = [r for r in rows if r.get("control") == "vis_only_balanced"]
    lines = ["# Level 1 probing results", "",
             "Decodable is not integrated, and integrated is not used.  These",
             "numbers say information is linearly readable from a state; they say",
             "nothing about whether the answer depends on it.", "",
             "## Balanced-control verdict", "",
             "The matched and mismatch classes share one image multiset on every",
             "training fold, so a probe given only the pooled vision-tower state",
             "must read chance.  Measured:", ""]
    if ctrl:
        lines += ["| model | task | AUROC | 95% CI | acc | learned ||w|| | n |",
                  "|---|---|---|---|---|---|---|"]
        bad = 0
        for c in ctrl:
            ci = (f"[{c['ci_lo']:.3f},{c['ci_hi']:.3f}]"
                  if c.get("ci_lo") is not None else "n/a")
            w_ = (f"{c['weight_norm']:.3f}" if c.get("weight_norm") is not None
                  else "n/a")
            lines.append(f"| {c['model']} | {c['task']} | "
                         f"{0 if c.get('auroc') is None else round(c['auroc'], 4)} | "
                         f"{ci} | {0 if c.get('acc') is None else round(c['acc'], 4)} | "
                         f"{w_} | {c.get('n')} |")
            if c.get("auroc") is not None and abs(c["auroc"] - 0.5) > 0.05:
                bad += 1
        lines += ["", (f"**{bad} of {len(ctrl)} balanced controls fall more than "
                       f"0.05 from chance.**"
                       if bad else
                       f"All {len(ctrl)} balanced controls sit within 0.05 of "
                       f"chance, which is what the closed-subgroup construction "
                       f"guarantees.")]
    else:
        lines.append("_No balanced control was produced (features missing)._")
    lines += ["", "## Probe results", "",
              f"Full table: `RESULTS/{name}`.", ""]
    (rep_dir / "LEVEL1_PROBING_RESULTS.md").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
    print("wrote reports/LEVEL1_PROBING_RESULTS.md")
    Q.write_json(Q.ROOT / "runtime" / f"q4_{args.stage}.json",
                 {"n_rows": len(rows), "n_controls": len(ctrl),
                  "n_off_chance_controls": sum(
                      1 for c in ctrl if c.get("auroc") is not None
                      and abs(c["auroc"] - 0.5) > 0.05), "time": Q.now()})
    # failures are a result, not a log line: ship them with the numbers
    Q.write_json(Q.ROOT / "runtime" / f"q4_failures_{args.stage}.json",
                 {"n_failures": len(PROBE_FAILURES),
                  "n_unconverged": len(PROBE_UNCONVERGED),
                  "max_iter": PROBE_MAX_ITER,
                  "failures": PROBE_FAILURES[:400],
                  "unconverged_sample": PROBE_UNCONVERGED[:200]})
    print(f"probe config: max_iter={PROBE_MAX_ITER}, "
          f"{len(PROBE_UNCONVERGED)} fits hit the cap")
    if PROBE_FAILURES:
        print(f"!! {len(PROBE_FAILURES)} probe fits raised and were skipped; "
              f"see runtime/q4_failures_{args.stage}.json")
    else:
        print("no probe fit raised")


if __name__ == "__main__":
    main()
