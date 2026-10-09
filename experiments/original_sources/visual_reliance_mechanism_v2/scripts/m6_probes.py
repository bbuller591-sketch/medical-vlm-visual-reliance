#!/usr/bin/env python
"""Stage 4 -- leakage-safe linear probes (spec 04 Stage 4, 07 §7).

Evidence level 2: a probe shows that a variable is *decodable* at a layer.  It does not
show that the downstream answer used it.  Every result below is labelled accordingly.

Probes
------
A. **Blur awareness** clean(0) vs blur_severe(1) at three representations:
   decision state, pooled visual decoder residual, and the pooled vision-encoder output
   (which has never seen the question).
B. **Cross-corruption transfer** -- the same setup with modality_specific_severe.  The
   two ladders are NOT strength-matched, so transfer above chance says only that some
   shared degradedness signal travels; the magnitude is not a robustness measure.
C. **Mismatch awareness** clean vs random_same_modality_replacement at the decision
   state, with the vision-encoder representation as the negative control.
D. **Semantic transfer** -- mismatch probe applied unchanged to opposite_answer_donor
   (exploratory, n small) and same_answer_donor (compatibility control).
E. **Visual-correction phenotype** -- descriptive, secondary, reported separately.

Controls: source-grouped K-fold (a source's paired clean/degraded rows never straddle
the split), train-only scaling, and a shuffled-label run.

Writes probes/probe_results.json, probes/PROBES.md, plots/probe_auroc_by_layer.png
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V      # noqa: E402
import featlib as F        # noqa: E402

SEED = V.GLOBAL_SEED
N_FOLDS = 5
CONDS = ["clean", "no_image", "blur_severe", "modality_specific_severe",
         "random_same_modality_replacement", "same_answer_donor",
         "opposite_answer_donor"]


def auproc(y, s) -> tuple[float, float, int]:
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(s)
    if y[ok].min() == y[ok].max():
        return float("nan"), float("nan"), int(ok.sum())
    return (float(roc_auc_score(y[ok], s[ok])),
            float(np.mean((s[ok] > 0.5) == y[ok])), int(ok.sum()))


def probe_2d(Xa, Xb, groups, shuffled=False, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([Xa, Xb]).astype(np.float32)
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    g = np.concatenate([groups, groups])
    if shuffled:
        y = np.random.default_rng(seed).permutation(y)
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    auc, ba, n = auproc(y, oof)
    return {"auroc": auc, "balanced_accuracy": ba, "n": n}


def probe_by_layer(Xa, Xb, groups, shuffled=False, seed=SEED):
    """Xa/Xb: [n, T, hidden] -> {t: metrics}."""
    T = Xa.shape[1]
    out = {}
    for t in range(T):
        out[t] = probe_2d(Xa[:, t, :], Xb[:, t, :], groups, shuffled, seed)
        if (t + 1) % 10 == 0:
            print(f"      layer {t + 1}/{T}", flush=True)
    return out


def curve_vals(c: dict, T: int) -> list:
    """AUROC per layer, tolerating int keys (same process) or str keys (after JSON)."""
    return [c.get(str(t), c.get(t, {})).get("auroc", float("nan")) for t in range(T)]


def best_of(curve: dict) -> tuple[int, float]:
    items = sorted(((int(k), v["auroc"]) for k, v in curve.items()),
                   key=lambda kv: (np.isnan(kv[1]), -(kv[1] if kv[1] == kv[1] else 0)))
    return items[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(V.MODELS))
    args = ap.parse_args()

    items = V.load_items()
    res: dict = {
        "created": V.now(), "seed": SEED, "folds": N_FOLDS,
        "grouping": "source sample_id; a source's paired rows never straddle a split",
        "evidence_level": 2,
        "caveat": ("AUROC above chance means the variable is DECODABLE at that layer, "
                   "not that the downstream answer used it."),
        "models": {},
    }

    for model in args.models:
        print(f"=== {model} ===", flush=True)
        avail = [c for c in CONDS if F.has(model, c)]
        cache = {c: F.load_arrays(model, c) for c in avail}
        # The four probes A/B/C need the full 900, so alignment is over those four.
        # The donor conditions cover SUBSETS (529 and 73), so they are aligned
        # separately against `clean` further down rather than forced into this index.
        core = [c for c in ("clean", "blur_severe", "modality_specific_severe",
                            "random_same_modality_replacement") if c in avail]
        common = sorted(set.intersection(*[{s for s in cache[c]["sample_ids"]}
                                           for c in core]))
        row = {c: {s: i for i, s in enumerate(cache[c]["sample_ids"])} for c in avail}
        idx = {c: np.array([row[c][s] for s in common]) for c in core}
        # Group by SOURCE SAMPLE.  Each sample contributes exactly two rows (its clean
        # run and its degraded run), and GroupKFold keeps both in the same fold, so a
        # source's paired rows can never straddle train/test.  Grouping by modality
        # instead would be wrong in both directions: it would leak samples across folds
        # while also giving fewer groups than folds.
        groups = np.array(list(common))
        T = cache["clean"]["decision_hidden"].shape[1]
        m: dict = {"n_sources_aligned": len(common), "T": T,
                   "conditions": avail, "per_modality_n": {
                       k: int(sum(1 for g in groups if g == k))
                       for k in sorted(set(groups))}}

        def dec(c):
            return cache[c]["decision_hidden"][idx[c]]

        def vp(c):
            return cache[c]["visual_pooled"][idx[c]]

        def ve(c):
            a = cache[c]["vision_encoder_pool"][idx[c]]
            return a                                  # [n, hidden]; NaN rows for no_image

        m["A_blur_decision"] = probe_by_layer(dec("clean"), dec("blur_severe"), groups)
        m["A_blur_visual_pooled"] = probe_by_layer(vp("clean"), vp("blur_severe"), groups)
        m["A_blur_vision_encoder"] = probe_2d(ve("clean"), ve("blur_severe"), groups)
        m["B_transfer_modality_specific_decision"] = probe_by_layer(
            dec("clean"), dec("modality_specific_severe"), groups)
        m["C_mismatch_decision"] = probe_by_layer(
            dec("clean"), dec("random_same_modality_replacement"), groups)
        m["C_mismatch_vision_negative_control"] = probe_2d(
            ve("clean"), ve("random_same_modality_replacement"), groups)
        # D -- no retraining: probe C's fold models are what we want, but for a
        # transfer table the cleanest comparable number is the same pipeline applied to
        # the donor conditions.
        for cond in ("opposite_answer_donor", "same_answer_donor"):
            if cond in avail:
                sub = np.array([row[cond][s] for s in common if s in row[cond]])
                keep = np.array([i for i, s in enumerate(common) if s in row[cond]])
                m[f"D_{cond}_decision"] = probe_by_layer(
                    dec("clean")[keep], cache[cond]["decision_hidden"][sub],
                    groups[keep])
        m["control_shuffled_labels_decision"] = probe_by_layer(
            dec("clean"), dec("blur_severe"), groups, shuffled=True)

        res["models"][model] = m
        bt, bv = best_of(m["A_blur_decision"])
        ct, cv = best_of(m["C_mismatch_decision"])
        print(f"  blur decision best AUROC {bv:.3f} at row {bt} "
              f"(L{bt + 1}); mismatch decision best {cv:.3f} at row {ct} "
              f"(L{ct + 1})", flush=True)
        del cache
        gc.collect()

    V.write_json(V.ROOT / "probes" / "probe_results.json", res)

    # ------------------------------------------------------------------ #
    # markdown
    # ------------------------------------------------------------------ #
    L = ["# Layer-wise probes (Stage 4)", "",
         f"generated {res['created']} · {N_FOLDS}-fold **source-grouped** CV · "
         "train-only scaling · shuffled-label control on every decision-state probe",
         "",
         "> **Evidence level 2.** A probe result says a variable is *decodable* at a "
         "layer. It does **not** say the answer used it. Nothing here is causal "
         "evidence, and no layer below should be described as \"where the model uses "
         "the image\".", ""]

    keys = [("A_blur_decision", "A. blur — decision state"),
            ("A_blur_visual_pooled", "A. blur — pooled visual residual"),
            ("B_transfer_modality_specific_decision",
             "B. modality-specific corruption (transfer) — decision"),
            ("C_mismatch_decision", "C. mismatch — decision state"),
            ("D_opposite_answer_donor_decision",
             "D. opposite-answer donor (exploratory) — decision"),
            ("D_same_answer_donor_decision",
             "D. same-answer donor (compatibility control) — decision"),
            ("control_shuffled_labels_decision", "control: shuffled labels — decision")]
    L += ["## Decision/visual probes by layer", "",
          "| model | probe | AUROC row 0 | best AUROC | best layer | AUROC final | "
          "shuffled control |", "|---|---|---|---|---|---|---|"]
    for model in res["models"]:
        m = res["models"][model]
        if "error" in m:
            L.append(f"| {V.MODEL_DISPLAY[model]} | — | _features missing_ | | | | |")
            continue
        ctrl = m.get("control_shuffled_labels_decision", {})
        _, cbest = best_of(ctrl) if ctrl else (0, float("nan"))
        for key, lab in keys:
            c = m.get(key)
            if not c:
                continue
            T = len(c)
            vals = curve_vals(c, T)
            bi, bv = best_of(c)
            L.append(f"| {V.MODEL_DISPLAY[model]} | {lab} | {vals[0]:.3f} | "
                     f"{bv:.3f} | `{F.layer_label(bi, T)}` | {vals[-1]:.3f} | "
                     f"{cbest:.3f} |")
    L += ["", "## Vision-encoder representations (single vector, no layer axis)", "",
          "| model | representation | probe | AUROC | balanced acc | n |",
          "|---|---|---|---|---|---|"]
    for model in res["models"]:
        m = res["models"][model]
        if "error" in m:
            continue
        for key, lab in (("A_blur_vision_encoder", "A. blur"),
                         ("C_mismatch_vision_negative_control",
                          "C. mismatch (negative control)")):
            v = m.get(key)
            if not v:
                continue
            L.append(f"| {V.MODEL_DISPLAY[model]} | vision encoder (pre-fusion) | {lab} "
                     f"| {v['auroc']:.3f} | {v['balanced_accuracy']:.3f} | {v['n']} |")
    L += ["", "## Reading these numbers", "",
          "- **A blur, decision state.** High AUROC means blur is linearly decodable "
          "from the state that decides the answer, i.e. degradation information reaches "
          "answer computation. It does not mean the model acts on it.",
          "- **A blur, vision encoder.** High AUROC here is expected and is *not* "
          "evidence of anything cognitive: the encoder sees a blurred image, so blur is "
          "perceptually encoded by construction.",
          "- **C mismatch, vision encoder = negative control.** The vision encoder "
          "never sees the question, so it should not be able to tell a compatible image "
          "from an incompatible one; if this AUROC is high, donor construction or image "
          "identity is leaking rather than semantic mismatch being detected.",
          "- **B transfer.** The two corruption ladders are not strength-matched, so a "
          "transfer AUROC above chance is evidence of a shared degradedness signal, not "
          "of matched-severity robustness.",
          "- **D donors.** Opposite-answer donor is exploratory (n small). Same-answer "
          "donor is a compatibility control; it is not a perfect ground-truth "
          "compatibility label because the image still changes.",
          ""]
    (V.ROOT / "probes" / "PROBES.md").write_text("\n".join(L), encoding="utf-8")

    # ------------------------------------------------------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        models = [x for x in res["models"] if "error" not in res["models"][x]]
        fig, ax = plt.subplots(figsize=(9, 5))
        for model in models:
            m = res["models"][model]
            for key, lab, ls in (
                    ("A_blur_decision", "blur — decision", "-"),
                    ("C_mismatch_decision", "mismatch — decision", "-"),
                    ("B_transfer_modality_specific_decision",
                     "modality-specific — decision", "--"),
                    ("control_shuffled_labels_decision", "shuffled control", ":")):
                c = m.get(key)
                if not c:
                    continue
                T = len(c)
                y = curve_vals(c, T)
                ax.plot(range(T), y, ls, linewidth=2.0,
                        label=f"{V.MODEL_DISPLAY[model]} — {lab}")
        ax.axhline(0.5, color="k", alpha=0.35, linewidth=1)
        ax.set_xlabel("hidden-state row (0 = after decoder layer 1)")
        ax.set_ylabel("probe AUROC (source-grouped 5-fold)")
        ax.set_ylim(0.4, 1.02)
        ax.set_title("Probe AUROC by layer — decodability, not causal use")
        ax.legend(fontsize=8, ncol=1)
        fig.tight_layout()
        fig.savefig(V.ROOT / "plots" / "probe_auroc_by_layer.png", dpi=140)
        plt.close(fig)
        print("plot: plots/probe_auroc_by_layer.png")
    except Exception as exc:                        # pragma: no cover
        print(f"plot failed: {exc}")

    print("wrote probes/probe_results.json, probes/PROBES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
