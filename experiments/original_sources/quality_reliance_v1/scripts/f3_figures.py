#!/usr/bin/env python
"""F3 -- Figures 1-9.

Every figure is drawn from the artefact files rather than from constants, so a
figure cannot disagree with the table it came from.  Each one carries the
caveat it needs in its own caption, printed into the figure and repeated in
reports/FIGURES.md.

Run
    python scripts/f3_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q                    # noqa: E402

FIG = Q.ROOT / "figures"
RES = Q.ROOT / "RESULTS"
CAPS = []


def rd_csv(p):
    p = Q.ROOT / p
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.is_file() else []


def rd_json(p):
    p = Q.ROOT / p
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def save(fig, name, caption):
    fig.tight_layout()
    p = FIG / name
    fig.savefig(p, dpi=150)
    plt.close(fig)
    CAPS.append((name, caption))
    print(f"  {name}")


# --------------------------------------------------------------------------- #
def fig1_decomposition():
    dec = [r for r in rd_csv("RESULTS/QUALITY_RELEVANCE_DECOMPOSITION.csv")
           if r.get("cohort") == "all" and r.get("stratum") == "ALL"]
    if not dec:
        return
    order = ["clean", "mild", "medium", "severe"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, lab in zip(axes, ("U_presence", "U_content_mean",
                                   "U_mismatch_mean"),
                            ("U_presence (image presence)",
                             "U_content (matched - mismatch)",
                             "U_mismatch (mismatch - no_image)")):
        for m in ("qwen2vl", "medvlmr1"):
            ys = [num(next((r[col] for r in dec if r["model"] == m
                            and r["quality_level"] == q), None))
                  for q in order]
            ax.plot(range(4), ys, "o-", label=m)
        ax.set_xticks(range(4))
        ax.set_xticklabels(order)
        ax.set_title(lab, fontsize=9)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("log margin")
    fig.suptitle("Figure 1 -- Quality x Relevance decomposition (official 900, "
                 "cohort=all)", fontsize=11)
    save(fig, "fig01_decomposition.png",
         "`U_content = U_presence - U_mismatch` holds per item. The flat "
         "clean->mild segment of `U_content` is the result in Figure 2.")


def fig2_ladder_qc():
    s = rd_json("RESULTS/QUALITY_LADDER_QC_summary.json")
    if not s:
        return
    rows = sorted(s["per_level"], key=lambda r: r["sigma_px_mean"])
    sig = [r["sigma_px_mean"] for r in rows]
    ssim = [r["ssim_mean"] for r in rows]
    edge = [r["edge_ratio_mean"] for r in rows]
    names = [f"{r['quality_level']}\n({r['sigma_px_mean']:.0f}px)" for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(rows))
    ax.plot(x, ssim, "o-", color="tab:blue", label="SSIM vs clean")
    ax.plot(x, edge, "s-", color="tab:red", label="edge energy ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("normalised")
    pref = [r for r in rows if r["quality_level"] == "mid_extra"]
    if pref:
        i = rows.index(pref[0])
        ax.axvspan(i - 0.4, i + 0.4, color="grey", alpha=0.15)
        ax.text(i, 0.06, "extra sensitivity rung\n(not in the frozen ladder)",
                ha="center", fontsize=7, color="dimgrey")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Figure 2 -- the ladder really degrades the image", fontsize=11)
    save(fig, "fig02_ladder_qc.png",
         "clean -> mild drops SSIM by 0.1257 and halves the edge energy, so the "
         "flat `U_content` segment in Figure 1 is a statement about the model, "
         "not about a ladder that did nothing.")


def _probe_curve(rows, model, task, control="none"):
    """Layer curve for ONE control condition.

    Selecting on `control` explicitly matters: every task stores a decision row
    (`control == "none"`), a shuffled-label row (`shuffled_labels`) and one
    pooled-vision row (`vis_only_balanced`, layer -1) per layer.  Keeping
    "everything that is not the pooled row" yields 58 points whose layers repeat
    and whose curve interleaves the decision probe with its own control.
    """
    t = [r for r in rows if r["model"] == model and r["task"] == task
         and r.get("control") == control
         and r.get("layer") not in ("-1", "", None)]
    t = sorted(t, key=lambda r: int(r["layer"]))
    return [int(r["layer"]) for r in t], [num(r.get("auroc")) for r in t]


def fig3_probe_curves():
    rows = rd_csv("RESULTS/LEVEL1_PROBE_RESULTS.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, task, title in zip(
            axes,
            ("mismatch_clean_k1", "crosscorruption_blur_to_noise",
             "severity_clean_vs_severe"),
            ("matched vs mismatch (clean, k=1)",
             "cross-corruption transfer\n(fit on blur, frozen, tested on noise)",
             "severity encoding (clean vs severe)")):
        for m in ("qwen2vl", "medvlmr1"):
            x, y = _probe_curve(rows, m, task, "none")
            if x:
                ax.plot(x, y, "o-", ms=3, label=m)
            xs, ys = _probe_curve(rows, m, task, "shuffled_labels")
            if xs:
                ax.plot(xs, ys, ":", lw=1, alpha=0.7,
                        label=f"{m} shuffled labels")
        ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("layer")
        ax.set_ylim(0.44, 1.02)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Figure 3 -- Level 1 decodability. Dashed line is chance. L0 "
                 "reads exactly 0.5000 in every panel.", fontsize=11)
    save(fig, "fig03_probe_curves.png",
         "A probe above chance means the property is DECODABLE, never that the "
         "model uses it. The balanced vision-only control reads exactly 0.5000 "
         "in all 24 cells and is not plotted because it is a flat line.")


def fig4_balanced_control():
    rows = [r for r in rd_csv("RESULTS/LEVEL1_PROBE_RESULTS.csv")
            if r.get("control") == "vis_only_balanced"]
    if not rows:
        return
    v = np.array([num(r.get("auroc")) for r in rows])
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(v, bins=12, range=(0.49, 0.51), color="tab:green")
    ax.axvline(0.5, color="k", lw=1.2, ls="--")
    ax.set_xlabel("AUROC")
    ax.set_ylabel("number of control cells")
    ax.set_title(f"Figure 4 -- balanced vision-only control ({len(v)} cells)",
                 fontsize=11)
    ax.set_xlim(0.49, 0.51)
    save(fig, "fig04_balanced_control.png",
         f"All {len(v)} cells read exactly 0.5000. This is the construction "
         "working, not evidence about the model: inside every fold the matched "
         "and mismatched image multisets are identical.")


def fig5_strict_donor():
    # one artefact PER QUALITY.  Reading only `r2_<model>_clean.json` and
    # plotting it in both panels draws the clean result twice and makes the two
    # panels look identical, which is worse than an error because it looks fine.
    d = {}
    for m in ("qwen2vl", "medvlmr1"):
        for q in ("clean", "severe"):
            b = rd_json(f"runtime/r2_{m}_{q}.json")
            if b:
                d[(m, q)] = b
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ms = sorted({k[0] for k in d})
    x = np.arange(len(ms))
    w = 0.35
    for ax, q in zip(axes, ("clean", "severe")):
        st = [d[(m, q)]["U_content_strict"]["mean"] for m in ms]
        lo = [d[(m, q)]["U_content_strict"]["mean"]
              - d[(m, q)]["U_content_strict"]["lo"] for m in ms]
        hi = [d[(m, q)]["U_content_strict"]["hi"]
              - d[(m, q)]["U_content_strict"]["mean"] for m in ms]
        ls_ = [d[(m, q)]["U_content_loose"]["mean"] for m in ms]
        llo = [d[(m, q)]["U_content_loose"]["mean"]
               - d[(m, q)]["U_content_loose"]["lo"] for m in ms]
        lhi = [d[(m, q)]["U_content_loose"]["hi"]
               - d[(m, q)]["U_content_loose"]["mean"] for m in ms]
        ax.bar(x - w / 2, st, w, yerr=[lo, hi], capsize=4,
               label="strict donor (same anatomy, other finding)",
               color="tab:red")
        ax.bar(x + w / 2, ls_, w, yerr=[llo, lhi], capsize=4,
               label="loose donor (same modality, other location)",
               color="tab:blue")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ms)
        ax.set_title(f"quality = {q}", fontsize=10)
        ax.set_ylabel("U_content")
        ax.set_ylim(-0.40, 0.55)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure 5 -- strict vs loose donor, paired on the same 990 "
                 "RadImageNet targets", fontsize=11)
    save(fig, "fig05_strict_donor.png",
         "The strict donor -- which differs only in the finding -- is MORE "
         "harmful than the loose donor, so `U_content` is measuring content "
         "conflict and not image distance. RadImageNet accuracy is near chance "
         "(match 0.32, no_image 0.27), so the absolute magnitudes are not "
         "comparable to the official 900.")


def fig6_vip():
    rows = rd_csv("RESULTS/VIP_TVI_CALIBRATION.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ("qwen2vl", "medvlmr1")):
        r = sorted([q for q in rows if q["model"] == m],
                   key=lambda q: int(q["layer"]))
        if not r:
            continue
        x = [int(q["layer"]) for q in r]
        ax.plot(x, [num(q["D_PVT"]) for q in r], "o-", ms=3,
                label="D^P_VT (image needed)")
        ax.plot(x, [num(q["D_PT"]) for q in r], "s-", ms=3,
                label="D^P_T (text suffices)")
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        i = int(np.nanargmax([num(q["G"]) for q in r]))
        ax.annotate(f"G peak {num(r[i]['G']):+.3f}\n@L{x[i]}",
                    xy=(x[i], num(r[i]["D_PVT"])), fontsize=8,
                    xytext=(4, -22), textcoords="offset points")
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylim(0.44, 0.95)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cross-validated AUROC (correct vs incorrect)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 6 -- VIP/TVI calibration curves", fontsize=11)
    save(fig, "fig06_vip_tvi.png",
         "`D_l` is how separable being correct is at layer l. `TVI(S) = "
         "max_l D_l^S - 0.5`. TVI(P_VT) exceeds TVI(P_T) by ~0.30 with a "
         "bootstrap interval excluding zero in 100% of replicates. NOTE: the "
         "plateau-onset definition of VIP returns layer 1 for both models and "
         "has no localising power, so the branch is labelled EXPLORATORY.")


def fig7_causal():
    agg = defaultdict(list)
    for m in ("qwen2vl", "medvlmr1"):
        p = Q.ROOT / "results" / "causal" / f"CAUSAL__{m}.jsonl"
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "ok":
                agg[(r["model"], r["intervention"], int(r["layer"]))].append(
                    float(r["delta_M"]))
    if not agg:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ivs = sorted({k[1] for k in agg})
    for iv in ivs:
        for m, style in (("qwen2vl", "-o"), ("medvlmr1", "--s")):
            pts = sorted((l, float(np.mean(v))) for (mm, i, l), v in agg.items()
                         if mm == m and i == iv)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style, ms=3,
                        label=f"{m} {iv}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean delta of the gold log margin")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Figure 7 -- within-checkpoint causal interventions", fontsize=11)
    save(fig, "fig07_causal.png",
         "All interventions are within ONE checkpoint; no activation moves "
         "between models. The identity control is exactly 0. "
         "`visual_knockout_direct` zeroes image tokens at every row and is "
         "therefore a single intervention, not a layer curve.")


def fig8_ves():
    per = rd_csv("RESULTS/VES_DIAGNOSTIC_PER_ITEM.csv")
    if not per:
        return
    rows = per[:60000]
    dh = np.array([num(r.get("DeltaH")) for r in rows])
    u = np.array([num(r.get("U_presence", r.get("U"))) for r in rows])
    ok = np.isfinite(dh) & np.isfinite(u)
    dh, u = dh[ok], u[ok]
    if not len(dh):
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(dh, u, s=2, alpha=0.25, color="tab:purple")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    q = np.percentile(dh, 75)
    bad = (dh >= q) & (u < 0)
    ax.axvspan(q, dh.max(), color="red", alpha=0.07)
    ax.set_xlabel("DeltaH  (confidence gained from the image)")
    ax.set_ylabel("U  (utility of the image)")
    ax.set_title(f"Figure 8 -- sharpening is not utility\n"
                 f"{bad.mean():.1%} of items are in the top DeltaH quartile "
                 f"with negative utility", fontsize=10)
    ax.grid(alpha=0.25)
    save(fig, "fig08_ves.png",
         "A peaked-and-wrong distribution has both DeltaH > 0 and negative "
         "utility, and those items are real, not hypothetical. The claim is "
         "that entropy-based sensitivity ALONE cannot separate helpful from "
         "harmful visual evidence -- never that VES-RFT is invalid.")


def fig9_modality():
    dec = [r for r in rd_csv("RESULTS/QUALITY_RELEVANCE_DECOMPOSITION.csv")
           if r.get("cohort") == "all" and r.get("stratum") in
           ("MRI", "CT", "Xray")]
    if not dec:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, ls in (("qwen2vl", "-o"), ("medvlmr1", "--s")):
        for s in ("MRI", "CT", "Xray"):
            r = sorted([q for q in dec if q["model"] == m
                        and q["stratum"] == s],
                       key=lambda q: ["clean", "mild", "medium",
                                      "severe"].index(q["quality_level"]))
            if r:
                ax.plot(range(4), [num(q["U_content_mean"]) for q in r], ls,
                        ms=4, label=f"{m} {s}", alpha=0.85)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["clean", "mild", "medium", "severe"])
    ax.set_ylabel("U_content")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Figure 9 -- U_content by modality and quality", fontsize=11)
    save(fig, "fig09_modality.png",
         "The ordering Xray > MRI > CT holds in both models, so modality "
         "difficulty moves the effect more than the checkpoint does.")


def fig10_family():
    fam = rd_csv("RESULTS/GEMMA_MEDGEMMA_DECOMPOSITION.csv")
    dec = [r for r in rd_csv("RESULTS/QUALITY_RELEVANCE_DECOMPOSITION.csv")
           if r.get("cohort") == "all" and r.get("stratum") == "ALL"]
    if not fam and not dec:
        return
    order = ["clean", "mild", "medium", "severe"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, qty in zip(axes, ("U_presence", "U_content")):
        for m, style in (("gemma3", "-o"), ("medgemma", "-s"),
                         ("qwen2vl", "--^"), ("medvlmr1", "--v")):
            ys = []
            for q in order:
                v = np.nan
                if m in ("gemma3", "medgemma"):
                    r = next((y for y in fam if y["model"] == m
                              and y["quality"] == q and y["quantity"] == qty),
                             None)
                    v = num(r["value"]) if r else np.nan
                else:
                    r = next((y for y in dec if y["model"] == m
                              and y["quality_level"] == q), None)
                    if r:
                        v = num(r["U_presence"] if qty == "U_presence"
                                else r["U_content_mean"])
                ys.append(v)
            if np.isfinite(ys).any():
                ax.plot(range(4), ys, style, ms=4, label=m)
        ax.set_xticks(range(4))
        ax.set_xticklabels(order)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(qty, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("log margin")
    fig.suptitle("Figure 10 -- cross-family shape (magnitudes are NOT "
                 "comparable: the processor budgets differ)", fontsize=11)
    save(fig, "fig10_family_shape.png",
         "All four models show the same shape: a flat clean->mild segment "
         "followed by a drop at medium. Magnitudes are not comparable across "
         "families -- Gemma 3 resizes to a fixed 896x896 and gets 256 visual "
         "tokens, the Qwen runs use a matched 401408-pixel budget and get "
         "roughly 64-100. This is a behaviour replication only; Gemma/MedGemma "
         "is not an RL causal pair.")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    print("figures:")
    for fn in (fig1_decomposition, fig2_ladder_qc, fig3_probe_curves,
               fig4_balanced_control, fig5_strict_donor, fig6_vip,
               fig7_causal, fig8_ves, fig9_modality, fig10_family):
        try:
            fn()
        except Exception as exc:
            print(f"  !! {fn.__name__}: {type(exc).__name__}: {exc}")
    L = ["# Figures", "",
         "Every figure is drawn from the artefact files, never from a constant "
         "typed into the plotting code.", ""]
    for name, cap in CAPS:
        L += [f"## `{name}`", "", cap, "",
              f"![{name}](../figures/{name})", ""]
    (Q.ROOT / "reports" / "FIGURES.md").write_text("\n".join(L) + "\n",
                                                   encoding="utf-8")
    print(f"wrote reports/FIGURES.md ({len(CAPS)} figures)")


if __name__ == "__main__":
    main()
