#!/usr/bin/env python
"""Stage 5 -- visual integration (VIP/TVI-like) and the logit-lens margin trajectory.

Spec 02 sections C and D, 04 Stage 5, 08 plots 5 and 6.

Two descriptive measurements, both CPU-only:

  1. **Visual integration distance.**  For paired clean / genuine-no-image runs, the
     cosine distance between the two decision states at every layer.  Where it rises is
     where the current image starts reshaping the answer-relevant representation.
     Reported as VIP-like (change point) and TVI-like (area) because the ICLR 2026
     VIP/TVI paper's own chain-of-embedding aggregation is not reproduced exactly.

  2. **Logit-lens margin trajectory.**  At every layer the architecture-valid final
     norm + LM head is applied to the decision state and the GT-vs-best-wrong margin is
     scored.  Descriptive only: it shows when clean evidence pulls the trajectory away
     from the blind prior, and whether mismatch drags it toward the donor and gets
     overwritten later.

Neither is causal evidence.  Writes runs/current900/tvi.json, plots/, reports/TVI_LOGIT.md
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V      # noqa: E402
import featlib as F        # noqa: E402

OUT = V.ROOT / "runs" / "current900"
PLOTS = V.ROOT / "plots"
CONDS = ["clean", "no_image", "blur_severe", "modality_specific_severe",
         "random_same_modality_replacement"]
MODALITIES = ["MRI", "CT", "Xray"]


def main() -> int:
    items = V.load_items()
    res: dict = {"created": V.now(), "processor_id": V.MATCHED_PROCESSOR_ID,
                 "models": {}}

    for model in V.MODELS:
        m: dict = {"tvi": {}, "margin_trajectory": {}, "n_layers": None,
                   "layer_labels": None}
        if not F.has(model, "clean") or not F.has(model, "no_image"):
            res["models"][model] = {"error": "features missing"}
            continue
        ids, A, B = F.align(model, "clean", "no_image")
        T = A["decision_hidden"].shape[1]
        m["n_layers"] = T
        m["layer_labels"] = [F.layer_label(t, T) for t in range(T)]

        mods = [items[s]["modality"] for s in ids]
        for name, sel in [("all", np.ones(len(ids), bool))] + \
                         [(mo, np.array([x == mo for x in mods])) for mo in MODALITIES]:
            d = F.cosine_distance_by_layer(A["decision_hidden"][sel],
                                           B["decision_hidden"][sel])
            m["tvi"][name] = {
                "n": int(sel.sum()),
                "cosine_distance_by_layer": [float(x) for x in d],
                **F.vip_tvi(d),
            }
        # distance between the two models' clean states, for reference
        res["models"][model] = m

        # ---- margin trajectories ------------------------------------------- #
        for cond in CONDS:
            if not F.has(model, cond):
                continue
            meta = F.load_meta(model, cond)
            lens = F.load_arrays(model, cond, keys=("layer_candidate_logits",))[
                "layer_candidate_logits"]
            marg = F.gt_margin_by_layer(meta, lens, items)
            mm = [items[x["sample_id"]]["modality"] for x in meta]
            entry = {"n": len(meta),
                     "gt_margin_by_layer": [float(x) for x in marg.mean(0)],
                     "by_modality": {}}
            for mo in MODALITIES:
                sel = np.array([x == mo for x in mm])
                if sel.sum():
                    entry["by_modality"][mo] = {
                        "n": int(sel.sum()),
                        "gt_margin_by_layer": [float(x) for x in marg[sel].mean(0)]}
            m["margin_trajectory"][cond] = entry
        print(f"{model}: T={T} layers, TVI(all)="
              f"{m['tvi']['all']['tvi_like']:.4f}, VIP(all)="
              f"{m['tvi']['all']['vip_like_layer_label']}")

    # ---- aggregate questions the spec asks ------------------------------------ #
    qs: dict = {}
    for model in V.MODELS:
        m = res["models"].get(model, {})
        if "error" in m or not m.get("tvi"):
            continue
        tr = m["margin_trajectory"]
        clean = tr.get("clean", {}).get("gt_margin_by_layer")
        blind = tr.get("no_image", {}).get("gt_margin_by_layer")
        if clean and blind:
            cross = next((i for i, (a, b) in enumerate(zip(clean, blind)) if a > b),
                         None)
            qs[model] = {
                "tvi_like_all": m["tvi"]["all"]["tvi_like"],
                "vip_like_all": m["tvi"]["all"]["vip_like_layer_label"],
                "vip_like_mri": m["tvi"]["MRI"]["vip_like_layer_label"],
                "vip_like_ct": m["tvi"]["CT"]["vip_like_layer_label"],
                "vip_like_xray": m["tvi"]["Xray"]["vip_like_layer_label"],
                "first_layer_clean_margin_exceeds_blind": cross,
                "clean_margin_L1": clean[0], "blind_margin_L1": blind[0],
                "clean_margin_final": clean[-1], "blind_margin_final": blind[-1],
            }
    res["summary"] = qs
    V.write_json(OUT / "tvi.json", res)

    # ---- plots ----------------------------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. visual integration distance vs layer, per modality and model
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, model in zip(axes, V.MODELS):
            m = res["models"].get(model, {})
            if "error" in m or not m.get("tvi"):
                ax.set_title(f"{V.MODEL_DISPLAY[model]}: no data")
                continue
            for name, style in [("all", "-"), ("MRI", "--"), ("CT", "-."),
                                ("Xray", ":")]:
                y = m["tvi"][name]["cosine_distance_by_layer"]
                ax.plot(range(len(y)), y, style,
                        label=f"{name} (n={m['tvi'][name]['n']})", linewidth=1.8)
            v = m["tvi"]["all"]["vip_like_layer_index"]
            ax.axvline(v, color="k", alpha=0.35, linewidth=1)
            ax.text(v, ax.get_ylim()[1] * 0.05, f" VIP-like {m['tvi']['all']['vip_like_layer_label']}",
                    fontsize=8)
            ax.set_title(f"{V.MODEL_DISPLAY[model]} — clean vs no-image "
                         f"(TVI-like {m['tvi']['all']['tvi_like']:.3f})")
            ax.set_xlabel("hidden-state row (0 = after decoder layer 1)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("cosine distance, decision state")
        fig.suptitle("Where the current image enters the decision representation")
        fig.tight_layout()
        fig.savefig(PLOTS / "tvi_visual_integration.png", dpi=140)
        plt.close(fig)

        # 2. GT margin trajectory per condition
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, model in zip(axes, V.MODELS):
            m = res["models"].get(model, {})
            tr = m.get("margin_trajectory", {})
            for cond in CONDS:
                if cond not in tr:
                    continue
                y = tr[cond]["gt_margin_by_layer"]
                ax.plot(range(len(y)), y, label=cond, linewidth=1.8)
            ax.axhline(0, color="k", alpha=0.3, linewidth=1)
            ax.set_title(f"{V.MODEL_DISPLAY[model]} — logit-lens GT margin by layer")
            ax.set_xlabel("hidden-state row")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("p(GT) - max p(wrong), logit lens")
        fig.suptitle("Logit-lens margin trajectory (descriptive, not causal)")
        fig.tight_layout()
        fig.savefig(PLOTS / "logit_lens_margin_by_layer.png", dpi=140)
        plt.close(fig)
        plots = ["tvi_visual_integration.png", "logit_lens_margin_by_layer.png"]
    except Exception as exc:                       # pragma: no cover
        plots = [f"plot failure: {exc}"]

    # ---- markdown -------------------------------------------------------------- #
    L = ["# Visual integration and logit-lens trajectories", "",
         f"generated {res['created']} · matched processor "
         f"`{V.MATCHED_PROCESSOR_ID}`", "",
         "Both measurements below are **descriptive**. Neither shows that the answer "
         "used the image; that is what the patching and knockout stages are for.", "",
         "## VIP/TVI-like: where clean and no-image decision states diverge", "",
         "| model | subset | n | TVI-like | VIP-like (50% of area) | distance at row 0 | "
         "peak distance |", "|---|---|---|---|---|---|---|"]
    for model in V.MODELS:
        m = res["models"].get(model, {})
        if "error" in m or not m.get("tvi"):
            continue
        for name in ["all"] + MODALITIES:
            t = m["tvi"].get(name)
            if not t:
                continue
            L.append(f"| {V.MODEL_DISPLAY[model]} | {name} | {t['n']} | "
                     f"{t['tvi_like']:.4f} | {t['vip_like_layer_label']} | "
                     f"{t['base_distance']:.4f} | {t['peak_distance']:.4f} |")
    L += ["", "## Logit-lens GT margin by layer", "",
          "| model | condition | n | row 0 | row 10 | row 20 | final |",
          "|---|---|---|---|---|---|---|"]
    for model in V.MODELS:
        tr = res["models"].get(model, {}).get("margin_trajectory", {})
        for cond in CONDS:
            if cond not in tr:
                continue
            y = tr[cond]["gt_margin_by_layer"]
            idx = [0, min(10, len(y) - 1), min(20, len(y) - 1), len(y) - 1]
            L.append(f"| {V.MODEL_DISPLAY[model]} | `{cond}` | {tr[cond]['n']} | "
                     + " | ".join(f"{y[i]:.4f}" for i in idx) + " |")
    L += ["", "## What the spec asked these to answer", "",
          "| question | answer from this run |", "|---|---|"]
    for model, q in qs.items():
        L.append(f"| {V.MODEL_DISPLAY[model]}: does visual integration begin later / "
                 f"stay weaker? | TVI-like {q['tvi_like_all']:.4f}, VIP-like "
                 f"{q['vip_like_all']}; by modality MRI {q['vip_like_mri']}, CT "
                 f"{q['vip_like_ct']}, X-ray {q['vip_like_xray']} |")
    if qs:
        L += ["", "Note on reading the two models' curves side by side: the distance is "
              "computed **within** a checkpoint (clean vs its own no-image), so the "
              "axes are comparable in shape but the states themselves are not in a "
              "shared space. Cross-checkpoint claims rest on the behaviour tables and "
              "on within-checkpoint patching, not on overlaying these curves.", ""]
    if plots:
        L += [f"Plots: {', '.join('`plots/' + p + '`' for p in plots)}", ""]
    (V.ROOT / "reports" / "TVI_LOGIT.md").write_text("\n".join(L), encoding="utf-8")

    print(f"wrote runs/current900/tvi.json, reports/TVI_LOGIT.md, plots: {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
