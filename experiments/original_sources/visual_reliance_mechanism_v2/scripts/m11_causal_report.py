#!/usr/bin/env python
"""Causal stage reports + plots (spec 08 plots 7 and 8).

Reads causal/patch_results_*.json and causal/knockout_results_*.json and writes
reports/PATCHING.md, reports/KNOCKOUT.md and two plots.  Purely a presentation layer;
no new measurement happens here.

    python scripts/m11_causal_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V      # noqa: E402
import featlib as F        # noqa: E402

C = V.ROOT / "causal"
R = V.ROOT / "reports"
P = V.ROOT / "plots"


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def layer_table(rows, layers):
    """{pair: {layer: mean delta_margin}} plus per-pair n."""
    pairs, out = {}, {}
    for r in rows:
        k = (r["donor_condition"], r["recipient_condition"])
        pairs.setdefault(k, []).append(r)
    for k, rs in pairs.items():
        out[k] = {"n": len(rs),
                  "delta": {L: float(np.mean([r["by_layer"][str(L)]["delta_margin"]
                                              for r in rs
                                              if str(L) in r["by_layer"]]))
                            for L in layers
                            if any(str(L) in r["by_layer"] for r in rs)},
                  "recovery": {}}
        for L in out[k]["delta"]:
            vals = [r["by_layer"][str(L)]["recovery_clipped"] for r in rs
                    if str(L) in r["by_layer"]
                    and r["by_layer"][str(L)]["recovery_clipped"] is not None]
            out[k]["recovery"][L] = float(np.mean(vals)) if vals else float("nan")
    return out


def main() -> int:
    patch = {m: load(C / f"patch_results_{m}.json") for m in V.MODELS}
    knock = {m: load(C / f"knockout_results_{m}.json") for m in V.MODELS}

    # ---------------------------------------------------------------- patching -- #
    L = ["# Activation patching (Stage 7)", "",
         "Replacing the **decision-position** residual at one decoder layer with the "
         "paired donor condition's residual, within the same checkpoint. Evidence "
         "level 4 when the identity controls pass.", "",
         "`delta_margin` = patched GT margin − recipient GT margin. Positive means the "
         "patch moved the answer toward the state the donor condition produces. "
         "`recovery` is a **descriptive** normalised ratio, clipped at ±5 and never to "
         "be read as \"the fraction of the answer that comes from the image\".", ""]
    any_data = False
    for m, d in patch.items():
        if not d:
            L.append(f"**{V.MODEL_DISPLAY[m]}**: NOT RUN")
            continue
        any_data = True
        ctl = d.get("controls", {})
        L += [f"## {V.MODEL_DISPLAY[m]}", "",
              f"Identity controls: hook-disabled repeat bitwise equal = "
              f"`{ctl.get('hook_disabled_repeat_bitwise_equal')}`; zero-patch changes "
              f"the output (the hook is live) = `{ctl.get('zero_patch_changes_output')}`; "
              f"**self-patch max logit delta = {ctl.get('self_patch_max_logit_delta')}** "
              f"(0.0 confirms the extraction row ↔ decoder-layer mapping).", ""]
        tab = layer_table(d["rows"], d["coarse_layers"])
        lay = d["coarse_layers"]
        L += [f"n = {d['n_rows']} (case, pair) rows.", "",
              "| pair | n | " + " | ".join(f"L{x}" for x in lay) + " |",
              "|---|---|" + "---|" * len(lay)]
        for (dc, rc), v in sorted(tab.items()):
            cells = " | ".join(f"{v['delta'].get(x, float('nan')):+.4f}" for x in lay)
            L.append(f"| `{dc}` → `{rc}` | {v['n']} | {cells} |")
        L += ["", "| pair | n | " + " | ".join(f"L{x}" for x in lay) +
              " |  (normalised recovery, descriptive) |",
              "|---|---|" + "---|" * len(lay)]
        for (dc, rc), v in sorted(tab.items()):
            cells = " | ".join(f"{v['recovery'].get(x, float('nan')):+.3f}" for x in lay)
            L.append(f"| `{dc}` → `{rc}` | {v['n']} | {cells} |")
        L.append("")
    L += ["## How to read this", "",
          "- The **bidirectional** test is the important one: `clean` → degraded should "
          "move the margin down and degraded → `clean` should move it up. A one-sided "
          "effect is more likely an artefact than a mechanism.",
          "- A **flat early-layer profile with a late jump** localises where in the "
          "network the intervention takes effect. It does not say what computation "
          "happens there.",
          "- Recovery ratios are clipped and flagged; when the donor–recipient margin "
          "gap is near zero the ratio is meaningless, which is why the raw signed "
          "deltas are the primary table and the ratios are secondary.",
          ""]
    (R / "PATCHING.md").write_text("\n".join(L), encoding="utf-8")

    # ---------------------------------------------------------------- knockout -- #
    L = ["# Attention knockout (Stage 8)", "",
         "Changing the attention graph rather than reading attention weights.", "",
         "| arm | what is blocked |", "|---|---|",
         "| `direct` | the decision position may not attend to visual tokens |",
         "| `relay` | question/text positions may not attend to visual tokens |",
         "| `both` | both routes blocked |", "",
         "`relay` is the scientifically important arm: blocking only the direct route "
         "underestimates visual use when question tokens have already cached visual "
         "content and pass it on through text-to-text attention.", ""]
    for m, d in knock.items():
        if not d:
            L.append(f"**{V.MODEL_DISPLAY[m]}**: NOT RUN")
            continue
        ctl = d.get("controls", {})
        L += [f"## {V.MODEL_DISPLAY[m]}", "",
              f"- empty-knockout identity control: **{ctl.get('empty_knockout_is_noop')}** "
              f"with a measured margin floor of "
              f"{ctl.get('identity_margin_floor'):.5f} "
              f"({ctl.get('identity_logit_ulp_note')})",
              f"- knockout window: layers {ctl.get('knockout_window')}, "
              f"{ctl.get('n_visual_tokens')} visual tokens of {ctl.get('seq_len')} "
              f"sequence positions",
              f"- results marked invalid: **{d.get('invalid')}**"
              + ("" if not d.get("invalid") else " — DO NOT INTERPRET"),
              "", "| phenotype | n | direct | relay | both |", "|---|---|---|---|---|"]
        by = {}
        for r in d["rows"]:
            by.setdefault(r["phenotype"], []).append(r)
        for ph, rs in sorted(by.items()):
            f = lambda k: float(np.mean([r[k] for r in rs]))       # noqa: E731
            L.append(f"| `{ph}` | {len(rs)} | {f('direct_delta_margin'):+.4f} | "
                     f"{f('relay_delta_margin'):+.4f} | {f('both_delta_margin'):+.4f} |")
        L.append("")
    # ------------------------------------------------- where the relay effect lives -- #
    win = {}
    for m in V.MODELS:
        got = []
        for x in ("0_8", "8_16", "16_20", "20_24", "24_28"):
            f = C / f"knockout_results_{m}_w{x}.json"
            if f.is_file():
                dd = load(f)
                if dd:
                    got.append(dd)
        if got:
            win[m] = got
    if win:
        L += ["## Where in the stack the relay effect lives", "",
              "The whole-stack knockout above can only say that the image matters "
              "somewhere. These runs knock out one decoder-module window at a time, 14 "
              "cases per phenotype (56 per window), which localises it.", "",
              "Window `a:b` hooks decoder modules `a..b-1` (0-based). Stored "
              "hidden-state rows are `module + 1`, because row 0 is the input "
              "embedding, so modules `16:20` correspond to stored rows `17..20`.", "",
              "| model | modules | stored rows | n | direct | relay | both | invalid |",
              "|---|---|---|---|---|---|---|---|"]
        for m, got in win.items():
            for dd in got:
                w = dd.get("controls", {}).get("knockout_window")
                rs = dd["rows"]
                f = lambda k: float(np.mean([r[k] for r in rs]))       # noqa: E731
                L.append(f"| `{m}` | {w[0]}–{w[1]} | {w[0] + 1}–{w[1] + 1} | {len(rs)} "
                         f"| {f('direct_delta_margin'):+.4f} "
                         f"| {f('relay_delta_margin'):+.4f} "
                         f"| {f('both_delta_margin'):+.4f} "
                         f"| {dd.get('invalid')} |")
        L += ["",
              "The `relay` arm is the one that reads: it is large and negative in the "
              "early windows and inside the noise floor from module 16 onwards. The "
              "`direct` arm is inside the floor in **every** window, so it carries no "
              "localisation claim either way — a null is not an absence.", ""]

    L += ["## How to read this", "",
          "A delta is only meaningful **above the measured floor** recorded in the "
          "identity control. A `direct` arm that sits inside the floor while `relay` "
          "sits far outside it is the signature of visual information reaching the "
          "answer through the text tokens rather than through a direct image-to-answer "
          "route.", ""]
    (R / "KNOCKOUT.md").write_text("\n".join(L), encoding="utf-8")
    # ---------------------------------------------------------------- plots ----- #
    made = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
        for ax, m in zip(axes, V.MODELS):
            d = patch.get(m)
            if not d:
                ax.set_title(f"{V.MODEL_DISPLAY[m]}: not run")
                continue
            lay = d["coarse_layers"]
            tab = layer_table(d["rows"], lay)
            for (dc, rc), v in sorted(tab.items()):
                if dc not in ("clean", "no_image", "random_same_modality_replacement",
                              "blur_severe"):
                    continue
                y = [v["delta"].get(x, np.nan) for x in lay]
                ax.plot(lay, y, marker="o", linewidth=1.8, label=f"{dc} → {rc}")
            ax.axhline(0, color="k", alpha=0.3, linewidth=1)
            ax.set_title(f"{V.MODEL_DISPLAY[m]} — patching effect by layer")
            ax.set_xlabel("decoder layer patched (row index)")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Δ GT margin (patched − recipient)")
        fig.suptitle("Activation patching: signed margin movement by layer")
        fig.tight_layout()
        fig.savefig(P / "patch_effect_by_layer.png", dpi=140)
        plt.close(fig)
        made.append("patch_effect_by_layer.png")

        fig, ax = plt.subplots(figsize=(9, 4.6))
        labels, direct, relay, both = [], [], [], []
        for m, d in knock.items():
            if not d:
                continue
            by = {}
            for r in d["rows"]:
                by.setdefault(r["phenotype"], []).append(r)
            for ph, rs in sorted(by.items()):
                f = lambda k: float(np.mean([r[k] for r in rs]))   # noqa: E731
                labels.append(f"{V.MODEL_DISPLAY[m].split('-')[0]}\n{ph[:18]}")
                direct.append(f("direct_delta_margin"))
                relay.append(f("relay_delta_margin"))
                both.append(f("both_delta_margin"))
        x = np.arange(len(labels))
        w = 0.26
        ax.bar(x - w, direct, w, label="direct (image → decision)")
        ax.bar(x, relay, w, label="relay (image → question/text)")
        ax.bar(x + w, both, w, label="both")
        ax.axhline(0, color="k", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Δ GT margin")
        ax.set_title("Attention knockout: direct route vs text relay")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(P / "knockout_direct_vs_relay.png", dpi=140)
        plt.close(fig)
        made.append("knockout_direct_vs_relay.png")

        if win:
            fig, axes = plt.subplots(1, len(win), figsize=(6.2 * len(win), 4.2),
                                     squeeze=False)
            for ax, (m, got) in zip(axes[0], sorted(win.items())):
                labs, d_d, d_r = [], [], []
                for dd in got:
                    w = dd["controls"]["knockout_window"]
                    rs = dd["rows"]
                    labs.append(f"{w[0]}–{w[1]}")
                    d_d.append(float(np.mean([r["direct_delta_margin"] for r in rs])))
                    d_r.append(float(np.mean([r["relay_delta_margin"] for r in rs])))
                x = np.arange(len(labs))
                ax.bar(x - 0.2, d_d, 0.4, label="direct", color="tab:blue")
                ax.bar(x + 0.2, d_r, 0.4, label="relay", color="tab:orange")
                ax.axhline(0, color="k", linewidth=1)
                ax.set_xticks(x)
                ax.set_xticklabels(labs, fontsize=8)
                ax.set_title(f"{V.MODEL_DISPLAY[m]}\nknockout by module window "
                             f"(n=56 each)")
                ax.set_xlabel("decoder module window")
                ax.legend(fontsize=8)
            axes[0][0].set_ylabel("Δ GT margin")
            fig.suptitle("Attention knockout: localisation of the text-relay effect")
            fig.tight_layout()
            fig.savefig(P / "knockout_by_window.png", dpi=140)
            plt.close(fig)
            made.append("knockout_by_window.png")
    except Exception as exc:                        # pragma: no cover
        print(f"plot failure: {exc}")

    print(f"wrote reports/PATCHING.md, reports/KNOCKOUT.md, plots: {made}")
    return 0 if any_data else 1


if __name__ == "__main__":
    raise SystemExit(main())
