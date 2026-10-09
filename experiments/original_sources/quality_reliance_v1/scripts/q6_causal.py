#!/usr/bin/env python
"""Q6-analysis -- summarise the Level 3 interventions and enforce the controls.

The identity control is checked first and hard.  Patching a row with the state the
model was about to produce must return the SAME margin to numerical precision; if
it does not, the hook is writing to the wrong row and nothing else in the file is
interpretable.  The report says so explicitly rather than burying it in a table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402


INTERVENTION_LABEL = {
    # The names describe what the intervention actually does to the tensor.  The
    # earlier labels ("direct" / "relay") suggested a distinction the code did not
    # implement, and one of them was in fact the same intervention repeated once
    # per layer -- seven identical rows in the table.
    "identity": "identity (row patched with its own state)",
    "content_patch": "content patch (donor state written into row L)",
    "random_patch": "random patch (unrelated item's state into row L)",
    "visual_knockout_direct": "visual ablation at ALL rows (L is irrelevant)",
    "visual_knockout_relay": "visual ablation from row L onward",
}
DIRECT_NOTE = (
    "`visual ablation at ALL rows` zeroes the image-token hidden states at every "
    "row, so it does not depend on L: the seven rows carrying different L values "
    "are the same intervention and are collapsed to one below. "
    "`visual ablation from row L onward` zeroes only from L onwards, which leaves "
    "the later rows free to re-derive something from the text stream."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2vl,medvlmr1")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]

    cdir = Q.ROOT / "results" / "causal"
    recs = []
    for m in models:
        p = cdir / f"CAUSAL__{m}.jsonl"
        if p.is_file():
            recs += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                     if x.strip()]
    ok = [r for r in recs if r.get("status") == "ok"]
    err = [r for r in recs if r.get("status") == "error"]
    rep = Q.ROOT / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    if not ok:
        (rep / "CAUSAL_RESULTS.md").write_text(
            "# Causal results\n\nNo Level 3 interventions have completed yet.\n"
            f"\nRecords present: {len(recs)} (errors: {len(err)}).\n"
            "\nThe planned design is in `scripts/q6_run_causal.py`: bidirectional "
            "content patching at selected rows with an identity control that must "
            "be exactly zero, plus direct and relay visual knockout.\n",
            encoding="utf-8")
        print("no causal results yet")
        return

    rows = []
    for m in models:
        sub = [r for r in ok if r["model"] == m]
        if not sub:
            continue
        # identity control, by layer
        ident = [r for r in sub if r["intervention"] == "identity"]
        max_ident = max((abs(r["delta_M"]) for r in ident), default=None)
        for ph in sorted({r["phenotype"] for r in sub}):
            for itv in sorted({r["intervention"] for r in sub}):
                if itv == "visual_knockout_direct":
                    # one intervention, one row -- L is not a variable here
                    layer_values = [0]
                else:
                    layer_values = sorted({r["layer"] for r in sub
                                           if r["intervention"] == itv})
                for L in layer_values:
                    s = [r for r in sub if r["phenotype"] == ph
                         and r["intervention"] == itv and r["layer"] == L]
                    if not s:
                        continue
                    d = np.array([r["delta_M"] for r in s])
                    js = np.array([r["js_vs_base"] for r in s])
                    rows.append({
                        "model": m, "phenotype": ph, "intervention": itv,
                        "intervention_label": INTERVENTION_LABEL.get(itv, itv),
                        "layer": L, "n": len(s),
                        "mean_delta_M": round(float(d.mean()), 4),
                        "median_delta_M": round(float(np.median(d)), 4),
                        "sd_delta_M": round(float(d.std(ddof=1))
                                            if len(d) > 1 else 0.0, 4),
                        "se_delta_M": round(float(d.std(ddof=1) / np.sqrt(len(d)))
                                            if len(d) > 1 else 0.0, 4),
                        "frac_delta_negative": round(float((d < 0).mean()), 4),
                        "frac_flip_correct_to_wrong": round(float(np.mean(
                            [r["correct_base"] and not r["correct_intervened"]
                             for r in s])), 4),
                        "frac_flip_wrong_to_correct": round(float(np.mean(
                            [not r["correct_base"] and r["correct_intervened"]
                             for r in s])), 4),
                        "mean_js_vs_base": round(float(js.mean()), 5),
                    })
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(Q.ROOT / "RESULTS" / "CAUSAL_RESULTS.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    lines = ["# Causal results (Level 3)", "",
             "Within-checkpoint only.  No activation is ever moved between "
             "checkpoints.", "",
             f"Records: {len(ok)} ok, {len(err)} errored.", "",
             "## Identity control", ""]
    if max_ident is not None:
        good = max_ident < 1e-3
        lines += [f"Largest |delta_M| for the identity intervention "
                  f"(row patched with the state the model was about to produce): "
                  f"**{max_ident:.2e}**.",
                  "",
                  ("The hook writes to the intended row; the effects below are "
                   "interpretable." if good else
                   "**This is not zero.  Either the hook targets the wrong row or "
                   "the forward pass is not deterministic here, and no effect "
                   "below can be interpreted.**")]
    else:
        lines.append("_No identity control recorded._")

    lines += ["", "## Intervention effects", "",
              DIRECT_NOTE, "",
              "| model | phenotype | intervention | layer | n | mean dM | "
              "median dM | frac dM<0 | correct->wrong | JS vs base |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['phenotype']} | {r['intervention']} | "
                     f"{r['layer']} | {r['n']} | {r['mean_delta_M']} | "
                     f"{r['median_delta_M']} | {r['frac_delta_negative']} | "
                     f"{r['frac_flip_correct_to_wrong']} | {r['mean_js_vs_base']} |")
    lines += [""]
    (rep / "CAUSAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote reports/CAUSAL_RESULTS.md ({len(rows)} rows)")
    Q.write_json(Q.ROOT / "runtime" / "q6_summary.json",
                 {"n_ok": len(ok), "n_err": len(err), "max_identity_delta_M":
                  max_ident, "n_rows": len(rows), "time": Q.now()})


if __name__ == "__main__":
    main()
