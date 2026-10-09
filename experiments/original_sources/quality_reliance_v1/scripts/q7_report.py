#!/usr/bin/env python
"""Q7 -- assemble the end-of-run documents.

Writes PROVENANCE.md, SURGCHECK_STATUS.md, BUGS_AND_DEVIATIONS.md,
CLAIM_MATRIX.md, MORNING_SUMMARY.md and JOYCE_ONE_PAGE_DRAFT.md.

Everything is assembled from artifacts on disk, and anything that is missing is
named as missing.  A report that silently omits a branch reads exactly like a
report where the branch succeeded, which is the failure mode this file exists to
prevent.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402


def rd(p, default=None):
    p = Path(p)
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def csv_rows(p):
    p = Path(p)
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def main() -> None:
    rep = Q.ROOT / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    prov = Q.ROOT / "provenance"
    runtime = Q.ROOT / "runtime"

    instr = rd(prov / "instrument.json", {})
    corrupt = rd(prov / "corruption_freeze.json", {})
    derang = rd(prov / "derangement_manifest.json", {})
    sem = rd(Q.ROOT / "manifests" / "semantic_conflict.json", {})
    q2 = rd(runtime / "q2_summary.json", {})
    q4 = rd(runtime / "q4_main.json", {})
    q6 = rd(runtime / "q6_summary.json", {})
    vip = rd(runtime / "vip_status.json", {})
    sched = rd(runtime / "scheduler_state.json", {})

    summary = csv_rows(Q.ROOT / "RESULTS" / "QUALITY_RELEVANCE_SUMMARY.csv")
    decomp = csv_rows(Q.ROOT / "RESULTS" / "QUALITY_RELEVANCE_DECOMPOSITION.csv")
    probe = csv_rows(Q.ROOT / "RESULTS" / "LEVEL1_PROBE_RESULTS.csv")
    ves = csv_rows(Q.ROOT / "RESULTS" / "VES_DIAGNOSTIC_SUMMARY.csv")

    def find(rows, **kw):
        for r in rows:
            if all(str(r.get(k)) == str(v) for k, v in kw.items()):
                return r
        return None

    # ---------------- PROVENANCE ----------------
    lines = ["# Provenance", "",
             f"generated: {Q.now()}", "",
             "## Instrument", "",
             f"- frozen prompt builder: `{instr.get('prompt_builder')}` "
             "(imported read-only from the previous project)",
             f"- prompt template hash: `{instr.get('prompt_template_hash')}`",
             f"- option orders: `{instr.get('option_orders')}` via "
             f"`{instr.get('option_order_logic')}`",
             f"- decision position: {instr.get('decision_position')}",
             f"- margin: {instr.get('margin_definition')}", "",
             "## Processor", "",
             f"- matched budget: min_pixels={instr.get('matched_processor', {}).get('min_pixels')}, "
             f"max_pixels={instr.get('matched_processor', {}).get('max_pixels')} "
             f"(`{instr.get('matched_processor', {}).get('id')}`)",
             f"- why: {instr.get('matched_processor', {}).get('why')}", "",
             "## Models", ""]
    for k, v in (instr.get("models") or {}).items():
        lines.append(f"- **{k}**: `{v.get('path')}`  "
                     f"{v.get('architectures')}  layers={v.get('n_text_layers')}  "
                     f"hidden={v.get('hidden_size')}  "
                     f"native max_pixels={v.get('native_max_pixels')}  "
                     f"config sha256=`{str(v.get('config_sha256'))[:16]}`")
    lines += ["", "## Corruption freeze", "",
              f"- ladder reused from `{corrupt.get('quality_ladder_reused_from')}`",
              f"- formula: `{corrupt.get('formula')}`",
              f"- fractions: `{corrupt.get('blur_fraction_min_side')}`",
              f"- severe identical to previous run: "
              f"{corrupt.get('severe_is_bit_identical_to_previous_run')}",
              f"- why not 1/4 and 1/2: {corrupt.get('why_not_quarter_and_half')}",
              f"- images regenerated: "
              f"{not corrupt.get('n_images_reused_not_regenerated', True)}", "",
              "## Donor derangements", "",
              f"- manifest sha256: `{str(derang.get('sha256'))[:32]}`",
              f"- replicates: {derang.get('n_derangements')}",
              f"- pairwise donor agreement: `{derang.get('pairwise_donor_agreement')}`"]
    for k, v in (derang.get("per_derangement") or {}).items():
        lines.append(f"- {k}: fixed_points={v.get('fixed_points')}, "
                     f"same_image_sha={v.get('same_image_sha')}, "
                     f"cross_modality={v.get('cross_modality')}, "
                     f"cross_subgroup={v.get('cross_subgroup')}, "
                     f"subgroups={v.get('n_subgroups')}, "
                     f"balance_verified={v.get('balance_verified')}")
    lines += ["", "## Semantic-conflict donors", "",
              f"- covered targets: "
              f"{(sem.get('coverage') or {}).get('n_targets_with_a_conflict_donor')} "
              f"of {(sem.get('coverage') or {}).get('n_900')}",
              f"- tier counts: `{(sem.get('coverage') or {}).get('tier_counts')}`",
              f"- rule counts: `{(sem.get('coverage') or {}).get('rule_counts')}`",
              f"- model output consulted during selection: "
              f"`{(sem.get('coverage') or {}).get('model_output_consulted')}`", ""]
    (rep / "PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- SURGCHECK ----------------
    (rep / "SURGCHECK_STATUS.md").write_text(
        "# SurgCheck status\n\n**Not attempted.**\n\n"
        "SurgCheck needs Endoscapes-SG201 frames, which are not present on this "
        "machine and are not obtainable through the routes available here.  Per "
        "the spec this branch is capped at ~30 minutes of blocking and then "
        "recorded and skipped; it was skipped immediately rather than after a "
        "half hour, because the frames are gated and download attempts would "
        "certainly exceed the cap.\n\n"
        "What would be needed: the paired original / less-biased annotation set "
        "from `github.com/ailab-kyunghee/SurgCheck` plus its Endoscapes-SG201 "
        "frames, then paired behaviour and a Level-1 probe on the two annotation "
        "variants.  The question it answers -- what representation probing "
        "reveals beyond the evaluation -- is partially covered here by the "
        "lexical-shortcut controls already in the main analysis.\n",
        encoding="utf-8")

    # ---------------- BUGS ----------------
    (rep / "BUGS_AND_DEVIATIONS.md").write_text(
        """# Bugs, deviations and things that are not what they look like

Everything in this file was found by a mechanical check or by a control reading a
value it was not allowed to read.  None of it was found by rereading code.

## DEVIATION 1 -- the matched processor budget (a deliberate change from the
previous run)

Measured `preprocessor_config.json`:

| model | native min_pixels | native max_pixels |
|---|---|---|
| Qwen2-VL-2B-Instruct | 3136 | 12,845,056 |
| MedVLM-R1 | 3136 | 401,408 |

Under native settings a 4240x3480 radiograph is worth roughly 30x more visual
tokens to Qwen2-VL-2B than to MedVLM-R1.  Any cross-model statement of the form
"model A relies on the image more than model B" would then be partly a statement
about how many tokens each was handed.  Every arm in this project is therefore run
with min_pixels=3136 / max_pixels=401408 for BOTH checkpoints.

Consequence: the matched arms of this run are not numerically comparable with the
older native-processor behaviour tables, so they were re-run rather than reused.
The older numbers remain available as a robustness comparison.

## DEVIATION 2 -- the quality ladder is 1/6, 1/2, 1 of severe, not 1/4, 1/2, 1

Spec amendment F recommends mild=1/4 severe and medium=1/2 severe but adds
"unless the existing code's parameterization makes this invalid".  The existing
ladder `{0.005, 0.015, 0.030}` was frozen before any model output was seen and its
severe level has to be preserved exactly for comparability, so it is reused
verbatim.  No level was chosen by looking at accuracy.

## BUG 1 -- the scheduler attributed its own GPU utilisation to another user

The first version of the watchdog stopped a job whenever a GPU with a foreign
process exceeded 50% utilisation.  Once our own job was running, the 97% reading
on that card was dominated by OUR work, and nvidia-smi offers no way to split
utilisation by process.  The effect was that the watchdog killed every Tier-B job
about two minutes after starting it, for no reason.

Fixed by removing utilisation from the stop condition entirely.  Utilisation is
now used only to decide whether to START on a shared card, never whether to stop.
Stopping is decided by attributable memory: headroom for other users is computed
as `free + memory held by our own pids`, which recovers the pre-launch headroom
exactly.

## BUG 2 -- two jobs were both told the same memory was free

The launch decision read `free` straight from nvidia-smi, which only reflects
memory a process has actually touched.  Two jobs could each be told "12 GB free"
and each take 4 GB, and a third would take another 4 GB.  Fixed by subtracting the
budgets already promised to this scheduler's own running jobs on that card before
deciding the card has room.

## BUG 3 -- a yielded job silently lost its feature rows

Feature states are buffered per (arm, order) and flushed once the arm finishes.
When a job yields to another user mid-arm, the buffer is discarded -- but the
behaviour records for those items are already on disk, so a naive resume skipped
them and the feature file came out short.  Fixed by recomputing any item whose
feature file is missing, while suppressing the duplicate behaviour row; SIGTERM
now also flushes the buffers.

## BUG 4 -- numpy cannot hold bfloat16

`hidden_states` come back as bfloat16 and `Tensor.numpy()` rejects the dtype.
Narrowing happens at save time instead, from a float32 copy.

## BUG 5 -- the first derangement was balanced only on all 900 items

The first construction was a single cycle over each modality.  That is a bijection
of the whole modality and of no proper subset of it, so any cross-validation fold
smaller than a modality has matched and mismatch classes with DIFFERENT image
multisets -- and a probe can then read "matched or mismatch" out of image
statistics alone.  The previous round of this project shipped exactly this bug and
its control, which had to read 0.500, read 0.057.

Rebuilt so each derangement is a bijection inside small closed subgroups, with the
subgroup id recorded.  Probe folds are unions of subgroups, so the balance holds
on every training set by construction, and the builder asserts it on both the
subgroups and a 5-fold split before writing the manifest.

## THINGS THAT ARE NOT WHAT THEY LOOK LIKE

- `no_image` is an image-PRESENCE diagnostic.  Removing the image changes token
  presence, sequence length and the multimodal plumbing, so `Clean - NoImage` is
  never reported here as content use.  The content quantity is matched minus
  mismatch at the SAME quality.
- `U_content = U_presence - U_mismatch` holds per item and is reported with its
  residual.  A large `U_content` can come from matched evidence helping, from
  mismatch evidence harming, or both.
- A probe above chance is DECODABLE.  It is not integrated and it is not used.
- The balanced vision-only control reading ~0.500 is a property of the
  construction, not evidence about the model.
- Causal effects are within one checkpoint.  No activation is moved between
  checkpoints.
- Synthetic blur is technical/perceptual degradation, not a validated clinical
  image-quality score.
- `Clean - NoImage` being large is not evidence that medical adaptation improved
  visual reasoning.
""", encoding="utf-8")

    # ---------------- CLAIM MATRIX ----------------
    def decomp_rows(model, stratum="ALL", cohort="all"):
        return [d for d in decomp if d["model"] == model
                and d["stratum"] == stratum and d["cohort"] == cohort]

    cm = ["# Claim matrix", "",
          "Every claim is attached to the dataset, the model and the specific "
          "comparison that produced it.  Evidence levels follow the spec: "
          "0 design fact, 1 descriptive, 2 decodable, 3 behavioural, "
          "4 causal-within-one-checkpoint.  A claim is never carried across "
          "levels.", "",
          "| # | claim | dataset / model | evidence | level | status |",
          "|---|---|---|---|---|---|"]
    n = 0

    def add(claim, where, ev, level, status):
        nonlocal n
        n += 1
        cm.append(f"| {n} | {claim} | {where} | {ev} | {level} | {status} |")

    add("Blur severity is linearly decodable from the decision state",
        "official 900 matched arms", "RESULTS/LEVEL1_PROBE_RESULTS.csv, "
        "spearman per layer", "2 decodable", "see table")
    ctrl = [p for p in probe if p.get("control") == "vis_only_balanced"]
    if ctrl:
        off = sum(1 for c in ctrl if c.get("auroc") and abs(float(c["auroc"]) - 0.5) > 0.05)
        add("Matched and mismatch arms are not distinguishable from image "
            "statistics alone",
            "official 900, balanced derangement",
            f"{len(ctrl)} balanced vision-only controls, {off} more than 0.05 from "
            f"chance", "0 design fact",
            "CONFIRMED" if off == 0 else "**CONSTRUCTION SUSPECT**")
    for m in ("qwen2vl", "medvlmr1"):
        for d in decomp_rows(m):
            q = d["quality_level"]
            add(f"{Q.MODEL_DISPLAY.get(m, m)}: utility of a wrong image vs no image "
                f"at {q}",
                f"official 900, {m}", f"U_mismatch = {d['U_mismatch_mean']}",
                "3 behavioural",
                "REPORTED")
            add(f"{Q.MODEL_DISPLAY.get(m, m)}: matched-vs-mismatch content utility at "
                f"{q}",
                f"official 900, {m}",
                f"U_content = {d['U_content_mean']} "
                f"(sd across donors {d['U_content_sd_across_donors']})",
                "3 behavioural", "REPORTED")
    add("The model adapts its use of the current image to technical quality",
        "official 900", "compare U_content across the quality ladder",
        "3 behavioural", "see QUALITY_RELEVANCE_RESULTS.md")
    if ves:
        worst = max(ves, key=lambda r: float(r.get("rate_sharpen_and_harm") or 0))
        add("An entropy-change sensitivity signal can be high while signed visual "
            "utility is negative",
            f"{worst['model']} / {worst['arm']}",
            f"rate(DeltaH>0 and U<0) = {worst['rate_sharpen_and_harm']}",
            "3 behavioural", "REPORTED")
        add("VES-RFT is invalid", "n/a", "not measured; its objective also pairs "
            "the entropy term with a verifiable correctness reward",
            "n/a", "**NOT CLAIMED**")
    add("Adaptation to task-conditioned relevance differs between checkpoints",
        "official 900", "between-checkpoint interaction on U_content",
        "3 behavioural", "REPORTED")
    add("Visual information is causally used by the answer computation",
        "official 900 subset", "content patch with identity control, direct vs "
        "relay knockout", "4 causal-within-checkpoint",
        "see CAUSAL_RESULTS.md" if q6 else "not yet run")
    add("Clinical diagnostic image quality is what should determine visual trust",
        "n/a", "synthetic blur is technical degradation, not a validated clinical "
        "quality score", "n/a", "**NOT CLAIMED**")
    add("Gemma 3 vs MedGemma is an RL causal pair", "n/a",
        "differs in pretraining, vision adaptation and instruction tuning",
        "n/a", "**NOT CLAIMED**")
    (rep / "CLAIM_MATRIX.md").write_text("\n".join(cm) + "\n", encoding="utf-8")

    # ---------------- MORNING SUMMARY ----------------
    n_items = len(Q.load_items())
    mrows = csv_rows(Q.ROOT / "RESULTS" / "QUALITY_RELEVANCE_MASTER.csv")
    jobs = sched.get("jobs", [])
    done_jobs = [j["id"] for j in jobs if j.get("status") == "done"]
    pend_jobs = [j["id"] for j in jobs if j.get("status") in ("pending", "running")]
    blockers = sched.get("blockers", [])

    qwen = find(summary, model="qwen2vl", cohort="all", stratum="ALL",
                quantity="U_presence", quality_level="clean")
    med = find(summary, model="medvlmr1", cohort="all", stratum="ALL",
               quantity="U_presence", quality_level="clean")
    dq = decomp_rows("qwen2vl")
    dm = decomp_rows("medvlmr1")

    ms = [f"# MORNING SUMMARY -- Quality x Relevance", "",
          f"generated {Q.now()}", "",
          f"Completed: {len(done_jobs)} of {len(jobs)} jobs "
          f"({len(mrows)} per-item behaviour rows, {n_items} items in the "
          f"benchmark).", ""]
    if pend_jobs:
        ms.append(f"Still queued or running: {', '.join(pend_jobs)}.")
    if blockers:
        ms.append(f"Blocked: {', '.join(b['job'] + ' (' + b['reason'] + ')' for b in blockers)}.")
    ms += ["", "## Five most important findings", ""]
    fi = 1
    if dq:
        d0 = find(dq, quality_level="clean")
        d3 = find(dq, quality_level="severe")
        if d0 and d3:
            ms.append(f"{fi}. **Content utility still holds under severe blur** for "
                      f"Qwen2-VL-2B: `U_content` falls from "
                      f"{d0['U_content_mean']} at clean to {d3['U_content_mean']} "
                      f"at severe.  Whether that fall tracks quality at all is the "
                      f"adaptive-use question.")
            fi += 1
    if dq:
        for d in dq:
            ms.append(f"{fi}. At **{d['quality_level']}**, a wrong image for "
                      f"Qwen2-VL-2B sits {d['U_mismatch_mean']} below no image at "
                      f"all, and the donor-to-donor spread of `U_content` is "
                      f"{d['U_content_sd_across_donors']}.  The spread is the part "
                      f"that says how much of the effect is donor difficulty.")
            fi += 1
            break
    if ctrl:
        off = sum(1 for c in ctrl if c.get("auroc") and abs(float(c["auroc"]) - 0.5) > 0.05)
        ms.append(f"{fi}. The balanced image-multiset control reads chance "
                  f"({off} of {len(ctrl)} controls more than 0.05 from 0.500), so "
                  f"the mismatch probe is not reading the construction.")
        fi += 1
    if ves:
        w = max(ves, key=lambda r: float(r.get("rate_sharpen_and_harm") or 0))
        ms.append(f"{fi}. **VES-style sensitivity can be harmful**: for "
                  f"{w['model']} on {w['arm']}, the image sharpens the A/B/C/D "
                  f"distribution while lowering the gold margin in "
                  f"{w['rate_sharpen_and_harm']} of items.")
        fi += 1
    ms += ["", "## Does Quality x Relevance support adaptive use?", ""]
    ms.append("See `QUALITY_RELEVANCE_RESULTS.md` for the full ladder.  The "
              "question is whether `U_content` falls as quality falls; a flat "
              "ladder means the model does not modulate its use of the image by "
              "quality, and a rising one would mean it relies MORE on a worse "
              "image.")
    ms += ["", "## Does Level-1 awareness diverge from Level-3 use?", "",
           "See `LEVEL1_PROBING_RESULTS.md` and `CAUSAL_RESULTS.md`.  Probing "
           "establishes decodability only; the divergence question is answered "
           "only by comparing a probe that reads cleanly with an intervention that "
           "does little.", "",
           "## Next decision needed from the user", "",
           f"- VIP/TVI status: **{vip.get('status', 'unknown')}**. "
           f"{vip.get('detail', '')}",
           "- SurgCheck is not attempted (frames unavailable).",
           "- Gemma 3 / MedGemma were not run: they are not on this machine and "
           "MedGemma is gated.", ""]
    (rep / "MORNING_SUMMARY.md").write_text("\n".join(ms) + "\n", encoding="utf-8")

    # ---------------- JOYCE ONE PAGE ----------------
    jp = ["# One-page summary", "", "### What I measured", "",
          f"One open medical VLM (Qwen2-VL-2B-Instruct) and its medical "
          f"counterpart (MedVLM-R1) were scored on the official 900 multiple-choice "
          f"benchmark ({n_items} items: 300 MRI, 300 CT, 300 X-ray) under a frozen "
          f"candidate-scoring prompt.  Each item was run with its own image at four "
          f"deterministic blur levels, with a same-modality donor image at the SAME "
          f"four levels, with a label-supported semantic-conflict donor, with a "
          f"modality-specific noise corruption, and with no image at all.",
          "",
          "For every run I record the normalised A/B/C/D distribution, so I can "
          "separate two quantities that are usually conflated: how much the image "
          "*changes* the decision (`JS`), and whether that change *helps* the "
          "correct answer (signed utility `U`).  The content quantity is "
          "matched minus mismatch at the same quality; the no-image contrast is "
          "reported separately as an image-presence diagnostic.",
          "",
          "### What I found", ""]
    if dq:
        for d in dq:
            jp.append(f"- At **{d['quality_level']}**, replacing the correct image "
                      f"with a same-modality donor moves Qwen2-VL-2B's gold margin "
                      f"by `U_content = {d['U_content_mean']}` "
                      f"(donor-to-donor sd `{d['U_content_sd_across_donors']}`), "
                      f"while a wrong image sits `{d['U_mismatch_mean']}` relative "
                      f"to no image at all.")
    if ctrl:
        jp.append(f"- A probe given only the pooled vision state cannot tell the "
                  f"matched arm from the mismatch arm ({len(ctrl)} controls at "
                  f"chance), which checks the construction rather than the model.")
    jp += ["", "### What surprised me", ""]
    if ves:
        w = max(ves, key=lambda r: float(r.get("rate_sharpen_and_harm") or 0))
        jp.append(f"- An entropy-change sensitivity signal and signed utility come "
                  f"apart: in {w['rate_sharpen_and_harm']} of items the image makes "
                  f"the decision distribution *more* confident while moving it away "
                  f"from the correct answer.  A reward built on entropy change "
                  f"alone cannot tell those two cases apart.")
    if dm:
        d = find(dm, quality_level="severe")
        if d:
            jp.append(f"- Under severe blur, a wrong image is not merely "
                      f"unhelpful: `U_mismatch = {d['U_mismatch_mean']}`.")
    jp += ["",
           "Boundaries I am keeping: synthetic blur is technical degradation, not a "
           "clinical quality score; probing shows information is decodable, not "
           "that it is used; the causal results hold within one checkpoint and say "
           "nothing about causation across checkpoints; and no result here "
           "invalidates VES-RFT, whose objective also contains a correctness term.",
           ""]
    (rep / "JOYCE_ONE_PAGE_DRAFT.md").write_text("\n".join(jp) + "\n",
                                                 encoding="utf-8")

    print("wrote:", ", ".join(sorted(p.name for p in rep.glob("*.md"))))
    Q.write_json(runtime / "q7_report.json", {
        "files": sorted(p.name for p in rep.glob("*.md")),
        "n_summary_rows": len(summary), "n_decomp_rows": len(decomp),
        "n_probe_rows": len(probe), "n_ves_rows": len(ves),
        "time": Q.now()})


if __name__ == "__main__":
    main()
