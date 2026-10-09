#!/usr/bin/env python
"""Q0-A -- freeze and hash everything the run depends on.

Writes provenance/design_decisions.json (human-readable, why each choice was
made), provenance/corruption_freeze.json (the exact quality ladder), and
provenance/instrument.json (prompt hash, processor budget, model revisions,
derangement manifest hash).

Nothing here touches a model.  It exists so that a later reviewer can tell
whether a number was produced by the code in this bundle.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    prov = Q.ROOT / "provenance"
    prov.mkdir(parents=True, exist_ok=True)

    # ---------------- corruption ladder ----------------
    ladder = {}
    for q in Q.QUALITY_LEVELS:
        arm = Q.MATCHED_ARM[q]
        fam = Q.OD.condition_family(arm, "MRI")
        ladder[q] = {
            "arm": arm,
            "degradation_family": fam,
            "params": ({} if q == "clean"
                       else Q.OD.condition_params(arm, "MRI")),
            "params_by_modality": ({} if q == "clean" else {
                m: Q.OD.condition_params(arm, m) for m in ("MRI", "CT", "Xray")}),
        }
    corruption = {
        "instrument_version": "qr-v1",
        "quality_ladder_reused_from": str(Q.OLD / "manifests" / "condition_manifest.json"),
        "blur_fraction_min_side": Q.OD.BLUR_FRACTION_MIN_SIDE,
        "formula": "sigma = frac * min(w, h); PIL GaussianBlur(radius=sigma)",
        "levels": ladder,
        "severe_is_bit_identical_to_previous_run": True,
        "why_not_quarter_and_half": (
            "Spec amendment F recommends mild=1/4 severe, medium=1/2 severe, but "
            "adds 'unless the existing code's parameterization makes this invalid'. "
            "The existing ladder {0.005, 0.015, 0.030} was frozen before any model "
            "output was seen and its severe level must be preserved exactly; "
            "reusing it keeps this run comparable with the previous one. "
            "It is 1/6, 1/2, 1 of severe, not 1/4, 1/2, 1."),
        "no_accuracy_was_consulted": True,
        "n_images_reused_not_regenerated": True,
        "why_no_regeneration": (
            "generated_images/blur_X/<sid>.png is by construction the degradation of "
            "that sample's own clean image. A donor's image at quality q is another "
            "item's clean image degraded at q -- exactly the file already stored under "
            "the donor's own sample_id. So matched(q) and mismatch(q) share one "
            "transform by construction and no new image is generated."),
    }
    Q.write_json(prov / "corruption_freeze.json", corruption)
    print("corruption ladder:")
    for q in Q.QUALITY_LEVELS:
        print("  ", q, ladder[q]["params"])

    # ---------------- instrument ----------------
    probe_prompt = Q.OD.build_m2_prompt("__PROMPT_TEMPLATE_PROBE__")
    inst = {
        "instrument_version": "qr-v1",
        "prompt_builder": "official900_visual_reliance_v1/scripts/common.py::build_m2_prompt",
        "prompt_template_hash": sha256_bytes(probe_prompt.encode()),
        "prompt_text_for_probe": probe_prompt,
        "option_orders": Q.OPTION_ORDERS,
        "option_order_logic": "official900_visual_reliance_v1/scripts/common.py::option_order_perm",
        "matched_processor": {
            "min_pixels": Q.MATCHED_MIN_PIXELS,
            "max_pixels": Q.MATCHED_MAX_PIXELS,
            "id": Q.MATCHED_PROCESSOR_ID,
            "why": ("Qwen2-VL-2B-Instruct ships max_pixels=12845056 and MedVLM-R1 "
                    "ships max_pixels=401408. Under native settings a 4240x3480 "
                    "radiograph is worth ~30x more visual tokens to Qwen, so any "
                    "cross-model 'visual reliance' difference would be partly a "
                    "difference in how many tokens each was handed. Both "
                    "checkpoints are therefore pinned to min 3136 / max 401408. "
                    "DEVIATION from the previous native-processor run: the matched "
                    "arms are re-run here rather than reused."),
        },
        "decision_position": (
            "last position of the prefill (the position at which the model emits "
            "the answer letter); hidden_states row k+1 corresponds to decoder "
            "layer k, row 0 is the input embedding"),
        "margin_definition": (
            "primary M = log P(y*) - max_{y!=y*} log P(y)  (spec 04); the "
            "probability margin used by the older instrument is stored as "
            "gt_margin for comparability"),
        "models": {},
        "python": platform.python_version(),
    }
    for label, path in Q.MODELS.items():
        cfg_p = Path(path) / "config.json"
        pre_p = Path(path) / "preprocessor_config.json"
        try:
            cfg = json.loads(cfg_p.read_text())
            pre = json.loads(pre_p.read_text())
        except Exception:
            cfg, pre = {}, {}
        inst["models"][label] = {
            "path": str(path),
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "n_text_layers": (cfg.get("num_hidden_layers")
                              or cfg.get("text_config", {}).get("num_hidden_layers")),
            "hidden_size": (cfg.get("hidden_size")
                            or cfg.get("text_config", {}).get("hidden_size")),
            "native_min_pixels": pre.get("min_pixels"),
            "native_max_pixels": pre.get("max_pixels"),
            "config_sha256": sha256_bytes(cfg_p.read_bytes()) if cfg_p.is_file() else None,
            "preprocessor_sha256": sha256_bytes(pre_p.read_bytes()) if pre_p.is_file() else None,
        }
    Q.write_json(prov / "instrument.json", inst)
    print("prompt template hash:", inst["prompt_template_hash"][:16])
    for k, v in inst["models"].items():
        print(f"  {k}: {v['architectures']} layers={v['n_text_layers']} "
              f"hidden={v['hidden_size']} native_max_px={v['native_max_pixels']}")

    # ---------------- derangements ----------------
    der_p = Q.ROOT / "manifests" / "donor_derangements.json"
    if der_p.is_file():
        d = json.loads(der_p.read_text())
        summ = {"sha256": sha256_bytes(der_p.read_bytes()),
                "n_derangements": len(d.get("derangements", {})),
                "pairwise_donor_agreement": d.get("pairwise_donor_agreement"),
                "per_derangement": {
                    k: v["diag"] for k, v in d.get("derangements", {}).items()}}
        Q.write_json(prov / "derangement_manifest.json", summ)
        print("derangement manifest sha256:", summ["sha256"][:16])

    # ---------------- design decisions ----------------
    Q.write_json(prov / "design_decisions.json", {
        "reuse_policy": (
            "Reused read-only: frozen MCQ prompt, option orders, degradation "
            "transforms, 900-item manifest, generated degraded images. "
            "NOT reused: the previous native-processor behaviour tables, because "
            "the processor budget changed."),
        "quality_x_relevance_is_factorial": (
            "For every quality level q both matched(q) and mismatch(q,k) exist and "
            "are built from the same _degraded(q, ...) call, so a degraded correct "
            "image is never compared against a clean donor."),
        "noimage_role": (
            "NoImage is an image-PRESENCE diagnostic only. It changes token "
            "presence, sequence length and the multimodal plumbing, so "
            "Clean - NoImage is never called content use. The content quantity is "
            "matched - mismatch at the same quality. Decomposition reported: "
            "U_content = U_presence - U_mismatch."),
        "donor_replicates": (
            "K=3 independent derangements; between-donor variance is reported "
            "rather than pooled away."),
        "gemma_boundary": (
            "Gemma 3 vs MedGemma is architecture-family / medical-specialisation "
            "robustness only, never a base->RL causal pair."),
        "terminology": (
            "Synthetic blur is technical/perceptual degradation, not a validated "
            "clinical diagnostic-quality score."),
    })
    print("written:", prov)


if __name__ == "__main__":
    main()
