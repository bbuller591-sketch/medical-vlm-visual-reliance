#!/usr/bin/env python
"""Shared library for quality_reliance_v1  (Quality x Relevance factorial).

This project executes `medical_vlm_quality_reliance_deepseek_bundle`.

DESIGN DECISIONS, AND WHY (all recorded in provenance/design_decisions.json):

1. REUSE OF THE FROZEN INSTRUMENT.
   The frozen MCQ prompt, option-order logic, option permutations, degradation
   transforms and the 900-item manifest are imported read-only from
   `official900_visual_reliance_v1/scripts/common.py`.  Nothing here rewrites
   them.  Spec amendment H forbids silently rewriting the prompt.

2. MATCHED PROCESSOR FOR ALL ARMS  (this is a DEVIATION from the previous run).
   Measured preprocessor_config.json:
       Qwen2-VL-2B-Instruct  max_pixels = 12,845,056
       MedVLM-R1             max_pixels =    401,408
   Under native settings a 4240x3480 image gives Qwen ~30x the visual tokens
   MedVLM-R1 gets, so a cross-model difference in "visual reliance" would be
   partly a difference in how many tokens the model was handed.  Every arm in
   THIS project therefore runs with min_pixels=3136 / max_pixels=401408 for both
   checkpoints.  Consequence: the matched arms of this run are NOT numerically
   comparable with the older native-processor M2 run, so they are re-run here
   rather than reused.  The older native numbers stay available as a robustness
   comparison and are labelled as such.

3. QUALITY LADDER IS REUSED, NOT REDEFINED.
   Spec amendment F recommends mild = 1/4 severe, medium = 1/2 severe, but adds
   "unless the existing code's parameterization makes this invalid".  The existing
   frozen ladder is frac in {0.005, 0.015, 0.030} (1/6, 1/2, 1 of severe), was
   frozen before any model output was seen, and its severe level is required to
   be preserved exactly.  It is therefore reused verbatim so that this run stays
   comparable with the previous one.  No level was chosen by looking at accuracy.

4. IMAGES ARE NOT REGENERATED.
   `generated_images/blur_{mild,medium,severe}/<sample_id>.png` is by
   construction the degradation of that sample's own clean image.  A mismatch
   donor at quality q is another item's clean image degraded at q, i.e. exactly
   the file that already exists under the donor's own sample_id.  So matched(q)
   and mismatch(q) provably receive the SAME transform (amendment C), and the
   whole factorial needs no new image generation.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]                 # quality_reliance_v1
MED = ROOT.parent                                          # <LOCAL_PROJECT_ROOT>/medical
OLD = MED / "official900_visual_reliance_v1" if (MED / "official900_visual_reliance_v1").is_dir() else MED / "official900_visual_reliance_v1"
V2 = MED / "visual_reliance_mechanism_v2"

sys.path.insert(0, str(OLD / "scripts"))
import common as OD   # noqa: E402  frozen design, imported read-only

for _d in ("configs", "provenance", "manifests", "scripts", "status", "logs",
           "features", "probes", "causal", "plots", "reports", "results",
           "results/raw", "runtime"):
    (ROOT / _d).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Frozen design, re-exported
# --------------------------------------------------------------------------- #
GLOBAL_SEED = 20260917
OPTION_ORDERS = OD.OPTION_ORDERS
apply_order = OD.apply_order
build_m2_prompt = OD.build_m2_prompt
build_mcq_text = OD.build_mcq_text
option_order_perm = OD.option_order_perm
stable_seed = OD.stable_seed
normalise_probs = OD.normalise_probs
entropy = OD.entropy
gt_margin = OD.gt_margin
js_divergence = OD.js_divergence
sha256_file = OD.sha256_file
sha256_text = OD.sha256_text
position_setup = OD.position_setup

# --------------------------------------------------------------------------- #
# Matched processor -- THE central cross-model control (see docstring §2)
# --------------------------------------------------------------------------- #
MATCHED_MIN_PIXELS = 3136
MATCHED_MAX_PIXELS = 401408
MATCHED_PROCESSOR_ID = f"matched_min{MATCHED_MIN_PIXELS}_max{MATCHED_MAX_PIXELS}"

# --------------------------------------------------------------------------- #
# Quality x Relevance design
# --------------------------------------------------------------------------- #
QUALITY_LEVELS = ["clean", "mild", "medium", "severe"]
# matched arm name for each quality level (names come from the frozen old design)
MATCHED_ARM = {"clean": "clean", "mild": "blur_mild",
               "medium": "blur_medium", "severe": "blur_severe"}
NOIMAGE_ARM = "no_image"
N_DONORS = 3


def mismatch_arm(quality: str, k: int) -> str:
    """Donor replicate k (1-based) at quality level `quality`."""
    return f"pm_{quality}_k{k}"


def semantic_arm(quality: str) -> str:
    """Semantic-conflict donor at quality level `quality`."""
    return f"sc_{quality}"


SEMANTIC_ARMS = [semantic_arm(q) for q in QUALITY_LEVELS]


MISMATCH_ARMS = [mismatch_arm(q, k) for q in QUALITY_LEVELS
                 for k in range(1, N_DONORS + 1)]
# Cross-corruption controls: the modality-specific degradations that already
# exist on disk from the previous run.  They are the "different corruption" the
# frozen-transfer test needs -- a probe trained to separate clean from blurred
# must not be able to read the same structure off a noise-degraded image.
CONTROL_ARMS = ["modality_specific_mild", "modality_specific_medium",
                "modality_specific_severe"]
# all arms that hand the model an image (matched + mismatch); no_image is separate
IMAGE_ARMS = ([MATCHED_ARM[q] for q in QUALITY_LEVELS] + MISMATCH_ARMS
              + SEMANTIC_ARMS + CONTROL_ARMS)
ARM_ORDER = ([MATCHED_ARM[q] for q in QUALITY_LEVELS] + MISMATCH_ARMS
             + SEMANTIC_ARMS + CONTROL_ARMS + [NOIMAGE_ARM])

# arm -> (quality_level, relevance_type)
ARM_SPEC: dict[str, tuple[str, str]] = {}
for _q in QUALITY_LEVELS:
    ARM_SPEC[MATCHED_ARM[_q]] = (_q, "matched")
for _q in QUALITY_LEVELS:
    for _k in range(1, N_DONORS + 1):
        ARM_SPEC[mismatch_arm(_q, _k)] = (_q, "plausible_mismatch")
for _q in QUALITY_LEVELS:
    ARM_SPEC[semantic_arm(_q)] = (_q, "semantic_conflict")
for _a in CONTROL_ARMS:
    ARM_SPEC[_a] = (_a.rsplit("_", 1)[-1], "modality_specific_control")
ARM_SPEC[NOIMAGE_ARM] = (None, "noimage")

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
MODELS = {
    "qwen2vl": MED / "models" / "Qwen2-VL-2B-Instruct",
    "medvlmr1": MED / "models" / "MedVLM-R1",
}
MODEL_DISPLAY = {"qwen2vl": "Qwen2-VL-2B-Instruct", "medvlmr1": "MedVLM-R1"}

# --------------------------------------------------------------------------- #
# Time / io
# --------------------------------------------------------------------------- #
def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def load_jsonl(p) -> list:
    p = Path(p)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def append_jsonl(p, rec: dict) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def dedupe_load_glob(directory, patterns) -> list:
    """Union glob patterns BY FILE NAME before reading (guards the M2_/M2__ bug)."""
    files = sorted({p for pat in patterns for p in Path(directory).glob(pat)})
    out = []
    for p in files:
        out += load_jsonl(p)
    return out


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
LEDGER = ROOT / "status" / "experiment_ledger.csv"
LEDGER_COLS = ["experiment_id", "priority", "dataset", "model", "condition",
               "sample_count_planned", "sample_count_done", "gpu_id", "status",
               "start_time", "end_time", "config_path", "output_path",
               "blocker_or_error"]


def ledger_upsert(experiment_id: str, **fields) -> None:
    rows = []
    if LEDGER.is_file():
        with open(LEDGER, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    for r in rows:
        if r.get("experiment_id") == experiment_id:
            r.update({k: ("" if v is None else str(v)) for k, v in fields.items()})
            break
    else:
        row = {c: "" for c in LEDGER_COLS}
        row["experiment_id"] = experiment_id
        row.update({k: ("" if v is None else str(v)) for k, v in fields.items()})
        rows.append(row)
    tmp = LEDGER.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LEDGER_COLS})
    tmp.replace(LEDGER)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_ITEMS: dict | None = None
_DERANG: dict | None = None


def load_items() -> dict:
    """The 900 official items, verbatim from the frozen manifest."""
    global _ITEMS
    if _ITEMS is None:
        man = read_json(OLD / "manifests" / "dataset_manifest.json")
        _ITEMS = {it["sample_id"]: it for it in man["items"]}
    return _ITEMS


def load_derangements() -> dict:
    """{'D1': {'assignment': {target: donor}, 'fixed_points': 0, ...}, ...}"""
    global _DERANG
    if _DERANG is None:
        p = ROOT / "manifests" / "donor_derangements.json"
        _DERANG = read_json(p) if p.is_file() else {}
    return _DERANG


def donor_of(sample_id: str, k: int) -> str | None:
    d = load_derangements()
    e = (d.get("derangements") or {}).get(f"D{k}")
    if not e:
        return None
    return e["assignment"].get(sample_id)


def subgroup_of(sample_id: str, k: int) -> int | None:
    """Closed-subgroup id for the donor of `sample_id` under replicate k.

    The derangement is built inside these subgroups, so any union of subgroups is
    closed under it and therefore has identical matched and mismatch image
    multisets.  Probe folds are built as unions of subgroups so this holds on the
    training set, not merely on all 900.
    """
    d = load_derangements()
    e = (d.get("derangements") or {}).get(f"D{k}")
    if not e:
        return None
    return (e.get("subgroup_of") or {}).get(sample_id)


def load_subgroups(k: int) -> dict:
    d = load_derangements()
    e = (d.get("derangements") or {}).get(f"D{k}") or {}
    return e.get("subgroup_of") or {}


def source_image_path(item: dict) -> Path:
    """The clean image, as the old project resolved it."""
    return (MED / "official900_replication" / "data" / "images_extracted" / "images"
            / item["image_file"])


def _degraded(quality: str, sample_id: str) -> Path | None:
    """Existing frozen degradation of `sample_id`'s own clean image at `quality`."""
    if quality == "clean":
        it = load_items().get(sample_id)
        return source_image_path(it) if it else None
    return OLD / "generated_images" / MATCHED_ARM[quality] / f"{sample_id}.png"


def image_path(arm: str, item: dict) -> Path | None:
    """Resolve the image for one arm.  None == GENUINE no-image.

    A blank/white image is a different control and is never substituted here.
    For a mismatch arm the donor's image is fetched through the same
    `_degraded(quality, ...)` used for the matched arm, which is what makes
    amendment C (identical transform on both arms) true by construction.
    """
    if arm == NOIMAGE_ARM:
        return None
    if arm in MATCHED_ARM.values():
        q = [q for q in QUALITY_LEVELS if MATCHED_ARM[q] == arm][0]
        return _degraded(q, item["sample_id"])
    if arm in CONTROL_ARMS:
        return OLD / "generated_images" / arm / f"{item['sample_id']}.png"
    if arm.startswith("sc_"):
        q = arm[3:]
        d = load_semantic_conflict().get("assignment", {}).get(item["sample_id"])
        if not d:
            return None
        return _degraded(q, d["donor"])
    if arm.startswith("pm_"):
        _, q, kk = arm.split("_")
        k = int(kk[1:])
        donor = donor_of(item["sample_id"], k)
        if donor is None:
            return None
        return _degraded(q, donor)
    return None


_SEM: dict | None = None


def load_semantic_conflict() -> dict:
    global _SEM
    if _SEM is None:
        p = ROOT / "manifests" / "semantic_conflict.json"
        _SEM = read_json(p) if p.is_file() else {}
    return _SEM


def semantic_record(sample_id: str) -> dict | None:
    """The frozen semantic-conflict donor for one target, with its tier and rule.

    Recorded on every row so that a later reader can separate Tier 1 (mutually
    exclusive: opposite polarity, or a question with a unique correct attribute)
    from Tier 2 (same question, different finding, which can co-occur) without
    re-deriving the classification.
    """
    d = load_semantic_conflict().get("assignment", {}).get(sample_id)
    if not d:
        return None
    items = load_items()
    return {"donor_sample_id": d["donor"],
            "donor_image_sha256": items[d["donor"]]["source_image_sha256"],
            "donor_semantic_gt": items[d["donor"]].get("gt_text"),
            "donor_modality": items[d["donor"]]["modality"],
            "semantic_conflict_tier": d["tier"],
            "semantic_conflict_rule": d["rule"]}


def donor_record(sample_id: str, k: int) -> dict | None:
    donor = donor_of(sample_id, k)
    if donor is None:
        return None
    items = load_items()
    return {"donor_sample_id": donor,
            "donor_image_file": items[donor]["image_file"],
            "donor_image_sha256": items[donor]["source_image_sha256"],
            "donor_modality": items[donor]["modality"],
            "donor_semantic_gt": items[donor].get("gt_text"),
            "donor_replicate": f"D{k}",
            "closed_subgroup": subgroup_of(sample_id, k)}


# --------------------------------------------------------------------------- #
# GPU helpers -- the machine is shared, so "free" must be measured, not assumed
# --------------------------------------------------------------------------- #
def gpu_stats() -> list[dict]:
    """Per-GPU used/total MiB and utilisation, plus the owning users.

    The owner list matters: a GPU with free memory but an active foreign process
    is not a GPU this job should saturate.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=True, timeout=30)
    except Exception as exc:                                    # pragma: no cover
        return [{"error": str(exc)}]
    stats = []
    for ln in out.stdout.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            continue
        stats.append({"index": int(parts[0]), "used": int(parts[1]),
                      "total": int(parts[2]), "util": int(parts[3])})
    # map pid -> gpu index to name owners
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=True, timeout=30).stdout
        uu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30).stdout
        uuid2idx = {}
        for ln in uu.splitlines():
            i, u = [x.strip() for x in ln.split(",")]
            uuid2idx[u] = int(i)
        owners: dict[int, set] = {}
        for ln in q.splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) < 3:
                continue
            idx = uuid2idx.get(parts[0])
            if idx is None:
                continue
            try:
                user = subprocess.run(["ps", "-o", "user=", "-p", parts[1]],
                                      capture_output=True, text=True,
                                      timeout=10).stdout.strip()
            except Exception:
                user = "?"
            owners.setdefault(idx, set()).add(user or "?")
        for s in stats:
            s["users"] = sorted(owners.get(s["index"], []))
    except Exception:
        for s in stats:
            s.setdefault("users", [])
    for s in stats:
        s.setdefault("free", s.get("total", 0) - s.get("used", 0))
    return stats


ME = os.environ.get("USER") or "zhuangcy"


def is_free(s: dict, need_free_mib: int, max_util: int = 10) -> bool:
    """A GPU is usable when it has room AND nobody else is actively computing on
    it.  Both conditions are required: taking a GPU that a foreign job is about
    to grow into would evict that job."""
    if "error" in s:
        return False
    others = [u for u in s.get("users", []) if u and u != ME]
    if others:
        return False
    return s["free"] >= need_free_mib and s["util"] <= max_util


def pick_gpus(n: int, need_free_mib: int, max_util: int = 10) -> list[int]:
    cand = [s for s in gpu_stats() if is_free(s, need_free_mib, max_util)]
    cand.sort(key=lambda s: -s["free"])
    return [s["index"] for s in cand[:n]]


def my_gpu_mib(pids: set) -> dict:
    """Map pid -> MiB of GPU memory that pid itself holds.

    Needed because the naive pressure test is wrong on a shared card: once we
    have allocated 8 GiB, `free` is small *because of us*, and a rule that looks
    only at `free` would instantly decide the card is under pressure and stop our
    own job.  The honest quantity is free memory that our own allocation is not
    responsible for, i.e. free + (memory held by our pids).
    """
    out = {}
    pids = {str(p) for p in pids}
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=True, timeout=30).stdout
    except Exception:
        return out
    for ln in q.splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) >= 2 and parts[0] in pids:
            try:
                out[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return out


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def prompt_hash(arm: str) -> str:
    """Hash of the exact prompt string the instrument will send (item-independent
    part is not enough -- the resolver is deterministic, so hash the builder's
    identity plus the frozen template)."""
    return OD.sha256_text(OD.build_m2_prompt("__PROMPT_TEMPLATE_PROBE__"))


def cuda_alloc_env() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
