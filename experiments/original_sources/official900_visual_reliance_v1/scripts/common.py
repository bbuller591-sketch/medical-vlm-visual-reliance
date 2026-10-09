#!/usr/bin/env python
"""Shared library for official900_visual_reliance_v1.

Everything frozen in this file is frozen on purpose. Conditions, parameters and
permutations are written once here and consumed by both the manifest builder and the
runner, so a parameter cannot drift between the images that were generated and the
record that describes them.

WHAT IS APPROXIMATE (must be repeated in every report)
  * MRI  : MRI_RICIAN_LIKE                 -- a magnitude-noise model, not any
                                              specific scanner or SNR.
  * CT   : CT_IMAGE_DOMAIN_POISSON_APPROXIMATION
                                           -- photon statistics applied to the
                                              already-reconstructed image. NOT a
                                              physically faithful low-dose CT
                                              simulation and NOT a scanner raw-data
                                              dose simulation. It must never be
                                              described as "low-dose CT at N mAs".
  * Xray : XRAY_PROCESSED_POISSON_GAUSSIAN -- applied to the processed image, not
                                              modelled at the detector.
  * Blur : cross-modality common degradation control, NOT a clinical artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
MED = ROOT.parent
REPL = MED / "official900_replication"

DATASET = REPL / "sources" / "MedVLM-R1_repo" / "MRI_CT_XRAY_300each_dataset.json"
SRC_IMAGES = REPL / "data" / "images_extracted" / "images"
PARENT_POOL = REPL / "sources" / "mm_eval_data.json"

# --------------------------------------------------------------------------- #
# Path resolution that also works from inside a delivery bundle.
#
# The sibling `official900_replication/` directory is where these live during
# development, but an extracted copy of this project must not depend on a directory
# that was not shipped. `sources/` inside this project is shipped with the bundles,
# so both locations are tried and the first that exists wins.
# --------------------------------------------------------------------------- #
LOCAL_SOURCES = ROOT / "sources"


def _first(candidates, what: str) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"{what} not found. Tried:\n" + "\n".join(f"  {c}" for c in candidates))


def dataset_json() -> Path:
    return _first([LOCAL_SOURCES / "MRI_CT_XRAY_300each_dataset.json", DATASET],
                  "the official 900 dataset json")


def parent_pool_json() -> Path:
    return _first([LOCAL_SOURCES / "mm_eval_data.json", PARENT_POOL],
                  "the parent multimodal pool json")


def src_images_dir() -> Path:
    return _first([LOCAL_SOURCES / "images", SRC_IMAGES],
                  "the 900 clean source images")

MODELS = {
    "qwen2vl": MED / "models" / "Qwen2-VL-2B-Instruct",
    "medvlmr1": MED / "models" / "MedVLM-R1",
}
MODEL_DISPLAY = {"qwen2vl": "Qwen2-VL-2B-Instruct", "medvlmr1": "MedVLM-R1"}

MOD_SLICE = {"MRI": (0, 300), "CT": (300, 600), "Xray": (600, 900)}

GLOBAL_SEED = 20260914
INSTRUMENT_VERSION = "m2-v1"

DIRS = {k: ROOT / k for k in
        ("configs", "manifests", "logs", "reports", "plots", "casebook",
         "donor_maps", "generated_images", "diagnostics")}
for _d in DIRS.values():
    _d.mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "raw").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "processed").mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Frozen degradation parameters
# --------------------------------------------------------------------------- #
MRI_RICIAN_SIGMA = {"mild": 0.01, "medium": 0.03, "severe": 0.05}

CT_POISSON_N = {"mild": 1000, "medium": 500, "severe": 100}

# Weaker than the previous calibration pass, per the v1 spec.
XRAY_PG_LAMBDA = {"mild": 5000.0, "medium": 1500.0, "severe": 500.0}
XRAY_PG_SIGMA_G = {"mild": 0.005, "medium": 0.012, "severe": 0.025}

# Reused verbatim from src/perturbations.py BLUR_SIGMA_FRACTION_MIN_SIDE, which was
# the already-validated relative-image-size ladder. sigma = frac * min(w, h).
BLUR_FRACTION_MIN_SIDE = {"mild": 0.005, "medium": 0.015, "severe": 0.030}

DEGRADATION_FAMILY = {
    "MRI": "MRI_RICIAN_LIKE",
    "CT": "CT_IMAGE_DOMAIN_POISSON_APPROXIMATION",
    "Xray": "XRAY_PROCESSED_POISSON_GAUSSIAN",
}

DEGRADATION_FORMULA = {
    "MRI_RICIAN_LIKE": "I' = sqrt((I + n1)^2 + n2^2),  n1,n2 ~ N(0, sigma^2)",
    "CT_IMAGE_DOMAIN_POISSON_APPROXIMATION": "I' = Poisson(I * N) / N",
    "XRAY_PROCESSED_POISSON_GAUSSIAN":
        "I' = Poisson(lambda * I) / lambda + eps,  eps ~ N(0, sigma_g^2)",
    "BLUR_GAUSSIAN": "I' = GaussianBlur(I, sigma),  sigma = frac * min(w,h)",
}

DEGRADATION_CAVEAT = {
    "MRI_RICIAN_LIKE": "Rician-like magnitude noise model; not a specific scanner, "
                       "SNR or clinical grade.",
    "CT_IMAGE_DOMAIN_POISSON_APPROXIMATION":
        "NOT a physically faithful low-dose CT simulation and NOT a scanner raw-data "
        "dose simulation. Exploratory CT noise stress test only; never describe as "
        "'low-dose CT at N mAs'.",
    "XRAY_PROCESSED_POISSON_GAUSSIAN":
        "Processed-image approximation, not a detector-level or clinical dose "
        "simulation.",
    "BLUR_GAUSSIAN": "Cross-modality common degradation control, not a clinical "
                     "artifact.",
}

# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
M2_CONDITIONS = [
    "clean",
    "no_image",
    "blank_image",
    "blur_mild", "blur_medium", "blur_severe",
    "modality_specific_mild", "modality_specific_medium", "modality_specific_severe",
    "random_same_modality_replacement",
    "same_answer_donor",
    "opposite_answer_donor",
    "ood_cross_modality",
]

M1_CONDITIONS = [
    "clean",
    "no_image",
    "blur_severe",
    "modality_specific_severe",
    "random_same_modality_replacement",
    "opposite_answer_donor",
    "ood_cross_modality",
]

M3_CONDITIONS = [
    "clean",
    "no_image",
    "blur_severe",
    "modality_specific_severe",
    "random_same_modality_replacement",
    "opposite_answer_donor",
    "ood_cross_modality",
]

SEVERITY_OF = {
    "clean": None, "no_image": None, "blank_image": None,
    "blur_mild": "mild", "blur_medium": "medium", "blur_severe": "severe",
    "modality_specific_mild": "mild",
    "modality_specific_medium": "medium",
    "modality_specific_severe": "severe",
    "random_same_modality_replacement": None,
    "same_answer_donor": None,
    "opposite_answer_donor": None,
    "ood_cross_modality": None,
}

# conditions whose image comes from a donor rather than from the source
REPLACEMENT_CONDITIONS = {
    "random_same_modality_replacement", "same_answer_donor",
    "opposite_answer_donor", "ood_cross_modality",
}


def condition_family(condition: str, modality: str) -> str | None:
    if condition.startswith("blur_"):
        return "BLUR_GAUSSIAN"
    if condition.startswith("modality_specific_"):
        return DEGRADATION_FAMILY[modality]
    return None


def condition_params(condition: str, modality: str) -> dict:
    sev = SEVERITY_OF.get(condition)
    if sev is None:
        return {}
    if condition.startswith("blur_"):
        return {"sigma_fraction_min_side": BLUR_FRACTION_MIN_SIDE[sev]}
    if condition.startswith("modality_specific_"):
        if modality == "MRI":
            return {"sigma": MRI_RICIAN_SIGMA[sev]}
        if modality == "CT":
            return {"N": CT_POISSON_N[sev]}
        return {"lambda": XRAY_PG_LAMBDA[sev], "sigma_g": XRAY_PG_SIGMA_G[sev]}
    return {}


# --------------------------------------------------------------------------- #
# Hashing / seeds
# --------------------------------------------------------------------------- #
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts) -> int:
    """Process-independent seed.

    Builtin hash() is salted per interpreter, which would give the two model workers
    different noise for the same (sample, condition, severity) -- the exact failure
    this whole experiment must not have.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def stable_rng(*parts) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


# --------------------------------------------------------------------------- #
# Text canonicalization (identical rule to donor_availability_audit.py)
# --------------------------------------------------------------------------- #
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("\u2019", "'").replace("\u2018", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-"))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def quality_flags(dataset: list[dict]) -> dict[int, list[str]]:
    """Cheap heuristics for items a reader should look at before trusting them.

    Lives here rather than in another project so a shipped bundle has no dependency
    on a directory that was not shipped. None of these flags affect any reported
    metric; they only drive the sensitivity analysis.
    """
    templates = Counter(r["question"] for r in dataset)
    flags: dict[int, list[str]] = {}
    for i, r in enumerate(dataset):
        f = []
        opts = [norm(o) for o in r["options"]]
        for a in range(len(opts)):
            for b in range(a + 1, len(opts)):
                x, y = opts[a], opts[b]
                if x == y or (len(x) > 6 and (x in y or y in x)):
                    f.append("dupopt")
                    break
            else:
                continue
            break
        if re.search(r"\bdataset\b|RadImageNet|\bmodality\b", r["question"], re.I):
            f.append("metaq")
        if templates[r["question"]] >= 10:
            f.append("tpl")
        if f:
            flags[i] = f
    return flags


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_official900() -> list[dict]:
    rows = json.loads(dataset_json().read_text())
    items = []
    for i, r in enumerate(rows):
        mod = next(m for m, (lo, hi) in MOD_SLICE.items() if lo <= i < hi)
        opts = list(r["options"])
        gt_idx = ord(r["solution"]) - 65
        items.append({
            "sample_id": f"{mod}_{i - MOD_SLICE[mod][0]:03d}",
            "file_index": i,
            "modality": mod,
            "image_file": r["image"][0].split("/")[-1],
            "question": r["question"],
            "problem": r["problem"],
            "options": opts,
            "gt_letter": r["solution"],
            "gt_index": gt_idx,
            "gt_text": opts[gt_idx],
            "gt_norm": norm(opts[gt_idx]),
            "question_norm": norm(r["question"]),
            "option_set_key": "\u0001".join(sorted(norm(o) for o in opts)),
            "n_options": len(opts),
            "letters": [chr(65 + k) for k in range(len(opts))],
            "answer_text": r["answer"],
        })
    return items


def build_mcq_text(question: str, options: list[str], letters: list[str]) -> str:
    """Same shape as the official `problem` field: question then A) .. B) .. lines."""
    return question + "\n" + "\n".join(f"{L}) {o}" for L, o in zip(letters, options))


# --------------------------------------------------------------------------- #
# Option orders
# --------------------------------------------------------------------------- #
OPTION_ORDERS = ["O0", "O1", "O2"]


def option_order_perm(n: int, order_id: str, sample_id: str) -> list[int]:
    """Return a permutation `perm` where display position k shows semantic option
    `perm[k]`.  O0 is the identity, O1 a deterministic cyclic rotation, O2 a seeded
    random permutation.

    For a two-option question only two permutations exist, so O2 necessarily
    coincides with O0 or O1; `n_distinct_orders_available` records that so a
    downstream check does not mistake arithmetic for a bug.
    """
    base = list(range(n))
    if order_id == "O0":
        return base
    rot = base[1:] + base[:1]
    if order_id == "O1":
        return rot
    if order_id == "O2":
        rng = stable_rng("option_order", sample_id, order_id, GLOBAL_SEED)
        p = base[:]
        for _ in range(200):
            rng.shuffle(p)
            if p != base and p != rot:
                return p
        return rot if rot != base else base
    raise ValueError(order_id)


def n_distinct_orders_available(n_options: int) -> int:
    return min(len(OPTION_ORDERS), math.factorial(n_options))


def apply_order(item: dict, order_id: str) -> dict:
    """Everything needed to render and to invert one option ordering."""
    n = item["n_options"]
    perm = option_order_perm(n, order_id, item["sample_id"])
    display_letters = [chr(65 + k) for k in range(n)]
    display_options = [item["options"][src] for src in perm]
    gt_display_idx = perm.index(item["gt_index"])
    return {
        "option_order_id": order_id,
        "permutation": perm,                       # display position -> semantic idx
        "display_letters": display_letters,
        "display_options": display_options,
        "display_to_semantic": {L: perm[k] for k, L in enumerate(display_letters)},
        "semantic_to_display": {perm[k]: display_letters[k] for k in range(n)},
        "gt_display_letter": display_letters[gt_display_idx],
        "mcq_text": build_mcq_text(item["question"], display_options,
                                   display_letters),
    }


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
MCQ_INSTRUCTION = ("Answer with the single letter of the correct option. "
                   "Reply with only that letter.")

ABSTAIN_OPTION_TEXT = "Cannot determine reliably from the provided image."
ABSTAIN_LETTER = "Z"


def build_m2_prompt(mcq_text: str) -> str:
    return f"{mcq_text}\n\n{MCQ_INSTRUCTION}"


def build_m3_prompt(mcq_text: str, abstain_placement: str) -> tuple[str, list[str]]:
    """Insert the abstention option at a deterministic placement.

    Returns (prompt, display_letters) where display_letters includes the abstain
    letter at its actual display position.
    """
    lines = mcq_text.split("\n")
    body, opts = lines[0], lines[1:]
    n = len(opts)
    if abstain_placement == "last":
        pos = n
    elif abstain_placement == "middle":
        pos = n // 2
    else:
        raise ValueError(abstain_placement)
    letters = []
    out = []
    k = 0
    for i in range(n + 1):
        if i == pos:
            L = ABSTAIN_LETTER
            out.append(f"{L}) {ABSTAIN_OPTION_TEXT}")
        else:
            L = chr(65 + k)
            out.append(opts[k].split(")", 1)[1].strip() if ")" in opts[k] else opts[k])
            out[-1] = f"{L}) {out[-1]}"
            k += 1
        letters.append(L)
    text = body + "\n" + "\n".join(out)
    return f"{text}\n\n{MCQ_INSTRUCTION}", letters


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #
def to_float01(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB")).astype(np.float64) / 255.0
    return a


def from_float01(a: np.ndarray) -> Image.Image:
    a = np.clip(a, 0.0, 1.0)
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def degrade_mri_rician(x: np.ndarray, sigma: float, rng) -> np.ndarray:
    n1 = rng.normal(0.0, sigma, x.shape)
    n2 = rng.normal(0.0, sigma, x.shape)
    return np.sqrt((x + n1) ** 2 + n2 ** 2)


def degrade_ct_poisson(x: np.ndarray, n_photons: float, rng) -> np.ndarray:
    return rng.poisson(x * n_photons) / n_photons


def degrade_xray_pg(x: np.ndarray, lam: float, sigma_g: float, rng) -> np.ndarray:
    return rng.poisson(x * lam) / lam + rng.normal(0.0, sigma_g, x.shape)


def blur_sigma(size: tuple[int, int], frac: float) -> float:
    return float(frac * min(size[1], size[0]))


def apply_degradation(img: Image.Image, condition: str, modality: str,
                      sample_id: str) -> tuple[Image.Image, dict]:
    """Return (degraded_image, provenance). Deterministic given the inputs."""
    sev = SEVERITY_OF[condition]
    family = condition_family(condition, modality)
    if family is None:
        raise ValueError(f"{condition} is not a degradation condition")

    if condition.startswith("blur_"):
        sigma = blur_sigma(img.size, BLUR_FRACTION_MIN_SIDE[sev])
        out = img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=sigma))
        return out, {"degradation_family": "BLUR_GAUSSIAN",
                     "degradation_params": {"sigma": sigma,
                                            "sigma_fraction_min_side":
                                                BLUR_FRACTION_MIN_SIDE[sev],
                                            "image_size": list(img.size)},
                     "seed": None}

    seed = stable_seed(sample_id, condition, sev, GLOBAL_SEED)
    rng = np.random.default_rng(seed)
    x = to_float01(img)

    if modality == "MRI":
        sigma = MRI_RICIAN_SIGMA[sev]
        out = from_float01(degrade_mri_rician(x, sigma, rng))
        params = {"sigma": sigma}
    elif modality == "CT":
        n_ph = CT_POISSON_N[sev]
        out = from_float01(degrade_ct_poisson(x, n_ph, rng))
        params = {"N": n_ph}
    else:
        lam, sg = XRAY_PG_LAMBDA[sev], XRAY_PG_SIGMA_G[sev]
        out = from_float01(degrade_xray_pg(x, lam, sg, rng))
        params = {"lambda": lam, "sigma_g": sg}

    return out, {"degradation_family": DEGRADATION_FAMILY[modality],
                 "degradation_params": params, "seed": seed}


def blank_image_like(img: Image.Image) -> Image.Image:
    return Image.new("RGB", img.size, (128, 128, 128))


# --------------------------------------------------------------------------- #
# Candidate scoring
# --------------------------------------------------------------------------- #
def position_setup(tokenizer, letters: list[str]) -> dict:
    """Token ids for the candidate labels, with an explicit multi-token check.

    Probed upstream: for this tokenizer every label A-J / Z is a single token with no
    leading space. The general path is still implemented so that a future tokenizer
    change cannot silently turn the shortcut into a wrong number.
    """
    ids = {}
    for L in letters:
        enc = tokenizer.encode(L, add_special_tokens=False)
        ids[L] = enc
    return {
        "label_token_ids": ids,
        "all_single_token": all(len(v) == 1 for v in ids.values()),
        "n_tokens": {L: len(v) for L, v in ids.items()},
    }


def normalise_probs(logprobs: dict[str, float]) -> dict[str, float]:
    m = max(logprobs.values())
    exps = {k: math.exp(v - m) for k, v in logprobs.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)


def gt_margin(p_semantic: dict[str, float], gt_key) -> float:
    """p(gt) - max p(j != gt). Signed, unlike an unsigned top1 gap."""
    if gt_key not in p_semantic:
        return float("nan")
    others = [v for k, v in p_semantic.items() if k != gt_key]
    return p_semantic[gt_key] - (max(others) if others else 0.0)


def js_divergence(p: dict, q: dict) -> float:
    keys = sorted(set(p) | set(q))
    eps = 1e-12
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64) + eps
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64) + eps
    pv /= pv.sum(); qv /= qv.sum()
    m = 0.5 * (pv + qv)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))   # noqa: E731
    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)


# --------------------------------------------------------------------------- #
# Paths / IO
# --------------------------------------------------------------------------- #
def gen_image_path(condition: str, sample_id: str) -> Path:
    return DIRS["generated_images"] / condition / f"{sample_id}.png"


def raw_path(track: str, model_label: str, worker_id: str) -> Path:
    return ROOT / "results" / "raw" / f"{track}__{model_label}__{worker_id}.jsonl"


def jsonl_append(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def read_json(path: Path):
    return json.loads(path.read_text())


def cuda_alloc_env() -> None:
    """Must run before torch builds its CUDA context. See official900_replication for
    the measured reason: the official max_pixels expands a few radiographs to ~16k
    visual tokens and the peak is ~22 GiB, which only fits with this set."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
