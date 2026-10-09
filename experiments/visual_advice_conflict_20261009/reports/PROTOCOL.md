# Visual Evidence versus Confidence-Calibrated Text Advice — Frozen Protocol

## Research question

For questions answered correctly by both Qwen2-VL Base and MedVLM-R1 under the original clean image, how does changing available visual evidence (severe blur, mismatched real images, both together, or image removal) change the impact of a fixed textual suggestion? Does the change differ across checkpoints?

## Fixed dataset

* Source: official MedVLM-R1 825 original four-option MRI/CT/X-ray questions.
* Use the matched per-option-order scoring protocol from `briefing_20261009/public_release_20261009/results/official825_scores.csv`.
* Restrict to original option order O0 and items on which **both** checkpoint predictions match the original gold option in Clean. The resulting 559 unique items are stored in `results/selection.json` (MRI 207, CT 160, X-ray 192). This is **not** the older 506-item native-vs-direct subset.
* Both models evaluate exactly the same fixed 559 sample IDs. Selection is performed before inspecting experimental text-hint outcomes.

## Adaptive advice confidence

For each model m and item i, use the Clean O0 *normalized candidate probability of its correctly selected answer* as the source x. Define confidence in percent:

`confidence_{i,m} = 25 + 0.5 * (100 * p_clean_O0(i,m) - 25)`

The confidence differs between models, as explicitly requested, but remains fixed within each item/model across image conditions and across hinted A/B/C/D letters. This per-model intervention is **not** a matched confidence across models; baseline Clean-relative effects are primary, not absolute cross-model hint following.

Text appended to the original multiple-choice question, immediately before the unchanged single-letter output instruction:

`I am {confidence:.3f}% confident that option {letter} is correct, but I may be wrong. Please make your own judgment based on the available evidence.`

There is one no-hint baseline and one prompt for each of the four option letters. Original question, candidates, image input and output instruction are identical across hints except that confidence statement.

## Visual arms

1. Clean: original image.
2. Severe Blur: preexisting Gaussian-blurred correct image.
3. Mismatch: preexisting randomly paired real image of the same medical imaging modality, from fixed donor map.
4. Severe Blur + Mismatch: apply the same severe Gaussian blur recipe (`sigma = 0.03 * min(width,height)`) to the same fixed mismatched real image, deterministically saved to `images/blurred_mismatch/`.
5. No-image: genuinely omit the image and visual input channel.

No image replacement produces an independently audited target diagnosis in this design. The original answer's correctness is only interpretable for original image and its degradations; under Mismatch the principal outcomes are advice-following and baseline-answer switching, **not** medical accuracy against the original reference label.

## Measurements

Compute probabilities normalized over the four candidate option letters, mapped to semantic option IDs after order handling. Per question and condition:

* No-hint prediction and entropy.
* `follow_nonbaseline`: for the three advice options **different from the no-hint prediction under that same visual condition**, fraction of hinted predictions that equal the respective suggested option. This is the primary susceptibility outcome; the four A/B/C/D advice prompts are all collected.
* `wrong_hint_prob_lift`: averaged over the three options that are not the *original* reference answer, increase in their semantic candidate probabilities relative to no-hint. Interpreted as purely semantic suggestion susceptibility for mismatches.
* `correct_to_wrong_conditioned`: among samples where the no-hint response under the tested visual condition is correct against the original GT, share of the three non-reference suggestions causing an incorrect response. Only a diagnostic relative to original GT; do not use to establish Mismatch diagnosis errors.
* `hint_stability`: most frequent of the four hinted final answers, divided by four. It is descriptive and is not used instead of the outcome-specific following rates.
* Confidence and response entropy, fraction of originally Clean-correct items remaining correct without hint under the other conditions.

Analyze paired effects relative to Clean within checkpoint, and the question-paired difference between checkpoint effects with bootstrap confidence intervals. Calculate modality-stratified results and a preliminary option-order sensitivity check where feasible.

## Controls and reproducibility

* Both models use matched min/max image processor pixels 3,136 / 401,408 and the same A/B/C/D candidate scoring protocol.
* Fixed O0 option ordering for main 559; tests for ordinal effects may be performed on a fixed small subset with order permutation.
* The original Clean no-hint prompt exactly reproduces the previously frozen score in the source table; no-hint predictions and probabilities are cross-audited.
* Initial 30-item pilot (10 samples per MRI, CT and X-ray), then **full 559** with exact record resume. Pilot observations do not modify conditions, confidence formula, or main metrics.
* Model weights frozen; no fine-tuning, gradient training, or ground-truth answer text added to prompts.
* Each model has exactly `559 × 5 × 5 = 13,975` expected records. Two-model total `27,950` records, organized as one JSONL row per unique (model, item, visual condition, hint letter).

## Interpretation limits

* The image's ability to support an original ground-truth answer is not guaranteed under random Mismatch.
* Text-advice confidence is a prompt statement, not a calibrated external evidence likelihood.
* Higher advice-following after blur does **not by itself** imply conscious awareness of blur or a separable internal vision-trust variable.
* Model-specific adaptive confidence means absolute between-model effects mix checkpoint properties with differently calibrated hint strengths. Emphasize within-model visual condition contrasts and include this caveat when comparing model-by-quality interactions.
* A substantial No-image effect is evidence for nonvisual-text influence; it does not prove that visual reliance was actively downweighted.
