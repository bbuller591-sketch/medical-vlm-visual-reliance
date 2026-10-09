# Environment provenance

The original experiments were carried out using an isolated GPU conda environment. The following installed versions were independently read on the source server on 2026-10-09:

| Package | Observed version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.7.1+cu128 |
| Transformers | 4.53.3 |
| qwen-vl-utils | 0.0.14 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| Pillow | 12.3.0 |
| scikit-learn | 1.9.1 |
| huggingface-hub | 0.36.2 |

This provenance is recorded separately from the *compatible* ranges in `requirements-*.txt`, because CUDA toolkit, Linux driver, and platform determine which PyTorch wheel can actually be installed. For strict numerical regeneration, aim to use these package versions and the commit-pinned model/dataset assets; if they cannot be installed on a different platform, report the deviations rather than silently replacing model/processor configurations. No claim of bitwise GPU reproducibility is made across different GPU generations or kernel variants.
