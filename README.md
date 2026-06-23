# Phantom fairness: a reproducible audit of instance-dependent label noise in chest radiograph classifiers

A reproducible, CPU-only pipeline that tests whether the apparent demographic fairness of chest
radiograph classifiers trained on NIH ChestX-ray14 is real or an artefact of how the labels were
generated, and identifies the operative noise mechanism.

> **Paper:** *Phantom fairness: a reproducible audit of instance-dependent label noise as a hidden
> source of demographic bias in chest radiograph classifiers.* Patel K., Beedala P., Vora D.,
> Mehta H. 

---

## Overview

Fairness audits of chest radiograph classifiers almost always score predictions against the labels
that ship with the dataset. For NIH ChestX-ray14, those labels were mined automatically from
free-text reports by a natural language processing (NLP) pipeline, so the reference standard is
itself noisy. This project asks whether that noise can manufacture an illusion of demographic
equity, and through which mechanism.

The pipeline scores the **same** held-out images under three label regimes and isolates the effect
of the labels from the effect of the model by using frozen image embeddings with per-finding linear
heads:

1. **NLP-mined** ChestX-ray14 labels (noisy),
2. **Radiologist-adjudicated** labels for four findings (Majkowska et al., 2020),
3. **PadChest** manually labelled samples from a second continent (exploratory).

The central experiment contrasts **class-conditional** against **instance-dependent** label-noise
injection to identify which mechanism reproduces the observed concealment.

## Key findings

- The labelling noise in ChestX-ray14 is **instance-dependent**, not class-conditional: among
  adjudicated positives, the model scores NLP-missed positives below NLP-caught positives in every
  finding (median rank-biserial 0.20; maximum adjusted p 0.014), replicated on two backbones never
  exposed to the clean labels (median 0.16; maximum adjusted p 0.038).
- Injecting the measured marginal noise at the model's lowest-scored positives reproduces the
  concealment (median 0.081), whereas the same marginal noise applied at random does not (0.057).
- The regime difference between clean-label and NLP-label gaps is directionally consistent across
  24 finding-by-axis comparisons (16 of 24 widen; median +0.021) but individually underpowered at
  the adjudicated sample size, so it is reported as a demonstrated mechanism rather than a precisely
  sized effect.
- A subgroup-specific threshold remedy can be targeted only where the disparity is visible on
  adjudicated labels, so mitigation depends on a clean audit set.

## Repository structure

```
phantom-fairness-cxr/
├── notebooks/
│   ├── Phantom_Fairness_Part1.ipynb     # Data wiring, label mapping, EDA, embedding extraction
│   ├── Phantom_Fairness_Part2.ipynb     # Fairness metrics, group-dependent noise, instance-dependence, backbones
│   └── Phantom_Fairness_Part3.ipynb     # Causal noise injection, decomposition, remedy, robustness, cross-continental
├── outputs/
│   ├── figures/                         # 12 PNG figures (4 main + 8 supplementary)
│   ├── tables/                          # 18 result tables (CSV) + RESULTS_phantom_fairness.md
│   ├── provenance/                      # Per-stage provenance manifests (hashes, parameters)
│   └── embeddings/                      # Embedding manifests only (n, dim, hash, timing) — no arrays
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
└── README.md
```

> Raw images, label files, frozen-embedding arrays, and caches are intentionally **not** included
> (see Data availability). The `.gitignore` blocks them from being committed by accident.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| **Part 1** | Loads ChestX-ray14, the Google adjudicated labels, and the PadChest sample; builds the canonical four-finding label space; performs patient-disjoint splitting; runs exploratory data analysis; extracts and caches frozen embeddings for each backbone. |
| **Part 2** | Trains per-finding logistic heads on NLP labels; computes equal-opportunity, equalised-odds, demographic-parity and AUROC gaps with bootstrap intervals; estimates group-dependent flip rates; quantifies instance-dependence (rank-biserial) across all three backbones. |
| **Part 3** | Runs the class-conditional versus instance-dependent noise-injection experiment with a dose-response sweep; decomposes the observed concealment; evaluates the equal-opportunity remedy; tests robustness to the elderly-women age cut; reports the exploratory cross-continental (PadChest) analysis. |

The notebooks are designed to run end to end on a **CPU-only laptop** (developed on a Windows
i7 machine under Anaconda). The global seed is fixed at **1997**, embeddings are cached
deterministically, and every input is hashed (SHA-256) so the reported numbers are tied by hash to
the files that produced them.

## Data availability

All three datasets are publicly available and de-identified. They are **not redistributed here**;
obtain them from the original sources and place them under a local `./data/` directory (ignored by
git).

| Dataset | Source | Access |
| --- | --- | --- |
| NIH ChestX-ray14 (images + `Data_Entry_2017.csv`) | NIH Clinical Center | Open download |
| Radiologist-adjudicated labels (4 findings) | Majkowska et al., *Radiology* 2020 (Google additional labels for ChestX-ray14) | Open download (`google2019_nih-chest-xray-labels.csv.gz`) |
| PadChest (sample used here) | BIMCV, University of Alicante / Hospital San Juan | Available on registration / data use agreement |

No PhysioNet credentialing is required for these datasets. The repository shares **code and
aggregate outputs only**; it does not contain any patient images or image-level data.

## What is and is not reproducible from this repository

- **Included:** all analysis code (the three notebooks), all aggregate result tables and figures,
  and the provenance manifests that document the run.
- **Not included (regenerate locally):** raw images, label CSVs, frozen-embedding arrays
  (`.npy` / `.parquet`), and intermediate caches. Re-running Part 1 against the downloaded data
  regenerates these locally.

## Environment and installation

Developed with Python 3.10 under Anaconda, CPU-only.

```bash
# Option A: conda
conda env create -f environment.yml
conda activate phantom-fairness

# Option B: pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `torchxrayvision`, `open-clip-torch` (CLIP ViT-B/32),
`scikit-learn`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `pyarrow`, `jupyter`.

## How to reproduce

1. Download the datasets above into `./data/` (NIH images, `Data_Entry_2017.csv`, the Google
   adjudicated labels CSV, and the PadChest sample).
2. Open `notebooks/Phantom_Fairness_Part1.ipynb` and run all cells to wire the data, build the
   label space, split by patient, and cache the embeddings.
3. Run `Phantom_Fairness_Part2.ipynb` for the fairness metrics, group-dependent noise, and
   instance-dependence analysis.
4. Run `Phantom_Fairness_Part3.ipynb` for the injection experiment, decomposition, remedy, and
   robustness checks.
5. Outputs are written to `outputs/figures/`, `outputs/tables/`, and `outputs/provenance/`.

Run order matters: Part 2 and Part 3 consume the cached embeddings and splits produced by Part 1.

## Outputs

- **`outputs/figures/`** — the phantom-fairness scatter, the group-dependent noise heatmap, the
  instance-dependent versus class-conditional dose-response, the threshold sweep, and supplementary
  reliability/ROC/PR/remedy plots.
- **`outputs/tables/`** — cohort and label-regime summary, the regime-difference family, the
  instance-dependence statistics across backbones, the noise decomposition, calibration, bias
  amplification, the remedy, robustness, and the exploratory cross-continental tables, plus a
  human-readable `RESULTS_phantom_fairness.md` summary.
- **`outputs/provenance/`** — per-stage manifests recording parameters, seeds, and SHA-256 hashes.

## Reproducibility and reporting

- Single global seed (1997); deterministic embedding cache keyed by a SHA-256 digest of the ordered
  image identifiers.
- Clean (adjudicated) labels are held out of all fitting and thresholding; the noisy-versus-clean
  contrast is confined to identical images.
- Reporting follows the TRIPOD+AI recommendations for prediction-model studies that use machine
  learning.

## Ethics

This study uses publicly available, de-identified datasets and did not require additional ethical
review or informed consent. All analyses were conducted in accordance with the relevant dataset
terms of use.

## Citation

If you use this code or the auditing protocol, please cite:

```bibtex
@article{patel_phantom_fairness_2026,
  title   = {Phantom fairness: a reproducible audit of instance-dependent label noise
             as a hidden source of demographic bias in chest radiograph classifiers},
  author  = {Patel, Krutarth and Beedala, Phanindra and Vora, Darshit and Mehta, Harshil},
  journal = {Computer Methods and Programs in Biomedicine},
  year    = {2026},
  note    = {Under review},
  doi     = {TO-BE-ADDED}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The datasets retain their own licences and
terms of use; this repository does not relicense or redistribute them.

## Contact

Krutarth Patel — ORCID [0009-0002-8748-8098](https://orcid.org/0009-0002-8748-8098).
Issues and questions: please open a GitHub issue.
