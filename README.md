# Phantom fairness: a reproducible audit of how automatically extracted labels can conceal demographic disparities in chest radiograph classifiers

A reproducible, CPU-only pipeline that scores the same chest radiographs against labels extracted from
radiology reports by natural language processing (NLP) and against a radiologist-adjudicated reference
standard. It asks whether a fairness audit based on NLP labels understates subgroup disparities, and what
kind of label noise is responsible.

> **Research paper:** *Phantom Fairness: A Reproducible Audit of How Automatically Extracted Labels Can
> Conceal Demographic Disparities in Chest Radiograph Classifiers.* Authors and journal withheld during
> peer review.

---

## Overview

Fairness audits of chest radiograph classifiers usually score predictions against the labels that ship
with the dataset. For NIH ChestX-ray14, those labels were extracted automatically from free-text reports,
so the reference against which fairness is judged is itself noisy. The pipeline:

1. trains one logistic classifier per finding on frozen image embeddings, using the NLP labels, as a model
   built in the usual way would be;
2. scores the **same** 4,376 held-out images (1,695 patients) against the NLP labels and against the
   radiologist-adjudicated labels of Majkowska et al. (2020) for four findings, with patient-level
   (cluster) bootstrap inference;
3. tests whether the NLP labeller misses the positives that classifiers score lowest (instance-dependent
   noise), using classifiers trained on NLP labels and scores that never saw them;
4. injects noise of the measured size into the reference-standard labels (random, calibrated and
   hardest-case arms) to show what each kind of noise does to an audit, with a patient-level bootstrap of
   the whole procedure and an out-of-sample check;
5. estimates how many adjudicated patients an audit needs to size a subgroup disparity.

## Key findings

- **The NLP labels miss many positives, at rates that differ between subgroups.** They missed 42% to 76%
  of the positives identified by radiologists.
- **The noise is instance-dependent.** NLP-missed positives scored lower than NLP-caught positives in all
  12 tests with classifiers trained on NLP labels (rank-biserial r 0.10 to 0.24) and in 14 of 15 tests
  with NLP-independent scores (classifiers trained on the adjudicated labels with patient-level
  cross-fitting, and the native outputs of a DenseNet-121 trained only on CheXpert). The exception was
  the zero-shot nodule or mass score, whose output barely discriminated the finding (AUROC 0.54).
- **Two-sided sensitivity analysis.** The main tests are one-sided by design. With two-sided tests, airspace
  opacity remained significant with all nine score sources and nodule or mass with seven of eight trained
  classifiers; pneumothorax, the finding with the fewest positives, remained significant with four of
  nine (`scripts/two_sided_sensitivity.py`).
- **Concealment is plausible but not yet sized.** Subgroup gaps were larger on adjudicated labels in 16
  of 24 comparisons, none significantly after correction. For pneumothorax in women older than 60 years,
  the true-positive-rate gap was 0.184 on adjudicated labels and 0.064 on NLP labels (difference 0.120;
  95% CI −0.074 to 0.235). Noise injection shows that noise of the measured size can conceal a gap of
  this size.
- **Reference sets must be sized for the question.** In this cohort, detecting a regime difference of
  0.10 with 80% power needed a median of about 2,437 adjudicated patients (range 569 to 9,524 across
  comparisons), against 1,694 available.

Every number is written by the notebooks to `outputs/tables/`, with a summary in
`outputs/tables/RESULTS_phantom_fairness.md` and `outputs/tables/REVISION_KEY_NUMBERS.json`.

## Repository structure

```
phantom-fairness/
├── notebooks/
│   ├── phantom_core.py                  # Shared, tested code: loaders, cohort, backbones, statistics, noise injection
│   ├── Phantom_Fairness_Part1.ipynb     # Data wiring, label mapping, exploratory analysis, embedding extraction
│   ├── Phantom_Fairness_Part2.ipynb     # Cohort and representativeness, NLP-trained classifiers, subgroup gaps, label noise, backbones, fracture
│   ├── Phantom_Fairness_Part3.ipynb     # Instance dependence, noise injection, mitigation, robustness
│   └── Phantom_Fairness_Part4.ipynb     # Analyses added in peer review
├── scripts/
│   └── two_sided_sensitivity.py         # Two-sided sensitivity analysis of the instance-dependence tests
├── outputs/
│   ├── figures/                         # PNG figures
│   ├── tables/                          # CSV tables, RESULTS_phantom_fairness.md, REVISION_KEY_NUMBERS.json
│   │   └── v1_original/                 # Tables of the original submission, archived on the first revision run
│   ├── provenance/                      # Per-part manifests, timings.json, hardware.json, combined MANIFEST
│   └── embeddings/                      # Embedding manifests only (n, dimension, identifier hash, timing); no arrays
├── requirements.txt
├── environment.yml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

Raw images, label files, embedding arrays and caches are not included (see Data availability). The
`.gitignore` keeps them out of the repository.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| **Part 1** | Loads ChestX-ray14, the adjudicated labels and the PadChest sample; builds the four-finding label space and the cross-dataset mapping; defines subgroups; exploratory analysis; registers the four backbones and extracts and caches embeddings deterministically. |
| **Part 2** | Rebuilds the patient-disjoint cohort and verifies it against the cached embeddings by SHA-256; cohort flow, clustering of images within patients and representativeness; trains classifiers on NLP labels; subgroup TPR and AUROC gaps on NLP versus adjudicated labels for the same images, with patient-level bootstrap intervals; subgroup flip rates; calibration; subgroup AUROCs; four-backbone comparison; fracture as the limiting case. |
| **Part 3** | Instance dependence with a patient-level bootstrap, within-cluster resampling and GEE models (unadjusted and adjusted for sex, age band, projection and follow-up number); noise injection with random, calibrated and hardest-case arms, a patient-level bootstrap and a sample-split check; threshold-based mitigation; robustness to the older-women age cut-off; exploratory PadChest check; Results text generated from the tables. |
| **Part 4** | Pretraining-exposure audit of each backbone; instance dependence with cross-fitted classifiers trained on the adjudicated labels and with a zero-shot CheXpert-only model; ontology and reference-standard sensitivity; reference-set size (analytic and simulated power); hyperparameter grid search; patient-grouped five-fold resampling; explainability (model of which positives the NLP labels miss, class activation maps); computational cost; key numbers. |

## Backbones

| Backbone | Pretraining data | Exposure to the evaluation data |
| --- | --- | --- |
| DenseNet-121, TorchXRayVision `densenet121-res224-all` (primary) | ChestX-ray14, Google adjudicated NIH labels, PadChest, CheXpert, MIMIC-CXR, OpenI, RSNA pneumonia (NIH images) | Exposure of evaluation images and labels cannot be ruled out |
| DenseNet-121, TorchXRayVision `densenet121-res224-chex` | CheXpert only | No documented training on NIH images or labels |
| ResNet-50, torchvision `IMAGENET1K_V2` | ImageNet-1k | No documented training on NIH images or labels |
| CLIP ViT-B/32, OpenAI weights | Web image-text pairs | Not trained on NIH or adjudicated labels; training images not documented at image level |

Cross-fitting the NLP-independent classifiers removes their dependence on the NLP labels, but not any
exposure of a frozen backbone during pretraining. The CheXpert-only backbone addresses that separate
concern.

## Data availability

All datasets are public and de-identified and are **not redistributed here**.

| Dataset | Source | Access |
| --- | --- | --- |
| NIH ChestX-ray14 (images and `Data_Entry_2017.csv`) | NIH Clinical Center | Open download |
| Radiologist-adjudicated labels (four findings) | Majkowska et al., *Radiology* 2020; the copy distributed with TorchXRayVision (`google2019_nih-chest-xray-labels.csv.gz`) is used | Open |
| PadChest (24-image public sample) | BIMCV | Registration and data use agreement |

The TorchXRayVision copy of the adjudicated labels contains 4,376 unique images (2,414 validation and
1,962 test), two more validation images than the original release reports. The notebooks audit this file
(`outputs/tables/table_adjudicated_file_checks.csv`).

## Environment

Python 3.11 under Anaconda, CPU only. The exact package versions of the reported run are recorded in
`outputs/provenance/provenance_part*.json`.

```bash
conda env create -f environment.yml
conda activate phantom-fairness

# or, with pip
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## How to reproduce

1. Download the data and point the notebooks at it by setting the environment variables `PF_NIH_ROOT`,
   `PF_PADCHEST_ROOT` and `PF_PROJECT_DIR`, or by editing the defaults in `phantom_core.Config`.
2. Keep `phantom_core.py` in the same folder as the notebooks.
3. Run Part 1, Part 2, Part 3 and Part 4 in order. Parts 2 to 4 reuse the cached embeddings, check the
   SHA-256 hash of every split's image list against the cache and stop if the cohort differs.
4. Run `python scripts/two_sided_sensitivity.py outputs/tables` to produce the two-sided sensitivity
   table (`table13c_two_sided_sensitivity.csv`).
5. For a quick dry run of the code paths, set `PF_QUICK=1`, which uses small resample counts. Do not
   report those numbers.

Approximate CPU time on a laptop: the first extraction of each backbone takes hours and is cached once.
With embeddings cached, the minimal paired audit (the patient-level analysis of all 24 regime
differences) takes about 80 s; Parts 2 and 3 take minutes, and Part 4 under an hour, most of it the
hyperparameter grid, the cross-fitted classifiers and the power simulation. Measured times are written to
`outputs/provenance/timings.json` and summarised in `outputs/tables/table20*_*.csv`.

## Reproducibility and reporting

- Single seed (1997); embeddings cached under a SHA-256 hash of the ordered image identifiers; every input
  file, table and figure hashed into `outputs/provenance/MANIFEST_phantom_fairness.json`, together with
  the hash of `phantom_core.py`.
- Patient-level splitting; the adjudicated labels are never used to train or set thresholds for the
  classifiers trained on NLP labels.
- Point estimates are computed over images; confidence intervals and primary tests resample patients.
- Reporting follows TRIPOD+AI and MI-CLAIM; the checklists are in the paper's supplementary materials.

## Ethics

Publicly available, de-identified data were used under their terms of use; no additional ethical review
was required.

## Citation

Citation details will be added after peer review.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences and are not relicensed or redistributed.
