# Changelog

## Revision 2.1 (final revision package)

- Added `scripts/two_sided_sensitivity.py`, which derives two-sided p values for the instance-dependence
  tests from the saved one-sided results and re-applies the Benjamini-Hochberg adjustment within each
  family (manuscript Table S13c).
- README rewritten for the four-notebook pipeline, with the final key numbers, the backbone exposure
  table and the minimal-audit run time.
- Wording aligned with the manuscript: backbones are described by their documented training data, and
  CLIP's training images are noted as undocumented at image level.

## Revision 2 (peer review)

### Code organisation
- Shared code moved into `notebooks/phantom_core.py` (loaders, cohort, backbones, statistics, injection),
  replacing three diverging copies inside the notebooks. Unit-tested against the original implementation:
  with unit weights, the vectorised gaps and AUROCs are identical to the original pandas code.
- The cohort is rebuilt with the original code and seed and verified against the cached embeddings by
  SHA-256 before any analysis; cached embeddings are never silently re-extracted.
- Data paths can be set with environment variables (`PF_NIH_ROOT`, `PF_PADCHEST_ROOT`, `PF_PROJECT_DIR`).
- On the first run, the original-submission tables and figures are archived to `v1_original/`.

### Statistics
- All confidence intervals and paired NLP-versus-reference comparisons use a patient-level (cluster) bootstrap;
  image-level intervals are reported alongside for comparison.
- Instance dependence is tested with a patient-level bootstrap of the rank-biserial correlation,
  within-cluster resampling, and GEE logistic models (unadjusted and adjusted for sex, age band and
  projection), in addition to the original Mann-Whitney test.
- Subgroup AUROCs carry patient-level bootstrap intervals in addition to DeLong intervals.

### New analyses
- Fourth backbone: TorchXRayVision DenseNet-121 trained only on CheXpert (no documented training on NIH images or labels).
- Instance dependence with scores never trained on NLP labels: reference-standard heads cross-fitted by
  patient inside the evaluation set, and zero-shot outputs of the CheXpert-only model.
- Noise injection: calibrated arm matching the measured score dependence; patient-level bootstrap of the
  whole procedure; sample-split (cross-fitted) prediction; evidence summary table.
- Fracture with the CheXpert-only backbone and cross-fitted thresholds.
- Cohort flow, clustering within patients, representativeness (standardised mean differences) and an
  audit of the adjudicated label file (`Set Id`, raw values, metadata agreement).
- Ontology sensitivity (wider NLP mapping for airspace opacity, crosswalk) and the Google test split alone.
- Reference-set size planning (analytic and simulation).
- Hyperparameter grid search on the validation split and sensitivity of the conclusions to it.
- Patient-grouped five-fold resampling of the development data.
- Explainability: GEE model of which positives the NLP labels miss; exact class activation maps for the
  linear heads.
- Computational cost: extraction, training and inference times, analysis run times, hardware.

### Text
- Notebook narrative aligned with the manuscript: the regime-difference family is descriptive, the
  injection is a mechanistic illustration, and the analysis plan was not preregistered.
