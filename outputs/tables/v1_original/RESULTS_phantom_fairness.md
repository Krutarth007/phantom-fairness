## Results

Among adjudicated-positive images, the model scored NLP-missed positives below NLP-caught positives in every modelled finding (median rank-biserial correlation 0.20, maximum FDR-adjusted p 0.0141), establishing that the NIH NLP labelling noise is instance-dependent rather than random.

The same pattern held on the ResNet50 and CLIP backbones, which were not trained on the adjudicated labels (median rank-biserial 0.16), indicating the effect is not an artefact of backbone label exposure.

Across 24 finding-by-axis-by-metric comparisons on identical images, the gap measured on adjudicated labels exceeded the gap measured on NLP labels by a median of 0.021; 0 of 24 comparisons reached significance after Benjamini-Hochberg correction. The demographic concealment effect is directionally present but not individually significant at this adjudicated sample size.

In the 3 of 6 cells with observed concealment, injecting the measured marginal noise into the clean labels reproduced a median concealment of 0.081 when the noise targeted the model's lowest-scored positives, against 0.057 when the same marginal noise was applied at random, isolating the instance-dependent pathway.

In the 4 adequately powered cell(s), equal-opportunity thresholds calibrated on clean labels did not reduce the clean-test gap (median 0.046 to 0.082), while calibration on noisy labels did not reduce it (median 0.103).

PadChest subgroup gaps are reported as exploratory and read for direction only (n = 24, every subgroup underpowered for a 0.15 TPR gap).
