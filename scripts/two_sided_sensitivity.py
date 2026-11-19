"""Two-sided sensitivity analysis for the instance-dependence tests (Table S13c of the manuscript).

The notebooks test the directional hypothesis that NLP-missed positives score lower than NLP-caught
positives, so their p values are one-sided. This script derives two-sided p values from the saved
one-sided results and re-applies the Benjamini-Hochberg adjustment within each test and family
(12 tests with classifiers trained on NLP labels; 15 tests with NLP-independent scores).

Two-sided p = twice the smaller tail. For the patient-level bootstrap (B = 5,000 resamples, one-sided
p = (k + 1)/(B + 1) with k resamples at or below zero) the two-sided value is
2 * min(k + 1, B - k + 1)/(B + 1), capped at 1. For the other tests it is 2 * min(p, 1 - p).

Usage:  python scripts/two_sided_sensitivity.py [path/to/outputs/tables]
Writes: table13c_two_sided_sensitivity.csv in the same folder.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 5000
TESTS = {"p_patient_boot": "bootstrap", "p_wcr": "wcr", "p_gee": "gee", "p_gee_adj": "gee_adj"}
FAMILIES = [("table10b_instance_dependence_backbones.csv", "trained on NLP labels"),
            ("table13_instance_dependence_nlp_independent.csv", "never trained on NLP labels")]


def bh(p):
    """Benjamini-Hochberg adjusted p values."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1)
    return out


def two_sided(p, bootstrap=False, n_boot=N_BOOT):
    p = np.asarray(p, float)
    if bootstrap:
        k = np.rint(p * (n_boot + 1) - 1)
        return np.minimum(1, 2 * np.minimum(k + 1, n_boot - k + 1) / (n_boot + 1))
    return np.minimum(1, 2 * np.minimum(p, 1 - p))


def main(tab_dir):
    tab_dir = Path(tab_dir)
    parts = []
    for fname, family in FAMILIES:
        t = pd.read_csv(tab_dir / fname)
        out = pd.DataFrame({"family": family, "score_source": t["score_source"], "finding": t["finding"],
                            "rank_biserial": t["rank_biserial"]})
        for col, name in TESTS.items():
            ok = t[col].notna()
            out[f"p_{name}_one_sided_bh"] = t[col + "_bh"]
            adj = pd.Series(np.nan, index=t.index)
            adj[ok] = bh(two_sided(t.loc[ok, col], bootstrap=(col == "p_patient_boot")))
            out[f"p_{name}_two_sided_bh"] = adj
        parts.append(out)
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(tab_dir / "table13c_two_sided_sensitivity.csv", index=False)
    for family, g in res.groupby("family", sort=False):
        same = (g["p_bootstrap_two_sided_bh"] < 0.05) & (g["rank_biserial"] > 0)
        rev = (g["p_bootstrap_two_sided_bh"] < 0.05) & (g["rank_biserial"] < 0)
        print(f"{family}: {int(same.sum())} of {len(g)} significant in the expected direction (two-sided, BH), "
              f"{int(rev.sum())} significant in the reversed direction")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/tables")
