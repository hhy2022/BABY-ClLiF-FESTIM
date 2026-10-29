"""Compare the surrogate with FESTIM at the same parameter set, run by run.

Calibration and the parameter and prediction ranges need thousands of model runs, so they use the
surrogate; this script checks that it gives the same IV and OV curves as FESTIM.  Both go through
the same downstream holdup layer.  For each run it prints the final values of both models, the
largest difference between them at the sampling times, and the -logL of each.

    python compare_surrogate_festim.py --params params_refit.txt --chain festim_chain
"""

import argparse

import numpy as np

import chain
import experimental_data
import likelihood
import parameters

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True, help="parameter file")
ap.add_argument(
    "--chain", required=True, help="FESTIM chain folder at the same parameters"
)
a = ap.parse_args()

th = parameters.read_params(a.params)
S, F = chain.forward(th), chain.forward_festim(th, a.chain)
(nll_s, tr_s, hd_s), (nll_f, tr_f, hd_f) = likelihood.scores(S), likelihood.scores(F)

print(
    f"parameters {parameters.rel(a.params)}, FESTIM chain {parameters.rel(a.chain)}\n"
)
print(
    f"{'run':>4} | {'IV surr':>8} {'FESTIM':>8} {'max|d|':>7} | "
    f"{'OV surr':>8} {'FESTIM':>8} {'max|d|':>7} | {'-logL surr':>10} {'FESTIM':>8}"
)
for r in experimental_data.RUNS:
    c = experimental_data.curves(r)
    diff = []
    for s, i in (("iv", 1), ("ov", 2)):
        t = c[f"{s}_t"]
        d = np.interp(t, S[r][0], S[r][i]) - np.interp(t, F[r][0], F[r][i])
        diff.append(float(np.max(np.abs(d))))
    held = "  (held out)" if r in experimental_data.HELD else ""
    print(
        f"{r:>4} | {S[r][1][-1]:8.2f} {F[r][1][-1]:8.2f} {diff[0]:7.3f} | "
        f"{S[r][2][-1]:8.2f} {F[r][2][-1]:8.2f} {diff[1]:7.3f} | "
        f"{nll_s[r]:10.2f} {nll_f[r]:8.2f}{held}"
    )
print("\n(max|d|: largest difference between the two models at the sampling times, Bq)")
print(
    f"training -logL: surrogate {tr_s:.2f}, FESTIM {tr_f:.2f}; "
    f"held-out: surrogate {hd_s:.2f}, FESTIM {hd_f:.2f}"
)
