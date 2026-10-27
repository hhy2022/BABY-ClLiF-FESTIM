"""How much worse than the best fit a parameter set may be and still count as fitting the data.

The threshold is 1.92 * chi2_red, where 1.92 is the usual one-parameter 95 % limit (half of 3.84)
and chi2_red scales it to the actual scatter of the data around the FESTIM fit.  chi2_red is the sum
of the squared normalised residuals of the training runs, divided by the number of data points minus
the number of free parameters.  The OV end values of runs 1 and 2 are left out, because the model
gives almost no OV release under pure He.

The threshold is printed and saved next to the FESTIM chain.

    python admissibility_threshold.py params_refit.txt festim_chain
"""

import argparse
import os

import numpy as np

import chain
import experimental_data
import likelihood
import parameters

# Pure-He runs: the model releases almost nothing to the OV, so their OV end values are not used
RUNS_WITHOUT_OV_END = (1, 2)

ap = argparse.ArgumentParser()
ap.add_argument("params", help="parameter file of the final point")
ap.add_argument("chain", help="FESTIM chain folder at that point")
a = ap.parse_args()

th = parameters.read_params(a.params)
P = chain.forward_festim(th, a.chain, experimental_data.TRAIN)
chi2, n = 0.0, 0
for r in experimental_data.TRAIN:
    c = experimental_data.curves(r)
    for s, i in (("iv", 1), ("ov", 2)):
        ym = np.interp(c[f"{s}_t"], P[r][0], P[r][i])
        inc_m, inc_e = np.diff(np.r_[0.0, ym]), np.diff(np.r_[0.0, c[s]])
        sig = np.sqrt(likelihood.NOISE_FLOOR**2 + (likelihood.NOISE_REL * inc_m) ** 2)
        chi2 += float(np.sum(((inc_e - inc_m) / sig) ** 2))
        n += len(inc_e)
        if s == "ov" and r in RUNS_WITHOUT_OV_END:
            continue
        se = np.sqrt(likelihood.NOISE_FLOOR**2 + (likelihood.END_REL * c[s][-1]) ** 2)
        chi2 += float(((ym[-1] - c[s][-1]) / se) ** 2)
        n += 1

k = len(parameters.FREE)
red = chi2 / (n - k)
print(
    f"chi2 {chi2:.1f}, N {n}, k {k}, chi2_red {red:.3f}, "
    f"threshold 1.92 * chi2_red = {1.92 * red:.3f}"
)
with open(os.path.join(a.chain, "admissibility_threshold.txt"), "w") as f:
    f.write(f"{1.92 * red:.4f}\n")
