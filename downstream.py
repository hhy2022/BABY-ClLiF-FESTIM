"""Turn the FESTIM release from the vessel into the activity collected in the IV and OV bubblers.

FESTIM gives the tritium leaving the vessel; part of it is held up in the lines on its way to the
bubblers (chain.py).  This script adds that holdup to a finished FESTIM chain, saves the collected
IV and OV activity of every run, and prints how close each run comes to the measurements.

The six holdup parameters are calibrated together with the rest using the surrogate.  With --refit
they are first fitted again to the training runs, this time on the FESTIM release, and the updated
parameter set is saved to a new file.

    python downstream.py params.txt chain_folder
    python downstream.py params.txt chain_folder --refit params_refit.txt
"""

import argparse
import os

import numpy as np
from scipy.optimize import differential_evolution

import chain
import experimental_data
import likelihood
import parameters

ap = argparse.ArgumentParser()
ap.add_argument("params", help="parameter file")
ap.add_argument("chain", help="FESTIM chain folder (baby_festim.py output)")
ap.add_argument(
    "--refit", metavar="OUT_PARAMS", help="re-fit the downstream parameters first"
)
a = ap.parse_args()
th = parameters.read_params(a.params)

if a.refit:
    # Search bounds of the calibration.  k_iv_he is not calibrated with the surrogate, so it is given
    # the bounds of k_ov_he.
    names = ("alpha_iv", "alpha_ov", "k_iv", "k_ov", "k_ov_he")
    BOX = [(n, *parameters.SPEC[n][3:6]) for n in names]
    BOX.append(("k_iv_he", *parameters.SPEC["k_ov_he"][3:6]))
    bnd = parameters.bounds(BOX)

    def cost(x):
        t = parameters.from_x(x, th, BOX)
        P = chain.forward_festim(t, a.chain, experimental_data.TRAIN)
        nll = sum(-likelihood.loglike_run(P[r], r) for r in experimental_data.TRAIN)
        return nll + likelihood.he_penalty(t)

    res = differential_evolution(
        cost,
        bnd,
        seed=1,
        maxiter=300,
        popsize=24,
        tol=1e-9,
        mutation=(0.4, 1.0),
        recombination=0.8,
        polish=True,
        init="sobol",
    )
    th = parameters.from_x(res.x, th, BOX)
    # a rate at its lower bound means the data only give an upper limit
    at_floor = [
        n for v, (n, *_), (lo, _hi) in zip(res.x, BOX, bnd) if abs(v - lo) < 1e-6
    ]
    train = ", ".join(map(str, experimental_data.TRAIN))
    parameters.write_params(
        a.refit,
        th,
        f"{parameters.rel(a.params)} with the downstream parameters re-fitted on the FESTIM chain\n"
        f"{parameters.rel(a.chain)} (training runs {train}; downstream.py --refit).\n"
        f"Training -logL {res.fun:.2f}.  At the lower bound (an upper limit): {at_floor or 'none'}.",
    )
    print(
        f"downstream refitted (training -logL {res.fun:.2f}) -> {parameters.rel(a.refit)}"
    )

P = chain.forward_festim(th, a.chain)
for r in experimental_data.RUNS:
    np.savetxt(
        os.path.join(a.chain, f"recovery_run{r}.csv"),
        np.column_stack(P[r]),
        delimiter=",",
        header="t_day,iv_Bq,ov_Bq",
        comments="",
    )
nll, tr, hd = likelihood.scores(P)
print(f"chain {parameters.rel(a.chain)}")
print(
    f"{'run':>4} {'IV model':>9} {'IV meas':>8} {'OV model':>9} {'OV meas':>8} {'-logL':>7}"
)
for r in experimental_data.RUNS:
    c = experimental_data.curves(r)
    held = "   (held out)" if r in experimental_data.HELD else ""
    print(
        f"{r:>4} {P[r][1][-1]:9.2f} {c['iv'][-1]:8.2f} {P[r][2][-1]:9.2f} {c['ov'][-1]:8.2f} "
        f"{nll[r]:7.2f}{held}"
    )
print(f"training -logL {tr:.2f}   held-out -logL {hd:.2f}")
