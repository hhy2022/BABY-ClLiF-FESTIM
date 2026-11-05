"""Score a finished FESTIM chain (festim_model.py) against the measurements and against the twin.

The collection-line parameters are calibrated with the twin; with --refit they are first fitted again
to the training runs on the FESTIM release, and the updated parameter set is saved.  The collected
curves of every run are written to CHAIN/collected_run<r>.csv.

    python evaluate.py PARAMS CHAIN [--refit OUT_PARAMS]
"""

import argparse
import os

import numpy as np
from scipy.optimize import differential_evolution

import festim_model
import model
import twin

KEYS = (("iv", "ins"), ("iv", "sol"), ("ov", "ins"), ("ov", "sol"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("chain")
    ap.add_argument("--refit", metavar="OUT_PARAMS")
    a = ap.parse_args()
    th = model.read_params(a.params)
    hz = {r: festim_model.read_run(os.path.join(a.chain, f"run_{r}")) for r in model.RUNS}

    def festim(t, runs=model.RUNS):
        return model.campaign(t, lambda r, _init: hz[r], runs)

    if a.refit:
        sp = model.space(model.LINE_KEYS)
        res = differential_evolution(
            lambda x: model.score(festim(model.from_x(x, th, sp), model.TRAIN), model.TRAIN)[0], model.bounds(sp),
            seed=1, maxiter=300, popsize=24, tol=1e-9, mutation=(0.4, 1.0), recombination=0.8, polish=True,
            init="sobol", x0=np.clip(model.to_x(th, sp), *np.array(model.bounds(sp)).T))
        th = model.from_x(res.x, th, sp)
        edge = [n for (n, *_), v, (lo, hi) in zip(sp, res.x, model.bounds(sp)) if min(v - lo, hi - v) < 1e-3]
        model.write_params(a.refit, th, f"{model.rel(a.params)} with the collection-line parameters fitted again on the\n"
                           f"FESTIM chain {model.rel(a.chain)} (training runs; evaluate.py --refit).\n"
                           f"Training -logL {res.fun:.3f}.  At a bound: {edge or 'none'}.")
        print(f"collection lines refitted on FESTIM (training -logL {res.fun:.3f}) -> {model.rel(a.refit)}")

    F, T = festim(th), twin.campaign(th)
    for r in model.RUNS:
        cols = [F[r]["t"]] + [F[r]["curves"][k] * model.BQ for k in KEYS]
        np.savetxt(os.path.join(a.chain, f"collected_run{r}.csv"), np.column_stack(cols), delimiter=",",
                   header="t_day,iv_insoluble_Bq,iv_soluble_Bq,ov_insoluble_Bq,ov_soluble_Bq", comments="")

    print(f"\nchain {model.rel(a.chain)}: end values [Bq], model / measured  (* = held out)")
    print(f"{'run':>5} " + " ".join(f"{l.upper() + ' ' + ('insoluble' if f == 'ins' else 'soluble'):>17}" for l, f in KEYS)
          + f" {'-logL':>8} {'balance':>9}")
    for r in model.RUNS:
        meas = model.measured(r)
        ends = " ".join(f"{F[r]['curves'][k][-1] * model.BQ:8.2f}/{meas[k][1][-1]:<8.2f}" for k in KEYS)
        with open(os.path.join(a.chain, f"run_{r}", "run_complete")) as f:
            bal = f.read().split("=")[-1].strip()
        print(f"{r:>4}{' ' if r in model.TRAIN else '*'} {ends} {model.score(F, [r])[0]:8.2f} {bal:>9}")
    for name, runs in (("training", model.TRAIN), ("held-out", model.HELD)):
        tot, by = model.score(F, runs)
        print(f"{name:9s} -logL {tot:8.3f}   " + "  ".join(
            f"{l.upper()} {f}: chi2 {by[(l, f)][1]:.0f}/{by[(l, f)][2]}" for l, f in KEYS))

    print("\ntwin vs FESTIM, collected curves: largest difference along the run [Bq]")
    worst = 0.0
    for r in model.RUNS:
        d = [np.max(np.abs(np.interp(T[r]["t"], F[r]["t"], F[r]["curves"][k]) - T[r]["curves"][k])) * model.BQ for k in KEYS]
        worst = max(worst, *d)
        print(f"{r:>4}  " + "  ".join(f"{l.upper()} {f} {v:6.3f}" for (l, f), v in zip(KEYS, d)))
    print(f"training -logL: twin {model.score(T, model.TRAIN)[0]:.3f}, FESTIM {model.score(F, model.TRAIN)[0]:.3f}; "
          f"held-out: twin {model.score(T, model.HELD)[0]:.3f}, FESTIM {model.score(F, model.HELD)[0]:.3f}; "
          f"largest curve difference {worst:.3f} Bq")


if __name__ == "__main__":
    main()
