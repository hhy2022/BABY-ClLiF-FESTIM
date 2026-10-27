"""Find the parameter set that best fits the training runs, using the surrogate.

The free parameters (parameters.SPEC) are adjusted to minimise the -log posterior of the training
runs; the fixed ones are taken from the start file.  The search first goes downhill from the start
point (Nelder-Mead), then searches the whole parameter range (differential evolution), and finally
refines the best point found (Nelder-Mead again).  It takes about an hour on 12 cores and can be
interrupted: running it again continues where it stopped.

    python calibrate.py --start params/paper_point.txt --out params/calibrated.txt
"""

import argparse
import json
import os
import time

# The optimizer runs one process per core, so each process should use a single thread.  NumPy reads
# this setting when it is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import differential_evolution, minimize

import chain
import experimental_data
import likelihood
import parameters


ap = argparse.ArgumentParser()
ap.add_argument(
    "--start", default=os.path.join(parameters.ROOT, "params", "paper_point.txt")
)
ap.add_argument(
    "--out", default=os.path.join(parameters.ROOT, "params", "calibrated.txt")
)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--generations", type=int, default=40)
ap.add_argument(
    "--maxfev", type=int, default=1500, help="Nelder-Mead evaluations per round"
)
a = ap.parse_args()
base = parameters.read_params(a.start)
CK = os.path.splitext(a.out)[0] + "_ckpt.json"
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
BND = parameters.bounds()


def f(x):
    return likelihood.objective_x(x, base)


def save(stage, x, **kw):
    with open(CK + ".tmp", "w") as fh:
        json.dump(dict(stage=stage, x=[float(v) for v in x], **kw), fh)
    os.replace(CK + ".tmp", CK)


def nelder_mead(x):
    r = minimize(
        f,
        x,
        method="Nelder-Mead",
        bounds=BND,
        options=dict(
            maxiter=a.maxfev,
            maxfev=a.maxfev,
            xatol=1e-7,
            fatol=1e-7,
            adaptive=True,
            initial_simplex=likelihood.simplex(x),
        ),
    )
    return r.x, float(r.fun)


if __name__ == "__main__":
    t0 = time.time()
    x0 = np.array(
        [
            min(max(v, lo + 1e-9), hi - 1e-9)
            for v, (lo, hi) in zip(parameters.to_x(base), BND)
        ]
    )
    ck = dict(stage="nm", x=list(x0))
    if os.path.exists(CK):
        with open(CK) as fh:
            ck = json.load(fh)
    stage, x = ck["stage"], np.array(ck["x"])
    print(
        f"start {parameters.rel(a.start)}: -logpost {f(x0):.3f}; stage {stage}",
        flush=True,
    )
    if stage == "nm":
        x, v = nelder_mead(x)
        save("de", x, gen=0)
        stage = "de"
        print(
            f"  Nelder-Mead from the start: {v:.3f} ({time.time() - t0:.0f} s)",
            flush=True,
        )
    if stage == "de":
        g0 = int(ck.get("gen", 0)) if ck["stage"] == "de" else 0

        # scipy recognises this form of callback by the argument name intermediate_result
        def progress(intermediate_result):
            save("de", intermediate_result.x, gen=g0 + int(intermediate_result.nit))
            print(
                f"  DE generation {g0 + int(intermediate_result.nit)}: {float(intermediate_result.fun):.3f}",
                flush=True,
            )

        if a.generations > g0:
            x = differential_evolution(
                likelihood.objective_x,
                BND,
                args=(base,),
                seed=1 + g0,
                maxiter=a.generations - g0,
                popsize=12,
                tol=1e-8,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,
                workers=a.workers,
                updating="deferred",
                init="sobol",
                x0=x,
                callback=progress,
            ).x
        save("polish", x)
        stage = "polish"
    if stage == "polish":
        v = f(x)
        for k in range(12):
            x1, v1 = nelder_mead(x)
            gain = v - v1
            if v1 < v:
                x, v = x1, v1
            print(f"  polish round {k + 1}: {v:.4f} (gain {gain:.4f})", flush=True)
            if gain < 0.01:
                break
        save("done", x)
    th = parameters.from_x(x, base)
    nll, tr, hd = likelihood.scores(chain.forward(th))
    edge = [
        n
        for (n, *_), v, (lo, hi) in zip(parameters.SPACE, x, BND)
        if min(v - lo, hi - v) < 1e-3
    ]
    train, held = (
        ", ".join(map(str, rs))
        for rs in (experimental_data.TRAIN, experimental_data.HELD)
    )
    parameters.write_params(
        a.out,
        th,
        f"Surrogate calibration on runs {train} (calibrate.py, start "
        f"{parameters.rel(a.start)}).\n-logpost {f(x):.3f}; surrogate training -logL {tr:.2f}, held-out "
        f"(runs {held}) {hd:.2f}.  At a bound: {edge or 'none'}.",
    )
    print(
        f"wrote {parameters.rel(a.out)} (-logpost {f(x):.3f}, training -logL {tr:.2f}, held-out {hd:.2f}; {time.time() - t0:.0f} s)"
    )
