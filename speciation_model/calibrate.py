"""Fit the free parameters (model.FREE) to the training runs, using the twin.

The objective is the -log likelihood of the four collected curves (IV / OV, insoluble / soluble) of
the training runs (model.score); the source is fixed.  The search goes downhill from the start point
(Nelder-Mead), searches the whole parameter box (differential evolution) and refines the best point
(Nelder-Mead again).  It can be interrupted: running it again continues where it stopped.

    python calibrate.py --start params/start.txt --out OUT/calibrated.txt [--workers 12]
"""

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution, minimize  # noqa: E402

import model  # noqa: E402
import twin  # noqa: E402

SPACE = model.space()
BND = model.bounds(SPACE)


def objective(x, base):
    try:
        v = model.score(twin.campaign(model.from_x(x, base, SPACE), model.TRAIN), model.TRAIN)[0]
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return 1e12
    return v if np.isfinite(v) else 1e12


def simplex(x):
    """Initial Nelder-Mead simplex: log parameters moved by 0.1, linear ones by 5 % of their range."""
    pts = [np.array(x, float)]
    for i, ((_n, _lo, _hi, lg), (lo, hi)) in enumerate(zip(SPACE, BND)):
        step = 0.1 if lg else 0.05 * (hi - lo)
        p = np.array(x, float)
        p[i] = p[i] + step if p[i] + step <= hi else p[i] - step
        pts.append(p)
    return np.array(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=os.path.join(model.HERE, "params", "start.txt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--maxfev", type=int, default=3000, help="Nelder-Mead evaluations per round")
    a = ap.parse_args()
    base = model.read_params(a.start)
    ck_path = os.path.splitext(a.out)[0] + "_ckpt.json"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def f(x):
        return objective(x, base)

    def save(stage, x, **kw):
        with open(ck_path + ".tmp", "w") as fh:
            json.dump(dict(stage=stage, x=[float(v) for v in x], **kw), fh)
        os.replace(ck_path + ".tmp", ck_path)

    def nelder_mead(x):
        r = minimize(f, x, method="Nelder-Mead", bounds=BND, options=dict(
            maxiter=a.maxfev, maxfev=a.maxfev, xatol=1e-7, fatol=1e-7, adaptive=True, initial_simplex=simplex(x)))
        return r.x, float(r.fun)

    t0 = time.time()
    x0 = np.array([min(max(v, lo + 1e-9), hi - 1e-9) for v, (lo, hi) in zip(model.to_x(base, SPACE), BND)])
    ck = dict(stage="nm", x=list(x0))
    if os.path.exists(ck_path):
        with open(ck_path) as fh:
            ck = json.load(fh)
    stage, x = ck["stage"], np.array(ck["x"])
    print(f"start {model.rel(a.start)}: -logL {f(x0):.3f}; stage {stage}", flush=True)
    if stage == "nm":
        x, v = nelder_mead(x)
        save("de", x, gen=0)
        stage = "de"
        print(f"  Nelder-Mead from the start: {v:.3f} ({time.time() - t0:.0f} s)", flush=True)
    if stage == "de":
        g0 = int(ck.get("gen", 0)) if ck["stage"] == "de" else 0

        def progress(intermediate_result):
            save("de", intermediate_result.x, gen=g0 + int(intermediate_result.nit))
            print(f"  DE generation {g0 + int(intermediate_result.nit)}: {float(intermediate_result.fun):.3f}", flush=True)

        if a.generations > g0:
            x = differential_evolution(
                objective, BND, args=(base,), seed=1 + g0, maxiter=a.generations - g0, popsize=12, tol=1e-8,
                mutation=(0.4, 1.0), recombination=0.8, polish=False, workers=a.workers, updating="deferred",
                init="sobol", x0=x, callback=progress).x
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
    th = model.from_x(x, base, SPACE)
    sim = twin.campaign(th)
    tr, he = model.score(sim, model.TRAIN)[0], model.score(sim, model.HELD)[0]
    edge = [n for (n, *_), v, (lo, hi) in zip(SPACE, x, BND) if min(v - lo, hi - v) < 1e-3]
    model.write_params(a.out, th, f"Twin calibration on runs {', '.join(map(str, model.TRAIN))} (calibrate.py, start "
                       f"{model.rel(a.start)}).\nTraining -logL {tr:.3f}, held-out (runs {', '.join(map(str, model.HELD))}) "
                       f"{he:.3f}.  At a bound: {edge or 'none'}.")
    print(f"wrote {model.rel(a.out)} (training -logL {tr:.3f}, held-out {he:.3f}; at a bound {edge or 'none'}; "
          f"{time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
