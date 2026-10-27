"""Range of each held-out prediction by prediction profile likelihood (Kreutz et al. 2012).

The predictions are the final cumulative IV and OV releases of the held-out runs.  For each one, the
largest and the smallest value are searched for among parameter sets that still fit the training
runs, i.e. whose -log posterior C stays within the threshold of its value C* at the calibrated point:

    max / min q(theta)   subject to   C(theta) <= C* + threshold

The constraint is added as a penalty, w * max(0, C - C* - threshold), and the problem is solved with
Nelder-Mead.  The search starts from the profile point of parameter_ranges.py that already gives the
most extreme value; if the result ends up slightly outside the constraint, it is moved back onto the
boundary along the line from the start point.

    python prediction_ranges.py --params P --parameter-ranges D --threshold THR --out OUT

Writes OUT/<case>.json, OUT/params/<case>.txt (the parameter set of each extreme, later solved with
FESTIM) and OUT/prediction_ranges.txt.  Finished cases are kept, so an interrupted run continues.
"""

import argparse
import json
import os
import time

# The optimizer runs one process per core, so each process should use a single thread.  NumPy reads
# this setting when it is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize

import chain
import experimental_data
import likelihood
import parameters


ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True)
ap.add_argument(
    "--parameter-ranges", required=True, help="output folder of parameter_ranges.py"
)
ap.add_argument("--threshold", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument(
    "--w",
    type=float,
    default=30.0,
    help="penalty weight [Bq per unit of -log posterior]",
)
ap.add_argument("--maxfev", type=int, default=3000)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "params"), exist_ok=True)
BASE = parameters.read_params(a.params)
BND = np.array(parameters.bounds())
# (name, run, index in chain.forward's (t, IV, OV)), in the order of the profile points' "ends"
QUANT = [
    (f"run{r}_{name}", r, i)
    for r in experimental_data.HELD
    for name, i in (("IV", 1), ("OV", 2))
]
CASES = [(q, r, i, s) for q, r, i in QUANT for s in ("max", "min")]


def clip(x):
    return np.clip(x, BND[:, 0], BND[:, 1])


def cost(x):
    return float(likelihood.objective_x(clip(x), BASE))


def predict(x):
    return chain.forward(parameters.from_x(clip(x), BASE), cache=likelihood.CACHE)


C0 = cost(parameters.to_x(BASE))
LIMIT = C0 + a.threshold


def solve(case):
    q, r, i, s = case
    f_out = os.path.join(a.out, f"{q}_{s}.json")
    if os.path.exists(f_out):
        with open(f_out) as fh:
            return json.load(fh)
    t0 = time.time()
    # start from the admissible profile point with the most extreme value of q
    pts = []
    with open(os.path.join(a.parameter_ranges, "profile_points.jsonl")) as fh:
        for line in fh:
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            if p["post"] <= LIMIT:
                pts.append(p)
    k = [n for n, *_ in QUANT].index(q)
    if pts:
        start = (max if s == "max" else min)(pts, key=lambda p: p["ends"][k])
    else:
        start = {"theta": BASE}
    x0 = np.clip(
        parameters.to_x({**BASE, **start["theta"]}), BND[:, 0] + 1e-9, BND[:, 1] - 1e-9
    )
    sign = 1.0 if s == "max" else -1.0

    def f(x):
        return -sign * float(predict(x)[r][i][-1]) + a.w * max(0.0, cost(x) - LIMIT)

    x, fbest = x0, f(x0)
    # Nelder-Mead restarted twice from the best point, with a fresh simplex, unless a restart gains < 0.02
    for n in (a.maxfev, a.maxfev // 2, a.maxfev // 2):
        res = minimize(
            f,
            x,
            method="Nelder-Mead",
            options=dict(
                maxfev=n,
                maxiter=n,
                xatol=1e-6,
                fatol=1e-4,
                adaptive=True,
                initial_simplex=likelihood.simplex(x),
            ),
        )
        gain = fbest - res.fun
        if res.fun < fbest:
            x, fbest = res.x, float(res.fun)
        if gain < 0.02:
            break
    x = clip(x)
    if cost(x) > LIMIT:
        # bisect along the line from the start back onto C = C* + threshold
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if cost(x0 + mid * (x - x0)) <= LIMIT else (lo, mid)
        x = x0 + lo * (x - x0)
    th = parameters.from_x(x, BASE)
    P = predict(x)
    out = dict(
        case=f"{q}_{s}",
        value=float(P[r][i][-1]),
        cost=cost(x) - C0,
        start=start.get("key", "base"),
        ends={name: float(P[rr][ii][-1]) for name, rr, ii in QUANT},
        theta={n: float(th[n]) for n in parameters.SPEC},
        seconds=time.time() - t0,
    )
    parameters.write_params(
        os.path.join(a.out, "params", f"{q}_{s}.txt"),
        th,
        f"Prediction-profile extreme: {s} of {q} with the training cost <= C* + {a.threshold}\n"
        f"(prediction_ranges.py): {q} = {out['value']:.3f} Bq at cost {out['cost']:.3f}.",
    )
    with open(f_out + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(f_out + ".tmp", f_out)
    print(
        f"  {q} {s}: {out['value']:.2f} Bq, cost {out['cost']:.2f}, {out['seconds']:.0f} s",
        flush=True,
    )
    return out


if __name__ == "__main__":
    from multiprocessing import Pool

    print(
        f"prediction profile around {parameters.rel(a.params)}: C* = {C0:.3f}, threshold {a.threshold}",
        flush=True,
    )
    with Pool(min(a.workers, len(CASES))) as pool:
        R = {o["case"]: o for o in pool.map(solve, CASES, chunksize=1)}
    lines = [
        f"prediction profile likelihood (surrogate), threshold {a.threshold} above C* = {C0:.3f}",
        "",
        f"{'prediction':<10} {'min':>8} {'max':>8}   {'measured':>8}",
    ]
    for q, r, i in QUANT:
        lo, hi = R[f"{q}_min"], R[f"{q}_max"]
        meas = float(experimental_data.curves(r)["iv" if i == 1 else "ov"][-1])
        lines.append(f"{q:<10} {lo['value']:8.2f} {hi['value']:8.2f}   {meas:8.2f}")
    with open(os.path.join(a.out, "prediction_ranges.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
