"""How far each parameter can move while the model still fits the training runs (profile likelihood).

One parameter is moved step by step away from its calibrated value, and at every step all the other
parameters are fitted again.  When the fit has become worse than the calibrated one by more than the
threshold (admissibility_threshold.py), that side of the range ends.  This is done in both directions
for every free parameter, using the surrogate; it takes several hours on 12 cores.

The ranges are printed and saved, together with every step (so an interrupted run continues) and the
parameter set at the outer end of each range.

    python parameter_ranges.py --params calibrated.txt --threshold 5.3 --outdir parameter_ranges
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
ap.add_argument("--threshold", type=float, required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument(
    "--step",
    type=float,
    default=0.12,
    help="first step in ln(value / calibrated value)",
)
ap.add_argument("--maxsteps", type=int, default=10)
ap.add_argument(
    "--maxfev", type=int, default=700, help="Nelder-Mead evaluations per point"
)
ap.add_argument(
    "--only", default="", help="comma-separated parameters to profile (default: all)"
)
a = ap.parse_args()
BASE = parameters.read_params(a.params)
os.makedirs(os.path.join(a.outdir, "params"), exist_ok=True)
PTS = os.path.join(a.outdir, "profile_points.jsonl")
BOX = {n: (lo, hi, lg) for n, lo, hi, lg in parameters.SPACE}
ONLY = [x for x in a.only.split(",") if x]


def point(name, value, start):
    """Minimise the -log posterior with `name` fixed at `value`, starting the others from `start`."""
    sp = [s for s in parameters.SPACE if s[0] != name]
    bnd = parameters.bounds(sp)
    held = dict(start, **{name: float(value)})
    x = np.array(
        [
            min(max(v, lo + 1e-9), hi - 1e-9)
            for v, (lo, hi) in zip(parameters.to_x(start, sp), bnd)
        ]
    )
    f = None
    # Nelder-Mead restarted twice from the best point, with a fresh simplex, unless a restart gains < 0.02
    for n in (a.maxfev, (5 * a.maxfev) // 7, (3 * a.maxfev) // 7):
        r = minimize(
            likelihood.objective_x,
            x,
            args=(held, sp),
            method="Nelder-Mead",
            bounds=bnd,
            options=dict(
                maxiter=n,
                maxfev=n,
                xatol=1e-5,
                fatol=1e-5,
                adaptive=True,
                initial_simplex=likelihood.simplex(x, sp),
            ),
        )
        done = f is not None and f - r.fun < 0.02
        if f is None or r.fun < f:
            x, f = r.x, float(r.fun)
        if done:
            break
    th = parameters.from_x(x, held, sp)
    P = chain.forward(th, cache=likelihood.CACHE)
    return dict(
        param=name,
        value=float(value),
        factor=float(value / BASE[name]),
        post=f,
        ends=[float(P[r][i][-1]) for r in experimental_data.HELD for i in (1, 2)],
        theta={n: th[n] for n in parameters.SPEC},
    )


def direction(job):
    """Step one parameter away from its calibrated value in one direction (sign = -1 or +1)."""
    name, sign, done, post0 = job
    lo, hi, lg = BOX[name]
    rows, start, lnf, step = [], dict(BASE), 0.0, a.step
    for _ in range(a.maxsteps):
        lnf += sign * step
        if lg:
            val = BASE[name] * np.exp(lnf)
        else:
            val = BASE[name] + sign * abs(BASE[name] or 1.0) * (np.exp(abs(lnf)) - 1.0)
        val = float(np.clip(val, lo, hi))
        key = f"{name}|{val:.10e}"
        if key in done:
            m = done[key]
        else:
            t0 = time.time()
            m = point(name, val, start)
            m["key"], m["seconds"] = key, time.time() - t0
            with open(PTS, "a") as fh:
                fh.write(json.dumps(m) + "\n")
        rows.append(m)
        start = m["theta"]
        cost = m["post"] - post0
        if cost > a.threshold or val in (lo, hi):
            break
        if cost < 0.25 * a.threshold:
            step *= 2.0  # flat direction: take larger steps
    return name, sign, rows


if __name__ == "__main__":
    from multiprocessing import Pool

    post0 = float(likelihood.objective_x(parameters.to_x(BASE), BASE))
    done = {}
    if os.path.exists(PTS):
        with open(PTS) as fh:
            for line in fh:
                # an interrupted run can leave a half-written last line
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[d["key"]] = d
    print(
        f"profile around {parameters.rel(a.params)}: -logpost {post0:.3f}, threshold {a.threshold}; {len(done)} points on disk",
        flush=True,
    )
    jobs = [
        (n, s, done, post0)
        for n, *_ in parameters.SPACE
        if not ONLY or n in ONLY
        for s in (-1, 1)
    ]
    with Pool(a.workers) as pool:
        res = pool.map(direction, jobs, chunksize=1)
    best = min([post0] + [m["post"] for _, _, rows in res for m in rows])
    lines = [
        f"profile likelihood around {parameters.rel(a.params)}",
        f"base -logpost {post0:.3f}; lowest found {best:.3f}"
        + (
            "  <- a profile point beats the base: recalibrate from it"
            if best < post0 - 0.05
            else ""
        ),
        f"admissible: -logpost within {a.threshold} of {best:.3f}",
        "",
        f"{'parameter':<9} {'estimate':>11} | {'lower edge':>22} | {'upper edge':>22}",
    ]
    for n, *_ in parameters.SPACE:
        if ONLY and n not in ONLY:
            continue
        edges = {}
        for name, sign, rows in res:
            if name != n:
                continue
            ok = [m for m in rows if m["post"] - best <= a.threshold]
            last = rows[-1]
            prev = ok[-1] if ok else dict(factor=1.0, post=post0)
            if last["post"] - best > a.threshold:
                # the edge: where -logpost reaches best + threshold, linear in ln(factor)
                w = (best + a.threshold - prev["post"]) / (last["post"] - prev["post"])
                if prev["factor"] > 0 and last["factor"] > 0:
                    lp, ll = np.log(prev["factor"]), np.log(last["factor"])
                    fac = float(np.exp(lp + w * (ll - lp)))
                else:
                    fac = last["factor"]
                edges[sign] = f"{BASE[n] * fac:10.4g} (x{fac:.3g})"
            else:
                # the threshold was not reached before the bound or the last step
                edges[sign] = f"{'>=' if sign > 0 else '<='}{last['value']:9.4g} (open)"
            if ok:
                side = "hi" if sign > 0 else "lo"
                parameters.write_params(
                    os.path.join(a.outdir, "params", f"{n}_{side}.txt"),
                    ok[-1]["theta"],
                    f"Outermost admissible profile point, {n} {'up' if sign > 0 else 'down'} "
                    f"(x{ok[-1]['factor']:.3g}); -logpost {ok[-1]['post']:.3f}, best {best:.3f}, "
                    f"threshold {a.threshold} (parameter_ranges.py).",
                )
        lines.append(
            f"{n:<9} {BASE[n]:11.4g} | {edges.get(-1, '-'):>22} | {edges.get(1, '-'):>22}"
        )
    with open(os.path.join(a.outdir, "parameter_ranges.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
