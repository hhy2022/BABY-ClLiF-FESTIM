"""Summary table of the final model, written to figures/summary.txt by run_all.sh.

It lists the fit of the final FESTIM model to every run, the range of each calibrated parameter
(parameter_ranges.py) and the range of each held-out prediction (prediction_ranges.py), with both
the surrogate's values and the FESTIM runs at the extremes.

    python summary.py figures
"""

import argparse
import glob
import json
import os

import chain
import experimental_data
import likelihood
import parameters

ap = argparse.ArgumentParser()
ap.add_argument("figures", help="output folder of run_all.sh")
a = ap.parse_args()
W = os.path.join(a.figures, "model_output")


def read(path):
    with open(path) as fh:
        return fh.read()


def split_case(q):
    """'run4_IV' -> (4, index of IV in chain.forward's (t, IV, OV), 'iv')."""
    run, stream = q[len("run") :].split("_")
    return int(run), (1 if stream == "IV" else 2), stream.lower()


final = os.path.join(W, "festim", "params_refit.txt")
th = parameters.read_params(final)
P = chain.forward_festim(th, os.path.join(W, "festim"))
nll, tr, hd = likelihood.scores(P)
thr = float(read(os.path.join(W, "festim", "admissibility_threshold.txt")))

print(f"final parameter set: {parameters.rel(final)}")
print(
    f"FESTIM training -logL {tr:.2f}, held-out {hd:.2f}; admissibility threshold {thr:.3f}\n"
)
print(
    f"{'run':>4} {'IV model':>9} {'IV meas':>8} {'OV model':>9} {'OV meas':>8} {'-logL':>7}"
)
for r in experimental_data.RUNS:
    c = experimental_data.curves(r)
    held = "  (held out)" if r in experimental_data.HELD else ""
    print(
        f"{r:>4} {P[r][1][-1]:9.2f} {c['iv'][-1]:8.2f} {P[r][2][-1]:9.2f} {c['ov'][-1]:8.2f} "
        f"{nll[r]:7.2f}{held}"
    )

prof = os.path.join(W, "parameter_ranges", "parameter_ranges.txt")
if os.path.exists(prof):
    print(
        "\nparameter ranges (profile likelihood with the surrogate, around the surrogate calibration):"
    )
    # the table, without the header lines
    print("".join(read(prof).splitlines(keepends=True)[4:]))

# held-out predictions: the surrogate's extremes, and the same parameter sets run with FESTIM
rows = {}
for f in sorted(glob.glob(os.path.join(W, "prediction_ranges", "*.json"))):
    j = json.loads(read(f))
    q, side = j["case"].rsplit("_", 1)
    r, i, _s = split_case(q)
    folder = os.path.join(W, "prediction_ranges", "festim", j["case"])
    try:
        th_case = parameters.read_params(
            os.path.join(W, "prediction_ranges", "params", j["case"] + ".txt")
        )
        fe = chain.forward_festim(th_case, folder, (r,))[r][i][-1]
    except FileNotFoundError:  # not solved with FESTIM yet
        fe = None
    rows.setdefault(q, {})[side] = (j["value"], fe)

if rows:
    print(
        "held-out predictions (final values, Bq): ranges from the surrogate, and the same parameter sets with FESTIM"
    )
    print(
        f"{'':<9} {'surr min':>9} {'max':>7} | {'FESTIM min':>10} {'max':>7} | "
        f"{'final':>6} {'measured':>8}"
    )
    for q, v in rows.items():
        r, i, s = split_case(q)
        fs = " ".join(
            f"{v[k][1]:7.2f}" if v[k][1] is not None else "      -"
            for k in ("min", "max")
        )
        print(
            f"{q:<9} {v['min'][0]:9.2f} {v['max'][0]:7.2f} | {'':>3}{fs} | "
            f"{P[r][i][-1]:6.2f} {experimental_data.curves(r)[s][-1]:8.2f}"
        )
