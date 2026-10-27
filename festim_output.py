"""Release streams of a run folder written by baby_festim.py.

IV = free surface + sparging; OV = the three outer Inconel faces.  Each exported rate is the rate at
the end of a step, so the release over a step is rate x step length (FESTIM uses implicit Euler)."""

import os

import numpy as np

import experimental_data

IV_FILES = ("flux_liquid_surface.csv", "flux_sparge.csv")
OV_FILES = ("flux_outer_bottom.csv", "flux_outer_side.csv", "flux_outer_top.csv")


def run_folder(chain, run):
    """The folder of a finished run in a FESTIM chain."""
    d = os.path.join(chain, f"run_{run}")
    if not os.path.exists(os.path.join(d, "run_complete")):
        raise FileNotFoundError(f"run {run} in {chain} is missing or unfinished")
    return d


def _total_rate(d, files):
    """Time [s] and summed rate [atoms/s] of the flux files in d (flux_sparge.csv exists only for runs
    with a sparger)."""
    t, total = None, 0.0
    for f in files:
        path = os.path.join(d, f)
        if os.path.exists(path):
            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            t, total = data[:, 0], total + data[:, 1]
    return t, total


def streams(d):
    """dict(t [day], rate_iv, rate_ov [atoms/s]): t starts at 0 and rate[i] is the release rate over
    the step ending at t[i]."""
    t, rate_iv = _total_rate(d, IV_FILES)
    _, rate_ov = _total_rate(d, OV_FILES)
    return dict(
        t=np.r_[0.0, t] / experimental_data.DAY,
        rate_iv=np.r_[0.0, rate_iv],
        rate_ov=np.r_[0.0, rate_ov],
    )
