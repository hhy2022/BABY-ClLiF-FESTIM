"""The campaign chain: the breeder side followed by the downstream holdup layer.

The breeder side is the surrogate (forward) or a finished FESTIM chain of baby_festim.py
(forward_festim).  Runs 1, 2, 3, 4 and 6 follow one another and carry the tritium left in the melt, the
wall and the two holdup pools; run 12 starts clean.
"""

import numpy as np

import experimental_data
import festim_output
import parameters
import surrogate

BQ = experimental_data.BQ_PER_ATOM
DAY = experimental_data.DAY
BREEDER_KEYS = ("tau_iv", "Bi", "lam_h2", "lam_he", "L_he", "L_h2")  # the parameters of the breeder side


def cyclotron_events(run, th):
    """Logged beam windows as (start_day, end_day, atoms); a window under one minute is widened to one."""
    out = []
    for e0, e1, q in experimental_data.cyclotron_windows(run):
        if e1 - e0 < 1.0 / 1440.0:
            e0 = e1 - 1.0 / 1440.0
        out.append((e0, e1, th["eta"] * q))
    return tuple(sorted(out))


class StreamCache:
    """Surrogate breeder solutions, keyed on everything they depend on (not on the downstream numbers)."""

    def __init__(self):
        self._store = {}

    def get(self, run, th, n_atoms, events, n0_salt, wall0):
        key = (
            run,
            tuple(float(th[k]) for k in BREEDER_KEYS),
            float(n_atoms),
            events,
            float(n0_salt),
            float(wall0),
        )
        if key not in self._store:
            if len(self._store) > 4096:
                self._store.clear()
            self._store[key] = surrogate.BreederModel(run).solve(th, n_atoms, events, n0_salt, wall0)
        return self._store[key]


def _pool(N, inflow, k, dt):
    """dN/dt = inflow - k N over dt with inflow and k constant: (N at the end, amount released)."""
    if k * dt < 1e-12:
        released = k * dt * (N + 0.5 * inflow * dt)
        return N + inflow * dt - released, released
    e = np.exp(-k * dt)
    N_end = N * e + inflow * (1.0 - e) / k
    return N_end, N + inflow * dt - N_end


def apply_layer(run, res, th, state, record=None):
    """Pass one run's breeder streams through the IV and OV holdup pools.

    A fraction alpha_IV of the IV stream and alpha_OV of the OV stream deposit on the cold
    line in either gas; each pool releases first-order at k (ppm/1000) + k_He, integrated exactly
    over every step.  res: dict with t [day] and rate_iv / rate_ov [atoms/s] over the step ending at t;
    state: N_iv, N_ov [atoms], updated.  record: a list that receives (t [day], N_iv, N_ov, released
    from the IV pool, from the OV pool [atoms]) after every step, if given.
    Returns cumulative IV and OV [atoms] on res["t"]."""
    t = res["t"]
    dt_s = np.diff(t) * DAY
    riv, rov = res["rate_iv"][1:], res["rate_ov"][1:]
    b = surrogate.BreederModel(run)
    ppm = np.array([b.ppm(x) for x in 0.5 * (t[1:] + t[:-1])])
    alpha_iv, alpha_ov = th["alpha_iv"], th["alpha_ov"]
    cum_iv, cum_ov = np.zeros(len(t)), np.zeros(len(t))
    N_iv, N_ov = state["N_iv"], state["N_ov"]
    for i, dt in enumerate(dt_s):
        h2 = ppm[i] / 1000.0
        N_iv, rel_iv = _pool(N_iv, alpha_iv * riv[i], th["k_iv"] * h2 + th["k_iv_he"], dt)
        N_ov, rel_ov = _pool(N_ov, alpha_ov * rov[i], th["k_ov"] * h2 + th["k_ov_he"], dt)
        cum_iv[i + 1] = cum_iv[i] + (1.0 - alpha_iv) * riv[i] * dt + rel_iv
        cum_ov[i + 1] = cum_ov[i] + (1.0 - alpha_ov) * rov[i] * dt + rel_ov
        if record is not None:
            record.append((t[i + 1], N_iv, N_ov, rel_iv, rel_ov))
    state["N_iv"], state["N_ov"] = N_iv, N_ov
    return cum_iv, cum_ov


def _needed(runs):
    """The runs that have to be solved: the scored ones and everything before them back to a reset."""
    need = set()
    for r in runs:
        j = experimental_data.RUNS.index(r)
        while j >= 0:
            need.add(experimental_data.RUNS[j])
            if experimental_data.RUNS[j] in experimental_data.RESET_BEFORE:
                break
            j -= 1
    return need


def forward(th, runs=experimental_data.RUNS, cache=None):
    """{run: (t_day, cumulative IV [Bq], cumulative OV [Bq])} with the surrogate breeder side."""
    cache = cache or StreamCache()
    need, out, state = _needed(runs), {}, None
    for r in experimental_data.RUNS:
        if state is None or r in experimental_data.RESET_BEFORE:
            state = dict(N_iv=0.0, N_ov=0.0, N_salt=0.0, wall=0.0)
        if r not in need:
            continue
        res = cache.get(
            r, th, parameters.production(th, r), cyclotron_events(r, th), state["N_salt"], state["wall"]
        )
        civ, cov = apply_layer(r, res, th, state)
        state["N_salt"], state["wall"] = float(res["inv"][-1]), res["wall_end"]
        if r in runs:
            out[r] = (res["t"], civ * BQ, cov * BQ)
    return out


def forward_festim(th, folder, runs=experimental_data.RUNS):
    """As forward(), on the output folder of a FESTIM chain (baby_festim.py)."""
    need, out, state = _needed(runs), {}, None
    for r in experimental_data.RUNS:
        if state is None or r in experimental_data.RESET_BEFORE:
            state = dict(N_iv=0.0, N_ov=0.0)
        if r not in need:
            continue
        res = festim_output.streams(festim_output.run_folder(folder, r))
        civ, cov = apply_layer(r, res, th, state)
        if r in runs:
            out[r] = (res["t"], civ * BQ, cov * BQ)
    return out
