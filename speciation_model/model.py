"""The chemical-speciation model of BABY-1L: parameters, rate coefficients, likelihood and the
collection-line holdup.  Shared by the twin (twin.py) and the FESTIM model (festim_model.py), so the
two solve the same equations with the same numbers.

Tritium is born in the melt as TF (T+, bonded to a halide).  Only the reduced form, HT, is volatile
enough to leave through the free surface and the sparge bubbles, and only HT enters the metal.

  melt       S -> TF                         production, uniform over the irradiation windows
             TF -> HT                        k_conv = k_red + k_sp [sparger] + k_x (x/1000)^n_x
             TF -> IV soluble                k_tf_s (free surface) + k_tf_sp [sparger]
             HT -> IV insoluble              k_s (free surface) + e_sp lambda_Henry [sparger]
  interface  J = v_w c_HT - v_back c_m       HT(melt) <-> T(metal) + H(metal)
             v_w = (k_w + k_w_h2 x/1000) V/A_wet,   v_back = v_w K_H c_H / K_S^2 (detailed balance)
  Inconel    diffusion; outer faces  K_r c_H c -> OV insoluble (HT),  k_hto c -> OV soluble (HTO)

x is the H2 fraction of the sweep gas in ppm, c_H = K_S sqrt(p_H2) the hydrogen dissolved in the
Inconel, with p_H2 from x + x_bg (x_bg: hydrogen background of the He).  The melt is well mixed.
Every step is first order in tritium (tracer level).  Inconel and ClLiF properties are the
literature values of the transport model (surrogate.py).

Collection lines: a fraction alpha of each insoluble stream deposits on its line and is released
at k (x/1000) + k_he; the soluble form is collected at once.
"""

import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import experimental_data  # noqa: E402
import surrogate  # noqa: E402

DAY = experimental_data.DAY
BQ = experimental_data.BQ_PER_ATOM
RUNS, TRAIN, HELD = experimental_data.RUNS, experimental_data.TRAIN, experimental_data.HELD
RESET_BEFORE = experimental_data.RESET_BEFORE

V_MELT, A_FREE, A_WET = surrogate.V_MELT, surrogate.A_FREE, surrogate.A_WET
K_B = 1.380649e-23  # [J/K]
SPARGE_SCCM, SPARGE_PSIG = 30.0, 3.0

# name: (group, unit, meaning, lower bound, upper bound, log scale); no bounds = fixed
# fmt: off
SPEC = {
    "tbr":      ("source", "-", "tritium breeding ratio", None, None, True),
    "rho_1":    ("source", "-", "run 1 neutron-rate factor", None, None, True),
    "rho_2":    ("source", "-", "run 2 neutron-rate factor", None, None, True),
    "rho_3":    ("source", "-", "run 3 neutron-rate factor", None, None, True),
    "rho_4":    ("source", "-", "run 4 neutron-rate factor", None, None, True),
    "rho_6":    ("source", "-", "run 6 neutron-rate factor", None, None, True),
    "rho_12":   ("source", "-", "run 12 neutron-rate factor", None, None, True),
    "eta":      ("source", "atoms/C", "tritium per logged cyclotron charge (logged C = true C / 1440)", None, None, True),
    "k_red":    ("chemistry", "1/s", "TF -> HT, always on (reduction by the wall metal and impurities)", 1e-9, 1e-4, True),
    "k_sp":     ("chemistry", "1/s", "extra TF -> HT when gas is bubbled through the melt", 1e-10, 1e-4, True),
    "k_x":      ("chemistry", "1/s", "extra TF -> HT by exchange with H2, at 1000 ppm", 1e-8, 1e-2, True),
    "n_x":      ("chemistry", "-", "order of the exchange in the H2 fraction", 0.1, 1.5, False),
    "k_w":      ("chemistry", "1/s", "HT uptake by the wetted wall, v_w A_wet / V, under He", 1e-10, 1e-1, True),
    "k_w_h2":   ("chemistry", "1/s", "increase of the HT uptake per 1000 ppm H2", 1e-12, 1e-1, True),
    "x_bg":     ("chemistry", "ppm", "hydrogen background of the He sweep gas, as H2", 1e-9, 10.0, True),
    "k_s":      ("release", "1/s", "HT through the free surface, k_L A_free / V", 1e-6, 1e-1, True),
    "e_sp":     ("release", "-", "HT stripped by the bubbles, as a fraction of Henry equilibrium", 1e-5, 1.0, True),
    "k_tf_s":   ("release", "1/s", "TF vapour through the free surface", 1e-12, 1e-5, True),
    "k_tf_sp":  ("release", "1/s", "TF carried out by the bubbles", 1e-12, 1e-5, True),
    "k_hto":    ("release", "m/s", "HTO release velocity of the oxidised outer faces", 1e-12, 1e-4, True),
    "alpha_iv": ("lines", "-", "IV-line deposition fraction of the insoluble stream", 0.0, 0.95, False),
    "alpha_ov": ("lines", "-", "OV-line deposition fraction of the insoluble stream", 0.0, 1.0, False),
    "k_iv":     ("lines", "1/s", "IV holdup release rate at 1000 ppm H2", 1e-8, 1e-3, True),
    "k_ov":     ("lines", "1/s", "OV holdup release rate at 1000 ppm H2", 1e-9, 1e-3, True),
    "k_iv_he":  ("lines", "1/s", "IV holdup release rate under He", 1e-9, 1e-4, True),
    "k_ov_he":  ("lines", "1/s", "OV holdup release rate under He", 1e-9, 1e-4, True),
}
# fmt: on
FREE = [n for n, s in SPEC.items() if s[3] is not None]
LINE_KEYS = [n for n, s in SPEC.items() if s[0] == "lines"]
GROUPS = (("source", "source (fixed at the transport model's calibration)"), ("chemistry", "melt chemistry"),
          ("release", "release paths"), ("lines", "collection-line holdup"))


# ------------------------------------------------------------------ parameter files
def read_params(path):
    th = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                k, v = (s.strip() for s in line.split("=", 1))
                if k not in SPEC:
                    raise ValueError(f"{path}: unknown parameter {k!r}")
                th[k] = float(v)
    missing = [n for n in SPEC if n not in th]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return th


def write_params(path, theta, title):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write("".join(f"# {l}\n" for l in title.strip().splitlines()) + "\n")
        for grp, head in GROUPS:
            f.write(f"# --- {head}\n")
            for n, (g, unit, meaning, *_r) in SPEC.items():
                if g == grp:
                    f.write(f"{n:<9} = {float(theta[n])!r:<24} # [{unit}] {meaning}\n")
            f.write("\n")
    os.replace(tmp, path)


def space(names=FREE):
    return [(n, SPEC[n][3], SPEC[n][4], SPEC[n][5]) for n in names]


def bounds(sp):
    return [(np.log(lo), np.log(hi)) if lg else (lo, hi) for _n, lo, hi, lg in sp]


def to_x(theta, sp):
    return np.array([np.log(theta[n]) if lg else theta[n] for n, _lo, _hi, lg in sp])


def from_x(x, base, sp):
    th = dict(base)
    th.update({n: float(np.exp(v)) if lg else float(v) for v, (n, _lo, _hi, lg) in zip(x, sp)})
    return th


def rel(path):
    return os.path.relpath(path, ROOT)


# ------------------------------------------------------------------ runs and source
def temperature(run):
    return experimental_data.RUN_TEMP_C[run] + 273.15


def ppm(run, t_day):
    sw, p0, p1 = experimental_data.H2_SCHEDULE[run]
    return p0 if sw is None or t_day < sw else p1


def phases(run):
    """[(start_day, end_day, ppm)] of the sweep gas."""
    sw, p0, p1 = experimental_data.H2_SCHEDULE[run]
    te = experimental_data.t_end(run)
    return [(0.0, te, p0)] if sw is None else [(0.0, sw, p0), (sw, te, p1)]


def sources(run, th):
    """[(start_day, end_day, atoms/s)]: the generator irradiation and the logged cyclotron windows (a
    window under one minute is widened to one), each spread uniformly over its window."""
    segs = experimental_data.irradiation_segments(run)
    n = th["tbr"] * experimental_data.neutrons(run) * th[f"rho_{run}"]
    t_irr = experimental_data.irradiation_seconds(run)
    out = [(a, b, n / t_irr) for a, b in segs]
    for e0, e1, q in experimental_data.cyclotron_windows(run):
        e0 = min(e0, e1 - 1.0 / 1440.0)
        if q > 0:
            out.append((e0, e1, th["eta"] * q / ((e1 - e0) * DAY)))
    te = experimental_data.t_end(run)
    return [(a, min(b, te), s) for a, b, s in out if a < te]


def born(run, th):
    return sum((b - a) * DAY * s for a, b, s in sources(run, th))


# ------------------------------------------------------------------ rate coefficients
def lambda_henry(T):
    """[1/s] Stripping rate of dissolved HT by sparge bubbles that leave in Henry equilibrium."""
    Q = SPARGE_SCCM * 1e-6 / 60.0 * (T / 273.15) * (14.7 / (14.7 + SPARGE_PSIG))
    return Q / (V_MELT * surrogate.arrhenius(surrogate.K_H_SALT, T) * K_B * T)


def coefficients(th, run, x):
    """Every rate of the model for run `run` with x ppm H2 in the sweep gas.
    Melt rates [1/s] act on the melt inventory; v_* and k_ht, k_hto [m/s] are face velocities."""
    T = temperature(run)
    sp = 1.0 if experimental_data.HAS_SPARGER[run] else 0.0
    K_H = surrogate.arrhenius(surrogate.K_H_SALT, T)
    K_S = surrogate.arrhenius(surrogate.K_S_INC, T)
    c_H = K_S * np.sqrt(surrogate.h2_pressure_pa(x + th["x_bg"]))
    v_w = (th["k_w"] + th["k_w_h2"] * x / 1000.0) * V_MELT / A_WET
    return dict(
        conv=th["k_red"] + sp * th["k_sp"] + (th["k_x"] * (x / 1000.0) ** th["n_x"] if x > 0 else 0.0),
        ht_surface=th["k_s"],
        ht_bubbles=sp * th["e_sp"] * lambda_henry(T),
        tf_surface=th["k_tf_s"],
        tf_bubbles=sp * th["k_tf_sp"],
        v_w=v_w,
        v_back=v_w * K_H * c_H / K_S**2,
        k_ht=surrogate.arrhenius(surrogate.K_R_INC, T) * c_H,
        k_hto=th["k_hto"],
        D_w=surrogate.arrhenius(surrogate.D_INC, T),
    )


# ------------------------------------------------------------------ hot-zone streams
STREAMS = ("iv_ht_surface", "iv_ht_bubbles", "iv_tf_surface", "iv_tf_bubbles", "ov_ht", "ov_hto")
FORM = {"iv_ht_surface": ("iv", "ins"), "iv_ht_bubbles": ("iv", "ins"), "iv_tf_surface": ("iv", "sol"),
        "iv_tf_bubbles": ("iv", "sol"), "ov_ht": ("ov", "ins"), "ov_hto": ("ov", "sol")}
INVENTORIES = ("TF", "HT", "wall")
# A run's hot zone is a dict: t [day] (t[0] = 0), each stream as the atoms released over the step
# ending at t[i] (entry 0 is 0), and the inventories TF, HT, wall [atoms] at t.


# ------------------------------------------------------------------ collection lines
def _pool(N, inflow, k, dt):
    """dN/dt = inflow - k N over dt with inflow and k constant: (N at the end, amount released)."""
    if k * dt < 1e-12:
        out = k * dt * (N + 0.5 * inflow * dt)
        return N + inflow * dt - out, out
    e = np.exp(-k * dt)
    N_end = N * e + inflow * (1.0 - e) / k
    return N_end, N + inflow * dt - N_end


def collected(run, hz, th, pools):
    """Pass a run's hot-zone streams through the two lines.  pools = [N_iv, N_ov] (updated).
    Returns {(line, form): cumulative atoms on hz["t"]} and the parts of each curve
    {stream or 'iv_line'/'ov_line': cumulative atoms}."""
    t = hz["t"]
    dts = np.diff(t) * DAY
    xs = np.array([ppm(run, m) for m in 0.5 * (t[1:] + t[:-1])])
    parts = {}
    for s in STREAMS:
        line, form = FORM[s]
        frac = (1.0 - th[f"alpha_{line}"]) if form == "ins" else 1.0
        parts[s] = np.concatenate([[0.0], np.cumsum(frac * hz[s][1:])])
    for k, line in enumerate(("iv", "ov")):
        ins = sum(hz[s][1:] for s in STREAMS if FORM[s] == (line, "ins"))
        N, rel_ = pools[k], np.zeros(len(dts))
        for i, dt in enumerate(dts):
            N, rel_[i] = _pool(N, th[f"alpha_{line}"] * ins[i] / dt, th[f"k_{line}"] * xs[i] / 1000.0 + th[f"k_{line}_he"], dt)
        pools[k] = N
        parts[f"{line}_line"] = np.concatenate([[0.0], np.cumsum(rel_)])
    curves = {(l, f): sum(parts[s] for s in STREAMS if FORM[s] == (l, f)) for l in ("iv", "ov") for f in ("ins", "sol")}
    curves[("iv", "ins")] = curves[("iv", "ins")] + parts["iv_line"]
    curves[("ov", "ins")] = curves[("ov", "ins")] + parts["ov_line"]
    return curves, parts


def campaign(th, hot_zone, runs=RUNS):
    """Chain the runs 1 -> 2 -> 3 -> 4 -> 6 (run 12 starts clean).  hot_zone(run, init) returns the
    run's hot zone, init = {TF, HT, wall} [atoms] carried in.  Returns {run: dict(t, curves, parts,
    hz, pools)} for the runs needed to reach `runs`."""
    need = set()
    for r in runs:
        j = RUNS.index(r)
        while j >= 0:
            need.add(RUNS[j])
            if RUNS[j] in RESET_BEFORE:
                break
            j -= 1
    out, init, pools = {}, None, None
    for r in RUNS:
        if init is None or r in RESET_BEFORE:
            init, pools = dict(TF=0.0, HT=0.0, wall=0.0), [0.0, 0.0]
        if r not in need:
            continue
        hz = hot_zone(r, init)
        p_in = list(pools)
        curves, parts = collected(r, hz, th, pools)
        out[r] = dict(t=hz["t"], curves=curves, parts=parts, hz=hz, pools_in=p_in, pools=list(pools), init=dict(init))
        init = {k: float(hz[k][-1]) for k in INVENTORIES}
    return out


# ------------------------------------------------------------------ likelihood
NOISE_FLOOR = 0.3  # [Bq] noise of one insoluble sample (transport model's value)
NOISE_REL = 0.10  # noise proportional to the release in a sample
END_REL = 0.05  # relative uncertainty of the final cumulative insoluble value
VIAL_SIGMA = 0.05  # [Bq] counting noise of one water vial, from the negative blank-subtracted vials of runs 1-4


def measured(run):
    """{(line, form): (sampling times [day], cumulative Bq)}; form 'ins' / 'sol' / 'total'."""
    cr = experimental_data.raw(run)["cumulative_tritium_release"]
    names = {"ins": "insoluble", "sol": "soluble", "total": "total"}
    return {(l.lower(), f): (np.asarray(cr[l]["sampling_times"]["value"], float), np.asarray(cr[l][n]["value"], float))
            for l in ("IV", "OV") for f, n in names.items()}


def clipped_moments(mu, s):
    """Mean and variance of max(0, mu + N(0, s^2))."""
    a = mu / s
    P, p = norm.cdf(a), norm.pdf(a)
    m1 = mu * P + s * p
    m2 = (mu**2 + s**2) * P + mu * s * p
    return m1, np.maximum(m2 - m1**2, 1e-12)


def soluble_as_measured(inc):
    """What the published soluble increment is expected to be for a true increment inc [Bq], and its
    sigma.  A soluble sample is two water vials, each blank-subtracted and set to 0 when negative; the
    tritium is in the first, the second holds only counting noise."""
    e1, v1 = clipped_moments(np.maximum(inc, 0.0), VIAL_SIGMA)
    e2, v2 = clipped_moments(np.zeros_like(inc), VIAL_SIGMA)
    return e1 + e2, np.sqrt(v1 + v2 + (NOISE_REL * inc) ** 2)


def nll_run(res, run):
    """{(line, form): (-logL, chi2, n)} of one run.  Insoluble: every sample with sigma =
    sqrt(0.3^2 + (0.1 model)^2), plus the final value with sqrt(0.3^2 + (0.05 measured)^2).
    Soluble: every sample through the clipped-vial measurement (soluble_as_measured)."""
    meas, out = measured(run), {}
    for key in (("iv", "ins"), ("iv", "sol"), ("ov", "ins"), ("ov", "sol")):
        te, ye = meas[key]
        ym = np.interp(te, res["t"], res["curves"][key] * BQ)
        inc_m = np.diff(np.r_[0.0, ym])
        inc_e = np.diff(np.r_[0.0, ye])
        if key[1] == "sol":
            mu, sig = soluble_as_measured(inc_m)
        else:
            mu, sig = inc_m, np.sqrt(NOISE_FLOOR**2 + (NOISE_REL * inc_m) ** 2)
        z = (inc_e - mu) / sig
        nll, chi2, n = float(np.sum(0.5 * z * z + np.log(sig))), float(np.sum(z * z)), len(z)
        if key[1] == "ins":
            se = np.sqrt(NOISE_FLOOR**2 + (END_REL * ye[-1]) ** 2)
            ze = (ym[-1] - ye[-1]) / se
            nll, chi2, n = nll + 0.5 * ze * ze + np.log(se), chi2 + ze * ze, n + 1
        out[key] = (nll, chi2, n)
    return out


def score(sim, runs):
    """(total -logL, {(line, form): (-logL, chi2, n)} summed over runs)."""
    by = {}
    for r in runs:
        for k, v in nll_run(sim[r], r).items():
            by[k] = tuple(a + b for a, b in zip(by.get(k, (0.0, 0.0, 0)), v))
    return sum(v[0] for v in by.values()), by
