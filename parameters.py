"""Parameter sets: one plain-text file per set, read by the surrogate (surrogate.py), the FESTIM
model (baby_festim.py) and the holdup layer (chain.py)."""

import os

import numpy as np

import experimental_data
import surrogate

ROOT = experimental_data.ROOT
DEFAULT = os.path.join(ROOT, "params", "final.txt")


def rel(path):
    """A path relative to the repository, for messages and file headers."""
    return os.path.relpath(path, ROOT)


# name: (group, unit, meaning, lower bound, upper bound, log scale); the bounds are the calibration box
# fmt: off
SPEC = {
    # source
    "tbr":      ("source", "-", "tritium breeding ratio", 1e-3, 6e-3, True),
    "rho_1":    ("source", "-", "run 1 neutron-rate factor", 0.4, 3.0, True),
    "rho_2":    ("source", "-", "run 2 neutron-rate factor", 0.4, 3.0, True),
    "rho_3":    ("source", "-", "run 3 neutron-rate factor", 0.4, 3.0, True),
    "rho_12":   ("source", "-", "run 12 neutron-rate factor", 0.4, 3.0, True),
    "eta":      ("source", "atoms/C", "tritium per logged cyclotron charge (logged C = true C / 1440)", 1e11, 1e17, True),
    # breeder side (FESTIM)
    "tau_iv":   ("breeder", "s", "free-surface time constant V_salt / (A_free k_d)", 4e4, 4e6, True),
    "Bi":       ("breeder", "-", "Biot number k_d L_salt / D_salt (sets the salt diffusivity)", 0.015, 15.0, True),
    "lam_h2":   ("breeder", "1/s", "sparge stripping rate with H2 in the gas", 2e-6, 2e-4, True),
    "lam_he":   ("breeder", "1/s", "sparge stripping rate under pure He", 1e-9, 2e-4, True),
    "L_he":     ("breeder", "atoms/m2/s/Pa", "salt|Inconel interface conductance under He", 1e9, 1e16, True),
    "L_h2":     ("breeder", "atoms/m2/s/Pa", "salt|Inconel interface conductance at 1000 ppm H2", 1e9, 1e16, True),
    # downstream holdup layer
    "alpha_iv": ("downstream", "-", "IV-line deposition fraction", 0.0, 1.0, False),
    "alpha_ov": ("downstream", "-", "OV-line deposition fraction", 0.0, 1.0, False),
    "k_iv":     ("downstream", "1/s", "IV holdup release rate at 1000 ppm H2", 1e-9, 1e-4, True),
    "k_ov":     ("downstream", "1/s", "OV holdup release rate at 1000 ppm H2", 1e-9, 1e-4, True),
    "k_ov_he":  ("downstream", "1/s", "OV holdup release rate under He", 1e-10, 1e-5, True),
    "k_iv_he":  ("downstream", "1/s", "IV holdup release rate under He (refitted on FESTIM only)", None, None, True),
    # fixed
    "rho_4":    ("fixed", "-", "run 4 neutron-rate factor", None, None, True),
    "rho_6":    ("fixed", "-", "run 6 neutron-rate factor", None, None, True),
}
# fmt: on
FREE = [n for n, s in SPEC.items() if s[3] is not None]  # calibrated with the surrogate
GROUPS = (
    ("source", "source"),
    ("breeder", "breeder side (FESTIM)"),
    ("downstream", "downstream holdup layer"),
    ("fixed", "fixed"),
)


def read_params(path=DEFAULT):
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


def production(theta, run):
    """Tritium bred in a run [atoms]: TBR x neutrons x rho_run."""
    return theta["tbr"] * experimental_data.neutrons(run) * theta[f"rho_{run}"]


def festim_inputs(theta):
    """The numbers of baby_festim.py for one parameter set.  D is the salt diffusivity at 650 C; FESTIM
    moves it to the run temperature with the literature activation energy, as the surrogate does."""
    k_d = surrogate.V_MELT / (surrogate.A_FREE * theta["tau_iv"])
    D = k_d * surrogate.L_SALT / theta["Bi"]
    return dict(
        k_d=k_d,  # [m/s] free-surface mass transfer
        D_mult=D / surrogate.arrhenius(surrogate.D_SALT, surrogate.T_REF_K),  # salt diffusivity / literature
        L_he=theta["L_he"],
        L_h2=theta["L_h2"],
        lam_h2=theta["lam_h2"],
        lam_he=theta["lam_he"],
        eta=theta["eta"],
        production={r: production(theta, r) for r in experimental_data.RUNS},
    )


# ------------------------------------------------------------------ the calibrated numbers as a vector
SPACE = [(n, SPEC[n][3], SPEC[n][4], SPEC[n][5]) for n in FREE]


def bounds(space=SPACE):
    return [(np.log(lo), np.log(hi)) if lg else (lo, hi) for _n, lo, hi, lg in space]


def to_x(theta, space=SPACE):
    return np.array([np.log(theta[n]) if lg else theta[n] for n, _lo, _hi, lg in space])


def from_x(x, base, space=SPACE):
    th = dict(base)
    th.update({n: float(np.exp(v)) if lg else float(v) for v, (n, _lo, _hi, lg) in zip(x, space)})
    return th
