"""How well a parameter set fits the measured release curves: likelihood, priors and the objective
that the calibration minimises."""

import numpy as np

import chain
import experimental_data
import parameters

NOISE_FLOOR = 0.3  # [Bq] noise of one collected sample
NOISE_REL = 0.10  # noise proportional to the release in a sample
END_REL = 0.05  # relative uncertainty of the final cumulative value
TBR_SIGMA = 0.15  # width of the TBR prior around the OpenMC value
HE_IMPURITY_PPM = 10.0  # H2 impurity of the He sweep gas, as ppm


def loglike_run(pred, run):
    """Gaussian log-likelihood of one run.

    Each sample (the release between two collections) is compared with the model, with
    sigma = sqrt(NOISE_FLOOR^2 + (NOISE_REL * model)^2).  The final cumulative value is compared once
    more, with sigma = sqrt(NOISE_FLOOR^2 + (END_REL * measured)^2), so that the total is not lost in
    the noise of the individual samples.
    """
    t, iv, ov = pred
    c = experimental_data.curves(run)
    ll = 0.0
    for s, y in (("iv", iv), ("ov", ov)):
        te, ye = c[f"{s}_t"], c[s]
        ym = np.interp(te, t, y)
        inc_m = np.diff(np.concatenate([[0.0], ym]))
        inc_e = np.diff(np.concatenate([[0.0], ye]))
        sig = np.sqrt(NOISE_FLOOR**2 + (NOISE_REL * inc_m) ** 2)
        z = (inc_e - inc_m) / sig
        ll += float(np.sum(-0.5 * z * z - np.log(sig)))
        se = np.sqrt(NOISE_FLOOR**2 + (END_REL * ye[-1]) ** 2)
        ll += float(-0.5 * ((ym[-1] - ye[-1]) / se) ** 2 - np.log(se))
    return ll


def logprior(th):
    """Log-normal priors on the TBR (around the OpenMC value) and on the neutron-rate factor of each
    training run (around 1, with the uncertainty of the foil measurement)."""
    lp = -0.5 * (np.log(th["tbr"] / experimental_data.openmc_tbr()) / TBR_SIGMA) ** 2
    for r, sig in experimental_data.rate_priors().items():
        lp += -0.5 * (np.log(th[f"rho_{r}"]) / sig) ** 2
    return lp


def he_penalty(th):
    """Penalty that keeps k_ov_he below k_ov * HE_IMPURITY_PPM / 1000: under He the OV holdup can only
    be released by the small H2 impurity in the gas."""
    ex = np.log(th["k_ov_he"] / (th["k_ov"] * HE_IMPURITY_PPM / 1000.0))
    return 1e3 * ex * ex if ex > 0 else 0.0


def objective(th, runs=experimental_data.TRAIN, cache=None):
    """-log posterior on the training runs.  Returns 1e12 if the model fails."""
    try:
        pred = chain.forward(th, runs, cache)
        val = -(sum(loglike_run(pred[r], r) for r in runs) + logprior(th)) + he_penalty(th)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return 1e12
    return val if np.isfinite(val) else 1e12


# ------------------------------------------------------------------ helpers for the calibration scripts
CACHE = chain.StreamCache()


def objective_x(x, base, space=parameters.SPACE):
    """objective() as a function of the optimizer's vector x; parameters not in space come from base."""
    return objective(parameters.from_x(x, base, space), experimental_data.TRAIN, CACHE)


def scores(P):
    """-logL of each run of a prediction, plus the training and held-out totals."""
    nll = {r: -loglike_run(P[r], r) for r in P}
    return (
        nll,
        sum(nll[r] for r in experimental_data.TRAIN if r in nll),
        sum(nll[r] for r in experimental_data.HELD if r in nll),
    )


def simplex(x, space=parameters.SPACE):
    """Initial simplex for Nelder-Mead around x.

    Log-scaled parameters are moved by 0.1 (about 10 %), linear ones by 5 % of their range, stepping
    back instead of forward at the upper bound.  scipy's default step is 5 % of the value itself, which
    is almost nothing for a log-scaled parameter close to 1.
    """
    bnd = parameters.bounds(space)
    pts = [np.array(x, float)]
    for i, ((_n, _lo, _hi, lg), (lo, hi)) in enumerate(zip(space, bnd)):
        step = 0.1 if lg else 0.05 * (hi - lo)
        p = np.array(x, float)
        p[i] = p[i] + step if p[i] + step <= hi else p[i] - step
        pts.append(p)
    return np.array(pts)
