"""Twin of the FESTIM model (festim_model.py) for the calibration, which needs the campaign at
thousands of parameter sets.

The same equations as festim_model.py: the melt is well mixed (two inventories, TF and HT), the
Inconel is the 256-cell finite-volume grid of the transport model's surrogate (surrogate.Vessel2D),
with the interface law on the wetted faces and the release laws on the outer faces, each in series
with the half cell behind the face.  Everything is linear, so with the gas and the source constant
over a step the system is integrated exactly in its eigenbasis.  It is checked against FESTIM along
every curve (evaluate.py).
"""

import numpy as np

import model
import surrogate
from experimental_data import R_CLLIF, R_INCONEL, T_BOTTOM

GRID_IRR = 0.01  # [day] output step while a source is on
GRID = 0.05  # [day] output step otherwise
GRID_MIN = 1.0 / 1440  # [day] first step after a source edge or a gas switch

_V = None


def _geometry():
    """Wall grid of surrogate.Vessel2D and the half-cell distances behind its wetted faces."""
    global _V
    if _V is None:
        v = surrogate.vessel()
        n_side = len(v.side_s)
        n_t = 4  # wall cells across the side wall and the bottom plate (Vessel2D default)
        half = np.r_[np.full(n_side, 0.5 * (R_INCONEL - R_CLLIF) / n_t), np.full(len(v.src_cells) - n_side, 0.5 * T_BOTTOM / n_t)]
        _V = (v, half)
    return _V


def operator(c):
    """dx/dt = M x + e_TF S, x = [TF, HT, wall cells] in atoms; R @ x = the stream rates [atoms/s]."""
    v, wet_half = _geometry()
    nw = v.n
    n = 2 + nw
    M = np.zeros((n, n))
    W = slice(2, n)
    out_tf = c["tf_surface"] + c["tf_bubbles"]
    M[0, 0] = -(c["conv"] + out_tf)
    M[1, 0] = c["conv"]
    M[1, 1] = -(c["ht_surface"] + c["ht_bubbles"])
    # wetted faces: J = v_w c_HT - v_back c_face, c_face behind a half cell of the wall
    D_w = c["D_w"]
    g = v.src_area * (D_w / wet_half) / (c["v_back"] + D_w / wet_half)
    up = g * c["v_w"] / model.V_MELT
    back = g * c["v_back"] / v.Vc[v.src_cells]
    M[1, 1] -= up.sum()
    np.add.at(M[1], 2 + v.src_cells, back)
    np.add.at(M, (2 + v.src_cells, 1), up)
    np.add.at(M, (2 + v.src_cells, 2 + v.src_cells), -back)
    M[W, W] -= D_w * v.K / v.Vc[None, :]
    # outer faces: (K_r c_H + k_hto) c_face
    ks = c["k_ht"] + c["k_hto"]
    kf = v.f_area / (1.0 / ks + v.f_half / D_w) if ks > 0 else np.zeros(len(v.f_area))
    np.add.at(M, (2 + v.f_cell, 2 + v.f_cell), -kf / v.Vc[v.f_cell])
    R = np.zeros((len(model.STREAMS), n))
    R[0, 1], R[1, 1] = c["ht_surface"], c["ht_bubbles"]
    R[2, 0], R[3, 0] = c["tf_surface"], c["tf_bubbles"]
    if ks > 0:
        np.add.at(R[4], 2 + v.f_cell, kf / v.Vc[v.f_cell] * c["k_ht"] / ks)
        np.add.at(R[5], 2 + v.f_cell, kf / v.Vc[v.f_cell] * c["k_hto"] / ks)
    return M, R


def time_grid(run, src):
    """Output times [day]: every source edge and gas switch is a grid point; after each, the step grows
    from GRID_MIN as 0.1 x the time since, up to GRID_IRR while a source is on and GRID otherwise."""
    te = model.experimental_data.t_end(run)
    marks = {0.0, te} | {p[0] for p in model.phases(run)}
    for a, b, _s in src:
        marks |= {a, b}
    marks = sorted(m for m in marks if 0.0 <= m <= te)
    t = [0.0]
    for a, b in zip(marks[:-1], marks[1:]):
        cap = GRID_IRR if any(s0 <= 0.5 * (a + b) < s1 for s0, s1, _s in src) else GRID
        x = a
        while x < b - 1e-12:
            dt = min(cap, max(GRID_MIN, 0.1 * (x - a)))
            n_left = (b - x) / dt
            if n_left <= 1.0 + 1e-9 or dt >= cap:
                n = max(1, int(np.ceil(n_left - 1e-9)))
                t.extend(np.linspace(x, b, n + 1)[1:])
                break
            x += dt
            t.append(x)
    return np.array(t)


def _phi(lam, dt):
    """int_0^dt e^{lam s} ds and int_0^dt int_0^s e^{lam u} du ds, safe for small |lam dt|."""
    z = lam * dt
    small = np.abs(z) < 1e-6
    zs = np.where(small, 1.0, z)
    p1 = np.where(small, dt * (1 + z / 2), np.expm1(zs) / zs * dt)
    p2 = np.where(small, dt * dt * (0.5 + z / 6), (p1 - dt) / np.where(small, 1.0, lam))
    return p1, p2


def hot_zone(th, run, init):
    """The run's hot zone (model.campaign's format) from the carried-in inventories init."""
    v, _ = _geometry()
    src = model.sources(run, th)
    t = time_grid(run, src)
    nt, n = len(t), 2 + v.n
    hz = {s: np.zeros(nt) for s in model.STREAMS}
    inv = np.zeros((nt, 3))
    x = np.zeros(n)
    x[0], x[1] = init["TF"], init["HT"]
    x[2:] = init["wall"] * v.Vc / v.Vc.sum()
    inv[0] = x[0], x[1], x[2:].sum()
    tm = 0.5 * (t[1:] + t[:-1])
    S = np.array([sum(s for a, b, s in src if a <= m < b) for m in tm])
    for p0, p1, x_ppm in model.phases(run):
        idx = np.nonzero((tm >= p0) & (tm < p1))[0]
        if len(idx) == 0:
            continue
        M, R = operator(model.coefficients(th, run, x_ppm))
        lam, U = np.linalg.eig(M)
        Ui = np.linalg.inv(U)
        b = Ui[:, 0]
        RU = R @ U
        IU = np.vstack([U[0], U[1], U[2:].sum(axis=0)])
        y = Ui @ x
        for i in idx:
            dt = (t[i + 1] - t[i]) * model.DAY
            p1, p2 = _phi(lam, dt)
            iy = p1 * y + p2 * b * S[i]
            y = np.exp(lam * dt) * y + p1 * b * S[i]
            rel = (RU @ iy).real
            for k, s in enumerate(model.STREAMS):
                hz[s][i + 1] = rel[k]
            inv[i + 1] = (IU @ y).real
        x = (U @ y).real
    hz["t"] = t
    hz["TF"], hz["HT"], hz["wall"] = inv.T
    return hz


def campaign(th, runs=model.RUNS):
    return model.campaign(th, lambda r, init: hot_zone(th, r, init), runs)
