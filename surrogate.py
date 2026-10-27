"""Surrogate of the FESTIM model (baby_festim.py), for the steps that evaluate the chain many times.

The model is the FESTIM chain, which gives every reported curve.  Calibration, profile likelihood
and prediction ranges need the chain at tens of thousands of parameter sets; there it is replaced by
this surrogate, which solves the same equations, with the same laws on the same faces, by finite
volumes on coarse (r, z) grids of the vessel (melt 203 cells, Inconel 256).  It is checked against
FESTIM along every curve (compare_surrogate_festim.py).

  melt       diffusion, uniform source over the irradiation (and cyclotron) windows, sparge sink
             lambda(t) c, free surface k_d c (-> IV), heater well and inner wall above the melt inert
  interface  J = L(x) [c_s / K_H - (c_m / K_S)^2] on the wetted faces
  Inconel    diffusion, release 2 K_r c^2 + K_r c_H c on the three outer faces (-> OV)

Time: TR-BDF2 for the melt, with every first-order loss inside the operator; the wall is integrated
exactly in its eigenbasis.  The two terms quadratic in the tracer-level concentration are taken
explicitly, so every step is linear.  The step is capped at GRID_CFL / (total loss rate of the
melt).
"""

import h_transport_materials as htm
import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.sparse import csr_matrix

import experimental_data
from experimental_data import (
    R_CLLIF,
    R_HEATER,
    R_INCONEL,
    T_BOTTOM,
    T_CAP,
    T_CLLIF,
    T_GAP,
    T_HEATER_GAP,
)

DAY = experimental_data.DAY

A_FREE = np.pi * (R_CLLIF**2 - R_HEATER**2)
V_MELT = np.pi * R_CLLIF**2 * T_CLLIF - np.pi * R_HEATER**2 * (T_CLLIF - T_HEATER_GAP)
A_WET = np.pi * R_CLLIF**2 + 2 * np.pi * R_CLLIF * T_CLLIF
L_SALT = V_MELT / A_FREE

# ------------------------------------------------------------------ literature properties
# h_transport_materials.  ClLiF takes FLiBe's (Calderoni 2008); Inconel 625: D and K_S
# Gervasini 1984, K_r Perujo 1996 (oxidised).
D_SALT = htm.diffusivities.filter(material="flibe").filter(author="calderoni")[0]
K_H_SALT = htm.solubilities.filter(material="flibe").filter(author="calderoni")[0]
D_INC = (
    htm.diffusivities.filter(material="inconel_625")
    .filter(author="gervasini")
    .filter(isotope="h")[0]
)
K_S_INC = htm.solubilities.filter(material="inconel_625").filter(author="gervasini")[0]
K_R_INC = next(
    p
    for p in htm.recombination_coeffs.filter(material="inconel_625").filter(
        author="perujo"
    )
    if "oxidised" in p.note
)
KB = 8.6173303e-5  # [eV/K] Boltzmann constant, the value FESTIM uses
T_REF_K = 650.0 + 273.15  # [K] Bi sets the salt diffusivity at this temperature


def arrhenius(prop, T_K):
    """An h_transport_materials property at temperature T_K, in the units FESTIM takes."""
    return prop.pre_exp.magnitude * np.exp(-prop.act_energy.magnitude / (KB * T_K))


def h2_pressure_pa(ppm):
    """H2 partial pressure [Pa] of the sweep gas (3 psig)."""
    return (ppm / 1e6) * (3.0 + 14.7) * 6894.76


GRID_CFL = 0.1  # step <= 0.1 / (total loss rate of the melt)
GRID_DT_MIN = 2e-3  # [day] floor of that cap
# [day] base steps: irradiation, first day after, tail
FIT_GRID = dict(dt_irr=1e-2, dt_near=5e-2, dt_far=0.25)
WALL_FACES = ("outer_bottom", "outer_side", "outer_top")  # FESTIM surfaces 31, 32, 33


class Vessel2D:
    """Finite-volume version of baby_2d.msh: the melt and the Inconel on (r, z) grids.

    Melt     R_HEATER <= r <= R_CLLIF over the depth T_CLLIF, plus the disc under the heater tip
             (r < R_HEATER, the lowest T_HEATER_GAP); the heater well is a no-flux hole.
    Inconel  bottom plate (0..R_INCONEL, T_BOTTOM), side wall (R_CLLIF..R_INCONEL, wetted + dry height),
             top cap (R_HEATER..R_INCONEL, T_CAP).
    The grids share their r edges under the melt and their z edges along the wetted wall, so each
    wetted wall face cell faces exactly one melt cell.  Graded toward the free surface, the bottom and
    the side wall, where the boundary layers are.
    """

    def __init__(self, nr=10, nz=20, n_b=3, n_t=4, n_dry=8, n_cap=6, ratio=1.15):
        def graded(n, q):  # n widths summing to 1, the last one the finest
            w = q ** np.arange(n)[::-1]
            return w / w.sum()

        # melt
        r_s = np.concatenate(
            [
                [0.0],
                R_HEATER
                + (R_CLLIF - R_HEATER)
                * np.concatenate([[0.0], np.cumsum(graded(nr, 1.2))]),
            ]
        )
        z_low = T_HEATER_GAP * np.concatenate(
            [[0.0], np.cumsum(graded(n_b, 1.5)[::-1])]
        )
        z_s = np.concatenate(
            [
                z_low,
                T_HEATER_GAP
                + (T_CLLIF - T_HEATER_GAP) * np.cumsum(graded(nz - n_b, ratio)),
            ]
        )
        ncol, nrow = len(r_s) - 1, len(z_s) - 1
        ring = np.pi * (r_s[1:] ** 2 - r_s[:-1] ** 2)
        dzs, drs = np.diff(z_s), np.diff(r_s)
        act = np.ones((nrow, ncol), bool)
        act[n_b:, 0] = False  # heater well
        idx, n = self._index(act)
        self.ns = n
        self.Vs = (ring[None, :] * dzs[:, None])[act]
        Ks = self._laplacian(act, idx, r_s, z_s)
        # banded storage, z-outer / r-inner
        bw = int(max(abs(i - j) for i, j in zip(*np.nonzero(Ks))))
        self.bw = bw
        self.Kb = np.zeros((2 * bw + 1, n))
        ii, jj = np.nonzero(Ks)
        self.Kb[bw + ii - jj, jj] = Ks[ii, jj]
        self.Ks = csr_matrix(Ks)
        self.top_cells, self.top_area, self.top_half = (
            idx[nrow - 1, 1:],
            ring[1:],
            0.5 * dzs[-1],
        )
        self.side_s, self.side_area, self.side_half = (
            idx[:, ncol - 1],
            2 * np.pi * R_CLLIF * dzs,
            0.5 * drs[-1],
        )
        self.bot_s, self.bot_area, self.bot_half = idx[0, :], ring, 0.5 * dzs[0]

        # Inconel
        r_e = np.concatenate([r_s, np.linspace(R_CLLIF, R_INCONEL, n_t + 1)[1:]])
        z_e = np.concatenate(
            [
                np.linspace(0.0, T_BOTTOM, n_t + 1),
                T_BOTTOM + z_s[1:],
                T_BOTTOM + T_CLLIF + np.linspace(0.0, T_GAP, n_dry + 1)[1:],
                T_BOTTOM + T_CLLIF + T_GAP + np.linspace(0.0, T_CAP, n_cap + 1)[1:],
            ]
        )
        nrw, nzw = len(r_e) - 1, len(z_e) - 1
        dr, dz = np.diff(r_e), np.diff(z_e)
        ringw = np.pi * (r_e[1:] ** 2 - r_e[:-1] ** 2)
        i_wall = ncol  # first wall column
        z_bot, z_side = n_t, n_t + nrow + n_dry  # rows: plate, side wall, cap
        actw = np.zeros((nzw, nrw), bool)
        actw[:z_bot, :] = True
        actw[z_bot:z_side, i_wall:] = True
        actw[z_side:, 1:] = True
        idw, nw = self._index(actw)
        self.n = nw
        self.Vc = (ringw[None, :] * dz[:, None])[actw]
        # x D_inc = conductance matrix [m3/s]
        self.K = self._laplacian(actw, idw, r_e, z_e)
        # wetted wall face cells, in the melt's order: side (bottom to top), then bottom plate
        # (axis outward)
        self.src_cells = np.concatenate(
            [idw[z_bot : z_bot + nrow, i_wall], idw[z_bot - 1, :i_wall]]
        )
        self.src_area = np.concatenate([self.side_area, self.bot_area])
        faces = {  # (cells, area, half-cell distance)
            "outer_bottom": [(idw[0, :], ringw, np.full(nrw, 0.5 * dz[0]))],
            "outer_side": [
                (
                    idw[:z_side, nrw - 1],
                    2 * np.pi * R_INCONEL * dz[:z_side],
                    np.full(z_side, 0.5 * dr[-1]),
                )
            ],
            "outer_top": [
                (
                    idw[z_side:, nrw - 1],
                    2 * np.pi * R_INCONEL * dz[z_side:],
                    np.full(nzw - z_side, 0.5 * dr[-1]),
                ),
                (idw[nzw - 1, 1:], ringw[1:], np.full(nrw - 1, 0.5 * dz[-1])),
            ],
        }
        self.f_cell, self.f_area, self.f_half, self.f_id = [], [], [], []
        for k, name in enumerate(WALL_FACES):
            for cells, area, half in faces[name]:
                self.f_cell.append(cells)
                self.f_area.append(area)
                self.f_half.append(half)
                self.f_id.append(np.full(len(cells), k))
        self.f_cell, self.f_area, self.f_half, self.f_id = (
            np.concatenate(x)
            for x in (self.f_cell, self.f_area, self.f_half, self.f_id)
        )
        self._modes_cache = {}

    @staticmethod
    def _index(act):
        idx = -np.ones(act.shape, int)
        idx[act] = np.arange(act.sum())
        return idx, int(act.sum())

    @staticmethod
    def _laplacian(act, idx, r_e, z_e):
        """Finite-volume conductances / D [m] between neighbouring active cells (symmetric, zero row sums)."""
        nzr, nr = act.shape
        rc, zc = 0.5 * (r_e[1:] + r_e[:-1]), 0.5 * (z_e[1:] + z_e[:-1])
        dz = np.diff(z_e)
        ring = np.pi * (r_e[1:] ** 2 - r_e[:-1] ** 2)
        K = np.zeros((int(act.sum()),) * 2)
        for iz in range(nzr):
            for ir in range(nr):
                if not act[iz, ir]:
                    continue
                i = idx[iz, ir]
                if ir + 1 < nr and act[iz, ir + 1]:
                    j = idx[iz, ir + 1]
                    gk = 2 * np.pi * r_e[ir + 1] * dz[iz] / (rc[ir + 1] - rc[ir])
                    K[i, i] += gk
                    K[j, j] += gk
                    K[i, j] -= gk
                    K[j, i] -= gk
                if iz + 1 < nzr and act[iz + 1, ir]:
                    j = idx[iz + 1, ir]
                    gk = ring[ir] / (zc[iz + 1] - zc[iz])
                    K[i, i] += gk
                    K[j, j] += gk
                    K[i, j] -= gk
                    K[j, i] -= gk
        return K

    # ------------------------------------------------------------------ wall state
    def wall_init(self, n_atoms=0.0):
        """Wall state (modes key, vector): (None, W) = nodal concentrations, else modal amplitudes.
        A carried inventory starts uniform, as in the FESTIM chain."""
        return [None, np.full(self.n, float(n_atoms) / float(self.Vc.sum()))]

    def wall_field(self, st):
        return st[1] if st[0] is None else self._modes(*st[0])[2] @ st[1]

    def wall_inventory(self, st):
        return (
            float(self.Vc @ st[1])
            if st[0] is None
            else float(self._modes(*st[0])[6] @ st[1])
        )

    def _modes(self, D_w, ks):
        """Eigen-decomposition of the linear wall operator for one (D_inc, K_r c_H).

        Vc dW/dt = -B W + q with B = D K + S(ks), S the outer-face exchange release through the outer
        half cell.  With W = Vc^-1/2 U y the modes decouple and a step with constant q is exact.
        Returns (mu, T_in, T_out, R_face, T_src, T_face, inv, T_face_in, T_src_in)."""
        key = (float(D_w), float(ks))
        hit = self._modes_cache.get(key)
        if hit is None:
            if len(self._modes_cache) > 64:
                self._modes_cache.clear()
            kf = (
                self.f_area / (1.0 / ks + self.f_half / D_w)
                if ks > 0.0
                else np.zeros_like(self.f_area)
            )
            S = np.zeros(self.n)
            np.add.at(S, self.f_cell, kf)
            sv = np.sqrt(self.Vc)
            mu, U = np.linalg.eigh((D_w * self.K + np.diag(S)) / np.outer(sv, sv))
            T_out = U / sv[:, None]
            T_in = U.T * sv[None, :]
            G = np.zeros((len(WALL_FACES), self.n))
            np.add.at(G, (self.f_id, self.f_cell), kf)
            C = np.ascontiguousarray
            hit = (
                np.maximum(mu, 0.0),
                C(T_in),
                C(T_out),
                G @ T_out,
                C(T_out[self.src_cells].T),
                C(T_out[self.f_cell]),
                self.Vc @ T_out,
                C(T_in[:, self.f_cell]),
                C(T_in[:, self.src_cells]),
            )
            self._modes_cache[key] = hit
        return hit

    def wall_advance(self, st, dt, src, L, K_S, D_w, kr, c_H):
        """Advance the wall by dt with `src` [atoms] entering its wetted face cells over the step.

        Returns (state, released per outer face [atoms], back into the melt per wetted face cell
        [atoms]).  Diffusion and the H2 exchange release are exact in time; the two terms quadratic
        in the tracer-level wall concentration (2 K_r x^2 outside, the interface back term L (x/K_S)^2)
        are 1e-9 of the others and taken explicitly, bounded by the cell content."""
        ks = kr * c_H
        key = (float(D_w), float(ks))
        mu, T_in, T_out, R_face, T_src, T_face, _inv, T_face_in, T_src_in = self._modes(
            *key
        )
        # project onto the modes on the first step, or when the gas changed
        y = st[1] if st[0] == key else T_in @ self.wall_field(st)
        e = np.exp(-np.minimum(mu * dt, 700.0))
        small = mu * dt < 1e-8
        mu_s = np.where(small, 1.0, mu)
        p1 = np.where(small, dt, (1.0 - e) / mu_s)
        qh = T_src @ (src / dt)
        p2 = np.where(small, 0.5 * dt * dt, (dt - p1) / mu_s)
        rel = R_face @ (p1 * y + p2 * qh) if ks > 0.0 else np.zeros(len(WALL_FACES))
        y = e * y + p1 * qh
        if kr > 0.0:  # T + T -> T2 on the outer faces
            wp = np.maximum(T_face @ y, 0.0)
            xf = wp / (1.0 + ks * self.f_half / D_w)
            vf = self.Vc[self.f_cell]
            amt = np.minimum(2.0 * kr * xf * xf * self.f_area * dt, 0.5 * wp * vf)
            y = y - T_face_in @ (amt / vf)
            rel = rel + np.bincount(self.f_id, weights=amt, minlength=len(WALL_FACES))
        back = np.zeros(len(self.src_cells))
        if L > 0.0:  # interface back term, wall -> melt
            wp = np.maximum(y @ T_src, 0.0)
            vs = self.Vc[self.src_cells]
            back = np.minimum(L * (wp / K_S) ** 2 * self.src_area * dt, 0.5 * wp * vs)
            y = y - T_src_in @ (back / vs)
        return [key, y], rel, back


_VESSEL = None


def vessel():
    global _VESSEL
    if _VESSEL is None:
        _VESSEL = Vessel2D()
    return _VESSEL


class BreederModel:
    """The breeder side of one run."""

    def __init__(self, run):
        self.run = run
        self.T_C = experimental_data.RUN_TEMP_C[run]
        self.segs = experimental_data.irradiation_segments(run)
        self.t_irr = sum(b - a for a, b in self.segs) * DAY
        self.sw, self.ppm0, self.ppm1 = experimental_data.H2_SCHEDULE[run]
        self.sparger = experimental_data.HAS_SPARGER[run]

    def ppm(self, t_day):
        if self.sw is None:
            return self.ppm0
        return self.ppm0 if t_day < self.sw else self.ppm1

    def src_on(self, t_day):
        return any(a <= t_day < b for a, b in self.segs)

    def time_grid(self, t_end, ext_events, dt_irr, dt_near, dt_far, dt_cap=None):
        """Time grid [day]: every irradiation, beam-window and gas-switch edge is a grid point; base steps
        dt_irr while irradiating, dt_near for a day after, dt_far later, each capped by dt_cap(t).
        """
        marks = {0.0, float(t_end)}
        for a, b in self.segs:
            marks |= {a, b}
        for e0, e1, _amt in ext_events:
            marks |= {float(e0), float(e1)}
        if self.sw is not None:
            marks.add(self.sw)
        marks = np.array(sorted(m for m in marks if 0.0 <= m <= t_end))
        t_last_irr = max((b for _, b in self.segs), default=0.0)
        parts = [np.array([0.0])]
        for a, b in zip(marks[:-1], marks[1:]):
            mid = 0.5 * (a + b)
            if any(s <= mid < e for s, e in self.segs):
                dt = dt_irr
            elif any(e0 <= mid < e1 for e0, e1, _a in ext_events):
                dt = dt_near
            elif mid < t_last_irr + 1.0:
                dt = dt_near
            else:
                dt = dt_far
            if dt_cap is not None:
                dt = min(dt, float(dt_cap(mid)))
                if self.sw is not None and abs(a - self.sw) < 1e-12:
                    # after a gas switch the loss rate jumps: start from one minute and grow
                    pts, step, x = [], 60.0 / DAY, a
                    while step < dt and x + step < b:
                        x += step
                        pts.append(x)
                        step *= 1.3
                    n = max(1, int(np.ceil((b - x) / dt)))
                    parts.append(np.concatenate([pts, np.linspace(x, b, n + 1)[1:]]))
                    continue
            n = max(1, int(np.ceil((b - a) / dt)))
            parts.append(np.linspace(a, b, n + 1)[1:])
        return np.concatenate(parts)

    def rates(self, p, tm):
        """Gas-dependent coefficients at the step midpoints tm [day]: c_H, L, lambda, K_H, K_S."""
        ppm = np.array([self.ppm(x) for x in tm])
        T_K = self.T_C + 273.15
        K_H = arrhenius(K_H_SALT, T_K)
        K_S = arrhenius(K_S_INC, T_K)
        c_H = K_S * np.sqrt(h2_pressure_pa(ppm))
        L = p["L_he"] + (p["L_h2"] - p["L_he"]) * ppm / 1000.0
        # sparge: lambda_H2 with H2 in the gas, lambda_He under pure He, none without a sparger
        lam = np.where(self.sparger, np.where(ppm > 0, p["lam_h2"], p["lam_he"]), 0.0)
        return c_H, L, lam, K_H, K_S

    def solve(self, p, n_prod_atoms, ext_events=(), n0_salt_atoms=0.0, wall0_atoms=0.0):
        """Solve the run.  p: tau_iv, Bi, lam_h2, lam_he, L_he, L_h2 (values at 650 C).
        ext_events: (start_day, end_day, atoms) bred by the cyclotron, added to the same source.
        Returns cumulative IV / OV [atoms], their rates and the end inventories on the time grid.
        """
        T_K = self.T_C + 273.15
        k_d = V_MELT / (A_FREE * p["tau_iv"])
        D = k_d * L_SALT / p["Bi"] * arrhenius(D_SALT, T_K) / arrhenius(D_SALT, T_REF_K)
        D_w = arrhenius(D_INC, T_K)
        kr = arrhenius(K_R_INC, T_K)
        S0 = n_prod_atoms / (self.t_irr * V_MELT) if self.t_irr > 0 else 0.0
        t_end = experimental_data.t_end(self.run)

        # step cap from the gas-dependent loss rates, evaluated on the intervals between the marks
        marks = self.time_grid(t_end, ext_events, dt_irr=1e9, dt_near=1e9, dt_far=1e9)
        _c, L_m, lam_m, K_H, _K = self.rates(p, 0.5 * (marks[1:] + marks[:-1]))
        k_tot = lam_m + k_d * A_FREE / V_MELT + A_WET * L_m / (K_H * V_MELT)
        edges = marks[1:-1]

        def dt_cap(mid):
            k = float(k_tot[int(np.searchsorted(edges, mid, side="right"))])
            return max(GRID_CFL / k / DAY, GRID_DT_MIN) if k > 0.0 else np.inf

        t = self.time_grid(t_end, ext_events, dt_cap=dt_cap, **FIT_GRID)
        tm = 0.5 * (t[1:] + t[:-1])
        c_H, L_arr, lam, K_H, K_S = self.rates(p, tm)
        S = np.where([self.src_on(x) for x in tm], S0, 0.0)
        for e0, e1, amt in ext_events:  # uniform over the beam window
            S = S + np.where(
                (tm >= e0) & (tm < e1), amt / ((e1 - e0) * DAY * V_MELT), 0.0
            )
        return self._integrate(
            t,
            np.diff(t) * DAY,
            n0_salt_atoms,
            wall0_atoms,
            k_d,
            D,
            D_w,
            kr,
            c_H,
            L_arr,
            lam,
            K_H,
            K_S,
            S,
        )

    def _integrate(
        self,
        t,
        dts,
        n0_salt_atoms,
        wall0_atoms,
        k_d,
        D,
        D_w,
        kr,
        c_H,
        L_arr,
        lam,
        K_H,
        K_S,
        S,
    ):
        """Melt: TR-BDF2 on  Vs dc/dt = -(D K + B) c + Vs S  with every first-order loss in B (sparge,
        free surface on the top faces, forward interface term on the wetted faces, each face in series
        with the half cell behind it).  What passes the wetted faces over a step goes to the wall cell
        behind it; the wall is then advanced exactly and reports its release face by face.
        """
        v = vessel()
        nt, bw, Vs = len(t), v.bw, v.Vs
        V_tot = float(Vs.sum())
        c = np.full(v.ns, float(n0_salt_atoms) / V_tot)
        st = v.wall_init(wall0_atoms)
        iv = np.zeros(nt)
        ov = np.zeros(nt)
        inv = np.zeros(nt)
        r_iv = np.zeros(nt)
        r_ov = np.zeros(nt)
        inv[0] = float(Vs @ c)
        GAM = 2.0 - np.sqrt(2.0)
        CO = (1.0 - GAM) / (2.0 - GAM)
        B1 = 1.0 / (GAM * (2.0 - GAM))
        B2 = (1.0 - GAM) ** 2 / (GAM * (2.0 - GAM))
        k_top = np.zeros(v.ns)  # free surface [m3/s] through the top half cell
        np.add.at(k_top, v.top_cells, v.top_area / (1.0 / k_d + v.top_half / D))
        wet = np.concatenate([v.side_s, v.bot_s])
        wet_half = np.concatenate(
            [np.full(len(v.side_s), v.side_half), np.full(len(v.bot_s), v.bot_half)]
        )
        Kb_u = v.Kb[: bw + 1]
        last = None
        for i in range(nt - 1):
            dt = dts[i]
            a = CO * dt
            v0 = float(L_arr[i]) / K_H
            if (v0, lam[i]) != last:
                kw = (
                    v.src_area / (1.0 / v0 + wet_half / D)
                    if v0 > 0.0
                    else np.zeros(len(wet))
                )
                bdiag = Vs * lam[i] + k_top
                np.add.at(bdiag, wet, kw)
                last = (v0, lam[i])
            ab = (a * D) * Kb_u
            ab[bw] += Vs + a * bdiag
            cb = cholesky_banded(ab, overwrite_ab=True, check_finite=False)
            q = S[i] * Vs
            # stage 1: trapezoidal over GAM dt; stage 2: BDF2 over the rest
            c_s = cho_solve_banded(
                (cb, False),
                Vs * c - a * (D * (v.Ks @ c) + bdiag * c) + GAM * dt * q,
                check_finite=False,
            )
            c_new = cho_solve_banded(
                (cb, False), Vs * (B1 * c_s - B2 * c) + a * q, check_finite=False
            )
            # through the wetted faces over the step
            src = a * kw * (B1 * (c + c_s) + c_new)[wet]
            n_before = float(Vs @ c)
            st, rel, back = v.wall_advance(
                st, dt, src, float(L_arr[i]), K_S, D_w, kr, c_H[i]
            )
            # free surface + sparge
            d_iv = n_before + S[i] * dt * V_tot - float(Vs @ c_new) - float(src.sum())
            np.add.at(c_new, wet, back / Vs[wet])
            c = c_new
            d_ov = float(rel.sum())
            iv[i + 1] = iv[i] + d_iv
            ov[i + 1] = ov[i] + d_ov
            r_iv[i + 1] = d_iv / dt
            r_ov[i + 1] = d_ov / dt
            inv[i + 1] = float(Vs @ c)
        return dict(
            t=t,
            iv=iv,
            ov=ov,
            rate_iv=r_iv,
            rate_ov=r_ov,
            inv=inv,
            wall_end=v.wall_inventory(st),
        )
