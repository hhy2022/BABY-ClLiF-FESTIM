"""BABY-1L chemical-speciation model: 2-D axisymmetric FESTIM model of the hot zone.

Two species on the (r, z) mesh of baby_mesh.py (the transport model's mesh):
  TF   oxidised tritium, melt only; all production is born as TF
  HT   reduced tritium: the HT molecule in the melt, the dissolved T atom in the Inconel

  melt        source -> TF;  TF -> HT at k_conv (volumetric);  sinks TF -> IV soluble, HT -> IV insoluble
              (sparge bubbles, volumetric), free surface J_TF = kd_TF c_TF, J_HT = kd_HT c_HT
  interface   J = v_w c_HT - v_back c_m                       (kinetic, one species on both sides)
  Inconel     diffusion and Sieverts solubility (Gervasini); outer faces J = (K_r c_H + k_hto) c,
              the K_r c_H part leaves as HT (OV insoluble), the k_hto part as HTO (OV soluble)
  inner faces above the melt, heater well: no flux.

The rates come from model.coefficients(), the same function the twin uses; the ones that depend on
the sweep gas are fem.Constants set before every step.  The melt is well mixed (its diffusivity is
MIX x Calderoni's).  The runs are solved in campaign order (1 -> 2 -> 3 -> 4 -> 6, run 12 clean);
what a run leaves in the melt and the wall is the uniform initial concentration of the next.

    python festim_model.py --params PARAMS --out FOLDER [--runs 1,2,3,4,6,12] [--mesh MESH]

Tested with FESTIM 2.2.0-rc.2, DOLFINx 0.10, h_transport_materials 0.18.
"""

import argparse
import json
import os
import shutil
import warnings

import festim as F
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from scifem import assemble_scalar

import model
import surrogate

# TF exists only in the melt, so the melt | Inconel interface does not apply to it, as intended
warnings.filterwarnings("ignore", message="species TF lives on volume subdomain")

DAY = model.DAY
ID_INCONEL, ID_CLLIF, ID_INTERFACE = 1, 3, 99
SURFACES = {"top_cap": 11, "gap_sidewall": 12, "liquid_surface": 13, "heater_cap": 14, "liquid_heater": 16,
            "axis_liquid": 21, "axis_inconel": 22, "outer_bottom": 31, "outer_side": 32, "outer_top": 33}
SALT_SURFACES = ("liquid_surface", "liquid_heater", "axis_liquid")
OUTER = ("outer_bottom", "outer_side", "outer_top")
GAS_RATES = ("conv", "tf_bubbles", "ht_bubbles", "v_w", "v_back", "k_ht", "k_hto", "k_out")

MIX = 1e4  # melt diffusivity / Calderoni's: a well-mixed melt
DT_MAX = 1800.0  # [s] largest step
DT_RESTART = 0.1  # after a source starts or the gas changes, the step is at most 0.1 x the time since
DT_MIN_CAP = 30.0  # [s] ... but not below this
DT_CFL = 0.05  # and at most 0.05 / (fastest first-order rate of the melt)


class SweepGasProblem(F.HydrogenTransportProblemDiscontinuous):
    """Coefficients that follow the sweep gas are set to their values at the end of each step."""

    def __init__(self, update_gas):
        super().__init__()
        self.update_gas = update_gas

    def iterate(self):
        self.update_gas(float(self.t.value) + float(self.dt.value))
        super().iterate()

    def initialise_exports(self):
        # FESTIM 2.2.0-rc.2 hands a CustomQuantity every species, also one that does not exist in the
        # export's subdomain (TF in the Inconel); give it zero there
        for sp in self.species:
            for vol in self.volume_subdomains:
                if vol not in sp.subdomains:
                    sp.subdomain_to_post_processing_solution.setdefault(vol, ufl.as_ufl(0.0))
        super().initialise_exports()


class KineticInterface(F.Interface):
    """Melt | Inconel exchange of a species present on both sides: J = v_w c_melt - v_back c_metal leaves the
    melt and enters the metal, so the interface conserves atoms exactly."""

    def __init__(self, id, subdomains, v_w, v_back):
        super().__init__(id, subdomains, penalty_term=0.0)
        self.v_w, self.v_back = v_w, v_back

    def get_formulation(self, dS, species, temperature):
        F_melt, F_metal = [], []
        for spe in species:
            c_s, c_m = self.us(spe)
            w_s, w_m = self.vs(spe)
            J = self.v_w * c_s - self.v_back * c_m
            F_melt.append(J * w_s * dS(self.id))
            F_metal.append(-J * w_m * dS(self.id))
        return sum(F_melt[1:], F_melt[0]), sum(F_metal[1:], F_metal[0])


def build_model(run, th, out, init, mesh_file):
    t_end = model.experimental_data.t_end(run) * DAY
    sw = model.experimental_data.H2_SCHEDULE[run][0]
    t_switch = None if sw is None else sw * DAY
    sparger = model.experimental_data.HAS_SPARGER[run]
    src = [(a * DAY, b * DAY, s) for a, b, s in model.sources(run, th)]

    md = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2)
    mesh = md.mesh
    K = {k: fem.Constant(mesh, 0.0) for k in GAS_RATES}
    c0 = model.coefficients(th, run, 0.0)
    kd_HT = c0["ht_surface"] * model.V_MELT / model.A_FREE  # [m/s]
    kd_TF = c0["tf_surface"] * model.V_MELT / model.A_FREE

    _sw, x_before, x_after = model.experimental_data.H2_SCHEDULE[run]

    def update_gas(t):  # a step ending on the switch is before it
        x = x_after if t_switch is not None and t > t_switch + 1e-3 else x_before
        c = model.coefficients(th, run, x)
        c["k_out"] = c["k_ht"] + c["k_hto"]
        for k in GAS_RATES:
            K[k].value = c[k]

    update_gas(0.0)
    m = SweepGasProblem(update_gas)
    m.show_progress_bar = False
    m.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    m.volume_meshtags, m.facet_meshtags = md.cell_tags, md.facet_tags

    salt = F.VolumeSubdomain(id=ID_CLLIF, material=F.Material(
        D_0=surrogate.D_SALT.pre_exp.magnitude * MIX, E_D=surrogate.D_SALT.act_energy.magnitude,
        K_S_0=surrogate.K_H_SALT.pre_exp.magnitude, E_K_S=surrogate.K_H_SALT.act_energy.magnitude, solubility_law="henry"))
    wall = F.VolumeSubdomain(id=ID_INCONEL, material=F.Material(
        D_0=surrogate.D_INC.pre_exp.magnitude, E_D=surrogate.D_INC.act_energy.magnitude,
        K_S_0=surrogate.K_S_INC.pre_exp.magnitude, E_K_S=surrogate.K_S_INC.act_energy.magnitude, solubility_law="sievert"))
    surf = {n: F.SurfaceSubdomain(id=i) for n, i in SURFACES.items()}
    m.subdomains = [wall, salt, *surf.values()]
    m.surface_to_volume = {s: salt if n in SALT_SURFACES else wall for n, s in surf.items()}

    TF = F.Species("TF", mobile=True, subdomains=[salt])
    HT = F.Species("HT", mobile=True, subdomains=[salt, wall])
    m.species = [TF, HT]
    m.interfaces = [KineticInterface(ID_INTERFACE, (salt, wall), K["v_w"], K["v_back"])]
    m.temperature = model.temperature(run)

    x = ufl.SpatialCoordinate(mesh)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=md.cell_tags)
    V_salt, V_wall = (2 * np.pi * assemble_scalar(fem.form(x[0] * dx(i))) for i in (ID_CLLIF, ID_INCONEL))
    m.initial_conditions = [F.InitialConcentration(value=n / V, volume=v, species=s)
                            for n, V, v, s in ((init["TF"], V_salt, salt, TF), (init["HT"], V_salt, salt, HT),
                                               (init["wall"], V_wall, wall, HT)) if n > 0]

    def source(t):  # a step ending at a window's start lies before it
        return sum(s for a, b, s in src if a < t <= b) / V_salt

    m.sources = [F.ParticleSource(value=source, volume=salt, species=TF)]
    m.reactions = [F.ArrheniusReaction(reactant=[TF], product=[HT], k_0=K["conv"], E_k=0.0, p_0=0.0, E_p=0.0, volume=salt)]
    if sparger:
        m.reactions += [F.ArrheniusReaction(reactant=[TF], k_0=K["tf_bubbles"], E_k=0.0, volume=salt),
                        F.ArrheniusReaction(reactant=[HT], k_0=K["ht_bubbles"], E_k=0.0, volume=salt)]

    m.boundary_conditions = [
        F.ParticleFluxBC(subdomain=surf["liquid_surface"], value=lambda c: -kd_TF * c, species=TF, species_dependent_value={"c": TF}),
        F.ParticleFluxBC(subdomain=surf["liquid_surface"], value=lambda c: -kd_HT * c, species=HT, species_dependent_value={"c": HT}),
    ] + [F.ParticleFluxBC(subdomain=surf[f], value=lambda c: -K["k_out"] * c, species=HT, species_dependent_value={"c": HT})
         for f in OUTER]

    restarts = sorted({a for a, _b, _s in src} | ({t_switch} if t_switch is not None else set()))

    def melt_rate(x):  # fastest first-order rate of the melt [1/s]
        c = model.coefficients(th, run, x)
        return max(c["conv"] + c["tf_surface"] + c["tf_bubbles"],
                   c["ht_surface"] + c["ht_bubbles"] + c["v_w"] * model.A_WET / model.V_MELT)

    k_melt = (melt_rate(x_before), melt_rate(x_after))

    def max_stepsize(t):
        since = min((t - r for r in restarts if t >= r - 1e-9), default=np.inf)
        k = k_melt[1 if t_switch is not None and t >= t_switch else 0]
        return float(min(DT_MAX, DT_CFL / k, max(DT_MIN_CAP, DT_RESTART * since)))

    milestones = sorted({t for a, b, _s in src for t in (a, b)} | ({t_switch} if t_switch is not None else set()) | {t_end})
    m.settings = F.Settings(atol=1e-6, rtol=1e-6, transient=True, final_time=t_end, stepsize=F.Stepsize(
        initial_value=10.0, growth_factor=1.1, cutback_factor=0.9, target_nb_iterations=4, max_stepsize=max_stepsize,
        milestones=[t for t in milestones if 0 < t <= t_end], milestone_tolerance=1e-10))

    def axisymmetric(f):  # release rates [atoms/s] and inventories [atoms]: 2 pi r weight
        return lambda **kw: 2 * np.pi * kw["x"][0] * f(kw)

    os.makedirs(out, exist_ok=True)
    q = [("iv_ht_surface", surf["liquid_surface"], lambda k: kd_HT * k["HT"]),
         ("iv_tf_surface", surf["liquid_surface"], lambda k: kd_TF * k["TF"]),
         ("TF", salt, lambda k: k["TF"]), ("HT", salt, lambda k: k["HT"]), ("wall", wall, lambda k: k["HT"])]
    for f in OUTER:
        q += [(f"ov_ht_{f}", surf[f], lambda k: K["k_ht"] * k["HT"]), (f"ov_hto_{f}", surf[f], lambda k: K["k_hto"] * k["HT"])]
    if sparger:
        q += [("iv_ht_bubbles", salt, lambda k: K["ht_bubbles"] * k["HT"]),
              ("iv_tf_bubbles", salt, lambda k: K["tf_bubbles"] * k["TF"])]
    m.exports = [F.CustomQuantity(expr=axisymmetric(f), subdomain=s, title=n, filename=f"{out}/{n}.csv") for n, s, f in q]
    return m


def read_run(d):
    """A finished run folder as model.campaign's hot zone: each stream is rate x step (implicit Euler)."""
    def load(name):
        p = os.path.join(d, f"{name}.csv")
        if not os.path.exists(p):
            return None
        a = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        return a[:, 0], a[:, 1]

    t, _ = load("TF")
    dt = np.diff(np.r_[0.0, t])
    hz = {"t": np.r_[0.0, t] / DAY}
    for s in model.STREAMS:
        files = [f"{s}_{f}" for f in OUTER] if s.startswith("ov_") else [s]
        rate = sum((v[1] for v in map(load, files) if v is not None), np.zeros(len(t)))
        hz[s] = np.r_[0.0, rate * dt]
    with open(os.path.join(d, "carried_in.json")) as f:
        init = json.load(f)
    for k in model.INVENTORIES:
        hz[k] = np.r_[init[k], load(k)[1]]
    return hz


def run_chain(th, out, runs=model.RUNS, mesh_file=os.path.join(model.ROOT, "baby_2d.msh")):
    """Solve the runs in campaign order.  A finished run leaves run_complete and is not solved again."""
    init, missing = dict(TF=0.0, HT=0.0, wall=0.0), None
    for run in model.RUNS:
        if run in model.RESET_BEFORE:
            init, missing = dict(TF=0.0, HT=0.0, wall=0.0), None
        d = os.path.join(out, f"run_{run}")
        if run in runs and missing is not None:
            raise RuntimeError(f"run {run} starts from run {missing}, which is not in {out}")
        if run in runs and not os.path.exists(os.path.join(d, "run_complete")):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            with open(os.path.join(d, "carried_in.json"), "w") as f:
                json.dump(init, f)
            print(f"run {run}: carried in {sum(init.values()) * model.BQ:.3f} Bq", flush=True)
            mdl = build_model(run, th, d, init, mesh_file)
            mdl.initialise()
            mdl.run()
            hz = read_run(d)
            rel = sum(hz[s].sum() for s in model.STREAMS)
            closure = (rel + sum(hz[k][-1] for k in model.INVENTORIES)) / (model.born(run, th) + sum(init.values()))
            with open(os.path.join(d, "run_complete"), "w") as f:
                f.write(f"atom balance (released + left) / (born + carried in) = {closure:.8f}\n")
            print(f"run {run}: done, atom balance {closure:.8f}", flush=True)
        if not os.path.exists(os.path.join(d, "run_complete")):
            missing = missing or run
            continue
        hz = read_run(d)
        init = {k: float(hz[k][-1]) for k in model.INVENTORIES}


def campaign(th, out, runs=model.RUNS):
    """model.campaign on a finished FESTIM chain."""
    return model.campaign(th, lambda r, init: read_run(os.path.join(out, f"run_{r}")), runs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BABY-1L speciation model in FESTIM, runs in campaign order")
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=",".join(map(str, model.RUNS)))
    ap.add_argument("--mesh", default=os.path.join(model.ROOT, "baby_2d.msh"))
    a = ap.parse_args()
    run_chain(model.read_params(a.params), a.out, [int(r) for r in a.runs.split(",")], a.mesh)
