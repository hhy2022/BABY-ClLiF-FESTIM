"""BABY-1L tritium transport: 2-D axisymmetric FESTIM model of the breeder side.

Two materials on the (r, z) mesh of baby_mesh.py, one mobile species (tritium):
  ClLiF melt   diffusion and Henry solubility (Calderoni, FLiBe), D scaled by D_mult;
               source over the irradiation windows and the logged cyclotron beam windows;
               free surface:  J = k_d c                                       -> IV
               sparging:      volumetric sink lambda(t) c                     -> IV
  interface    J = L(x) [c_s / K_H - (c_m / K_S)^2],  L(x) = L_He + (L_1000 - L_He) x / 1000 ppm
  Inconel 625  diffusion and Sieverts solubility (Gervasini);
               outer faces:   T + T -> T2 and T + H -> HT, J = 2 K_r c^2 + K_r c_H c   (Perujo) -> OV
               inner faces above the melt: no flux.
x is the H2 fraction of the sweep gas and c_H = K_S sqrt(p_H2) the hydrogen dissolved at the outer faces.

The runs are solved in campaign order (1 -> 2 -> 3 -> 4 -> 6 -> 12); the tritium left in the salt and in
the wall at the end of a run is the uniform initial concentration of the next, and run 12 starts clean.
The numbers come from a parameter file (parameters.py); the downstream holdup layer is applied to the
exported fluxes afterwards (chain.py).

    python baby_festim.py [--params params/final.txt] [--out figures/model_output/festim]
                          [--runs 1,2,3,4,6,12] [--dt-max-fast 300]

Tested with FESTIM 2.2.0-rc.2 and with FESTIM's main branch of 2026-10-16 (identical results), DOLFINx 0.10,
h_transport_materials 0.18.
"""

import argparse
import os
import shutil

import festim as F
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from scifem import assemble_scalar

import experimental_data
import parameters
import surrogate

ROOT = os.path.dirname(os.path.abspath(__file__))
DAY = experimental_data.DAY

# physical groups of baby_mesh.py
ID_INCONEL, ID_CLLIF = 1, 3
SURFACES = {
    "top_cap": 11,
    "gap_sidewall": 12,
    "liquid_surface": 13,
    "heater_cap": 14,
    "liquid_heater": 16,
    "axis_liquid": 21,
    "axis_inconel": 22,
    "outer_bottom": 31,
    "outer_side": 32,
    "outer_top": 33,
}
SALT_SURFACES = ("liquid_surface", "liquid_heater", "axis_liquid")
OUTER = ("outer_bottom", "outer_side", "outer_top")
ID_INTERFACE = 99

# time stepping
DT_MAX = 7200.0  # [s] largest step
# within FAST_WINDOW after a source or a gas change, while the sparge strips at least FAST_LAMBDA,
# the largest step is --dt-max-fast
FAST_WINDOW = 2 * DAY  # [s]
FAST_LAMBDA = 1e-5  # [1/s]
CYC_DT_MAX = 0.1 * DAY  # [s] largest step inside a cyclotron beam window


def h2_pressure(ppm):
    """H2 partial pressure [Pa] of the sweep gas (3 psig)."""
    return ppm / 1e6 * (3.0 + 14.7) * 6894.76


class SweepGasProblem(F.HydrogenTransportProblemDiscontinuous):
    """Discontinuous problem with coefficients that follow the sweep gas.

    The interface conductance, the sparge rate and the dissolved hydrogen are fem.Constants that
    update_gas(t) sets before every step, t being the end of the step (implicit Euler).  The interface
    penalty_term is not time-dependent in FESTIM, so it is updated here with the others.
    """

    def __init__(self, update_gas):
        super().__init__()
        self.update_gas = update_gas

    def iterate(self):
        self.update_gas(float(self.t.value) + float(self.dt.value))
        super().iterate()


def build_model(run, inputs, out, salt0=0.0, wall0=0.0, dt_max_fast=300.0):
    """FESTIM model of one run.  inputs: parameters.festim_inputs(); salt0, wall0: tritium carried in [atoms]."""
    T_run = experimental_data.RUN_TEMP_C[run] + 273.15
    t_end = experimental_data.t_end(run) * DAY
    switch_day, ppm_before, ppm_after = experimental_data.H2_SCHEDULE[run]
    t_switch = None if switch_day is None else switch_day * DAY
    sparger = experimental_data.HAS_SPARGER[run]
    # Tritium sources in time [s].  DT irradiations: (start, end).  Cyclotron beam-on windows from its log:
    # (start, end, atoms), atoms = eta x logged charge. A window shorter than one minute is widened to one
    # minute (same amount, finite rate).  eta = 0 switches the cyclotron off.
    irradiations = experimental_data.irradiation_segments(run, unit="s")
    beam = []
    if inputs["eta"] > 0:
        for start, end, charge in experimental_data.cyclotron_windows(run):
            start, end = start * DAY, end * DAY
            beam.append((min(start, end - 60.0), end, inputs["eta"] * charge))

    # ---- mesh
    mesh_data = gmshio.read_from_msh(
        os.path.join(ROOT, "baby_2d.msh"), MPI.COMM_WORLD, 0, gdim=2
    )
    mesh = mesh_data.mesh

    # ---- the sweep gas: fem.Constants set before every step
    L = fem.Constant(mesh, 0.0)  # interface conductance [atoms/m2/s/Pa]
    lam = fem.Constant(mesh, 0.0)  # sparge stripping rate [1/s]
    c_H = fem.Constant(mesh, 0.0)  # hydrogen dissolved at the outer faces [atoms/m3]
    K_S_run = surrogate.arrhenius(surrogate.K_S_INC, T_run)

    def update_gas(t):
        # a step ending on the switch is before it
        after = t_switch is not None and t > t_switch + 1e-3
        ppm = ppm_after if after else ppm_before
        L.value = inputs["L_he"] + (inputs["L_h2"] - inputs["L_he"]) * ppm / 1000.0
        lam.value = (
            (inputs["lam_h2"] if ppm > 0 else inputs["lam_he"]) if sparger else 0.0
        )
        c_H.value = K_S_run * np.sqrt(h2_pressure(ppm))

    update_gas(0.0)

    my_model = SweepGasProblem(update_gas)
    my_model.show_progress_bar = False
    my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    my_model.volume_meshtags = mesh_data.cell_tags
    my_model.facet_meshtags = mesh_data.facet_tags

    # ---- materials and subdomains
    salt_material = F.Material(
        D_0=surrogate.D_SALT.pre_exp.magnitude * inputs["D_mult"],
        E_D=surrogate.D_SALT.act_energy.magnitude,
        K_S_0=surrogate.K_H_SALT.pre_exp.magnitude,
        E_K_S=surrogate.K_H_SALT.act_energy.magnitude,
        solubility_law="henry",
    )
    inconel = F.Material(
        D_0=surrogate.D_INC.pre_exp.magnitude,
        E_D=surrogate.D_INC.act_energy.magnitude,
        K_S_0=surrogate.K_S_INC.pre_exp.magnitude,
        E_K_S=surrogate.K_S_INC.act_energy.magnitude,
        solubility_law="sievert",
    )
    salt = F.VolumeSubdomain(id=ID_CLLIF, material=salt_material)
    wall = F.VolumeSubdomain(id=ID_INCONEL, material=inconel)
    surf = {name: F.SurfaceSubdomain(id=i) for name, i in SURFACES.items()}
    my_model.subdomains = [wall, salt, *surf.values()]
    my_model.surface_to_volume = {
        s: salt if name in SALT_SURFACES else wall for name, s in surf.items()
    }

    # FESTIM's penalty interface flux is penalty x (c_s/K_H - (c_m/K_S)^2): the kinetic law with penalty = L
    my_model.interfaces = [F.Interface(ID_INTERFACE, (salt, wall), penalty_term=L)]
    # local equilibrium (Henry | Sieverts continuity) instead:
    # F.Interface(ID_INTERFACE, (salt, wall), penalty_term=10.0, method="nitsche")

    tritium = F.Species("tritium", mobile=True, subdomains=[salt, wall])
    my_model.species = [tritium]
    my_model.temperature = T_run

    # ---- initial inventories, uniform in each material
    x = ufl.SpatialCoordinate(mesh)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags)
    V_salt, V_wall = (
        2 * np.pi * assemble_scalar(fem.form(x[0] * dx(i)))
        for i in (ID_CLLIF, ID_INCONEL)
    )
    my_model.initial_conditions = [
        F.InitialConcentration(value=n / V, volume=v, species=tritium)
        for n, V, v in ((salt0, V_salt, salt), (wall0, V_wall, wall))
        if n > 0
    ]

    # ---- source: bred tritium, uniform in the melt
    rate_irr = inputs["production"][run] / (
        sum(b - a for a, b in irradiations) * V_salt
    )

    def source(t):  # a step ending at a window's start lies before it
        s = rate_irr if any(a < t <= b for a, b in irradiations) else 0.0
        return s + sum(n / ((b - a) * V_salt) for a, b, n in beam if a < t <= b)

    my_model.sources = [F.ParticleSource(value=source, volume=salt, species=tritium)]

    # ---- sparging: first-order sink in the melt; what it removes goes to the IV line
    if sparger:
        my_model.reactions = [
            F.ArrheniusReaction(reactant=[tritium], k_0=lam, E_k=0.0, volume=salt)
        ]

    # ---- boundary conditions
    k_d = inputs["k_d"]
    my_model.boundary_conditions = [
        F.ParticleFluxBC(
            subdomain=surf["liquid_surface"],
            value=lambda c: -k_d * c,
            species=tritium,
            species_dependent_value={"c": tritium},
        ),
    ]
    for face in OUTER:
        my_model.boundary_conditions += [
            # T + T -> T2
            F.SurfaceReactionBC(
                reactant=[tritium, tritium],
                gas_pressure=0.0,
                k_r0=surrogate.K_R_INC.pre_exp.magnitude,
                E_kr=surrogate.K_R_INC.act_energy.magnitude,
                k_d0=0.0,
                E_kd=0.0,
                subdomain=surf[face],
            ),
            # T + H -> HT
            F.SurfaceReactionBC(
                reactant=[tritium],
                gas_pressure=0.0,
                k_r0=surrogate.K_R_INC.pre_exp.magnitude * c_H,
                E_kr=surrogate.K_R_INC.act_energy.magnitude,
                k_d0=0.0,
                E_kd=0.0,
                subdomain=surf[face],
            ),
        ]

    # ---- time stepping: land on every source edge and on the gas switch; small steps where the melt
    # empties fast
    fast = (
        [(0.0, FAST_WINDOW)]
        + [(a, b + FAST_WINDOW) for a, b in irradiations]
        + [(a, b + FAST_WINDOW) for a, b, _n in beam]
    )
    if t_switch is not None:
        fast.append((t_switch, t_switch + FAST_WINDOW))

    def max_stepsize(t):
        dt = DT_MAX
        if (
            dt_max_fast > 0
            and lam.value >= FAST_LAMBDA
            and any(a <= t < b for a, b in fast)
        ):
            dt = min(dt, dt_max_fast)
        for a, b, _n in beam:
            if a <= t < b:
                dt = min(dt, CYC_DT_MAX, 0.5 * (b - a))
        return dt

    milestones = sorted(
        {t for a, b in irradiations for t in (a, b)}
        | {t for a, b, _n in beam for t in (a, b)}
        | ({t_switch} if t_switch is not None else set())
        | {t_end}
    )
    my_model.settings = F.Settings(
        atol=1e-6,
        rtol=1e-6,
        transient=True,
        final_time=t_end,
        stepsize=F.Stepsize(
            initial_value=10.0,
            growth_factor=1.1,
            cutback_factor=0.9,
            target_nb_iterations=4,
            max_stepsize=max_stepsize,
            milestones=[m for m in milestones if m <= t_end],
            # FESTIM's default 1e-5 is 26 s at day 30; beam windows last minutes
            milestone_tolerance=1e-10,
        ),
    )

    # ---- exports: release rates from the boundary laws [atoms/s] and inventories [atoms], as
    # CustomQuantity integrals with the axisymmetric weight 2 pi r (F.SurfaceFlux and F.TotalVolume of
    # FESTIM 2.2.0-rc.2 integrate without it)
    def axisymmetric(integrand):
        return lambda tritium, T, x, **kwargs: 2 * np.pi * x[0] * integrand(tritium, T)

    def K_r(T):
        return surrogate.K_R_INC.pre_exp.magnitude * ufl.exp(
            -surrogate.K_R_INC.act_energy.magnitude / (F.k_B * T)
        )

    os.makedirs(out, exist_ok=True)
    my_model.exports = [
        F.CustomQuantity(
            expr=axisymmetric(lambda c, T: k_d * c),
            subdomain=surf["liquid_surface"],
            title="free surface",
            filename=f"{out}/flux_liquid_surface.csv",
        ),
        *(
            F.CustomQuantity(
                expr=axisymmetric(lambda c, T: 2 * K_r(T) * c**2 + K_r(T) * c_H * c),
                subdomain=surf[face],
                title=face,
                filename=f"{out}/flux_{face}.csv",
            )
            for face in OUTER
        ),
        F.CustomQuantity(
            expr=axisymmetric(lambda c, T: c),
            subdomain=salt,
            title="salt inventory",
            filename=f"{out}/inventory_salt.csv",
        ),
        F.CustomQuantity(
            expr=axisymmetric(lambda c, T: c),
            subdomain=wall,
            title="wall inventory",
            filename=f"{out}/inventory_wall.csv",
        ),
    ]
    if sparger:
        my_model.exports.append(
            F.CustomQuantity(
                expr=axisymmetric(lambda c, T: lam * c),
                subdomain=salt,
                title="sparge",
                filename=f"{out}/flux_sparge.csv",
            )
        )
    # concentration fields (VTX, ~1 GB per run):
    # my_model.exports += [F.VTXSpeciesExport(f"{out}/salt.bp", field=tritium, subdomain=salt),
    #                      F.VTXSpeciesExport(f"{out}/wall.bp", field=tritium, subdomain=wall)]
    return my_model


def last_value(path):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return float(np.atleast_1d(a[a.dtype.names[1]])[-1])


def run_chain(params_file, out, runs=experimental_data.RUNS, dt_max_fast=300.0):
    """Solve the runs in campaign order, each starting from what the previous one left in the salt and the wall
    (nothing before run 12).  A finished run leaves a run_complete file and is not solved again.
    """
    inputs = parameters.festim_inputs(parameters.read_params(params_file))
    carried = (0.0, 0.0)
    for run in runs:
        d = os.path.join(out, f"run_{run}")
        if run in experimental_data.RESET_BEFORE:
            carried = (0.0, 0.0)
        if not os.path.exists(os.path.join(d, "run_complete")):
            shutil.rmtree(d, ignore_errors=True)
            print(
                f"run {run}: starts with {carried[0]:.4e} atoms in the salt, {carried[1]:.4e} in the wall",
                flush=True,
            )
            my_model = build_model(run, inputs, d, *carried, dt_max_fast=dt_max_fast)
            my_model.initialise()
            my_model.run()
            with open(os.path.join(d, "run_complete"), "w") as f:
                f.write("finished\n")
        carried = (
            last_value(f"{d}/inventory_salt.csv"),
            last_value(f"{d}/inventory_wall.csv"),
        )
        print(f"run {run}: done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="BABY-1L FESTIM model, runs solved in campaign order"
    )
    ap.add_argument(
        "--params",
        default=parameters.DEFAULT,
        help="parameter file (default params/final.txt)",
    )
    ap.add_argument(
        "--out", default=os.path.join(ROOT, "figures", "model_output", "festim")
    )
    ap.add_argument("--runs", default=",".join(map(str, experimental_data.RUNS)))
    ap.add_argument(
        "--dt-max-fast",
        type=float,
        default=300.0,
        help="largest step [s] in the fast sparged phases; 0: 7200 s throughout",
    )
    a = ap.parse_args()
    run_chain(a.params, a.out, [int(r) for r in a.runs.split(",")], a.dt_max_fast)
