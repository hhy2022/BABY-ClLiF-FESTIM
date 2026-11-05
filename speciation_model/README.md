# BABY-1L tritium speciation

A 2-D axisymmetric [FESTIM](https://github.com/festim-dev/FESTIM) model in which the chemical form of
the tritium in the ClLiF salt decides how it leaves BABY-1L. It uses the mesh, run data and material
properties of the transport model in the folder above. It is calibrated on runs 1, 2, 3 and 12 and then
predicts runs 4 and 6.

The tritium is born as **TF** (oxidised, bonded to a halide), which can neither leave through the free
surface as a gas nor enter the metal. It first has to be reduced to **HT**. The bubblers collect two forms:

* **insoluble** (vials after the catalyst): HT, from the free surface, the sparge bubbles and the outer
  Inconel wall;
* **soluble** (vials before the catalyst): TF vapour on the IV side, HTO from the oxidised outer wall on
  the OV side.

| step | law |
|---|---|
| TF → HT in the salt | first order, k = k_red + k_sp [sparger] + k_x (x/1000)^n_x |
| salt → IV | HT at k_s (free surface) and e_sp · λ_Henry (bubbles); TF vapour at k_tf_s and k_tf_sp |
| salt ↔ Inconel | J = v_w c_HT − v_back c_m, v_back from detailed balance |
| Inconel → OV | K_r c_H c → HT (insoluble); k_hto c → HTO (soluble) |
| collection lines | a fraction α of each insoluble stream is held up, released at k_he + k (x/1000) |

x is the H₂ in the sweep gas [ppm]. The Inconel properties come from the literature. The tritium source
is fixed at the transport model's calibration.

## Getting started

```bash
conda activate baby-tritium-transport-env        # ../environment.yml
BABY_PY=$(which python) ./run_all.sh
```

`run_all.sh` produces all results in `out/release/`, from the data to the figures. A full run takes
about 15 hours; if it is stopped, running it again continues where it left off.

To solve one parameter set with FESTIM:

```bash
python festim_model.py --params params/start.txt --out my_run
python evaluate.py params/start.txt my_run
```

## How it works

| step | script | time |
|---|---|---|
| fit the free parameters to the training runs, using the twin | `calibrate.py` | ~1 h |
| solve FESTIM at the fitted point | `festim_model.py` | ~15 h |
| fit the collection lines again on the FESTIM release; scores; twin vs FESTIM | `evaluate.py` | ~15 min |
| figures | `figures.py` | a minute |

The calibration needs thousands of model runs, so it uses a twin (`twin.py`): the same equations, with
the salt well mixed and the Inconel on the coarse grid of `../surrogate.py`, integrated exactly. A
campaign takes under a second. FESTIM and the twin agree to within 0.1 Bq along every curve, except in
the first hours of run 12, when the release is fast (0.5 Bq). All figures come from FESTIM.

In FESTIM, the salt | Inconel interface is a flux law rather than continuity. `KineticInterface`
overrides `F.Interface.get_formulation`, so the interface conserves atoms exactly and needs no penalty.

## Files

| file | content |
|---|---|
| `model.py` | parameters, rates, source, collection lines and likelihood, shared by the twin and FESTIM |
| `festim_model.py` | the FESTIM model |
| `twin.py` | the twin used for the calibration |
| `calibrate.py` | calibration on the training runs |
| `evaluate.py` | scores of a FESTIM run, and comparison with the twin |
| `figures.py` | figures |
| `params/start.txt` | the starting point of `calibrate.py` |
