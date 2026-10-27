# BABY-1L tritium transport

> This project started from
> [LIBRA-project/BABY-tritium-transport](https://github.com/LIBRA-project/BABY-tritium-transport)
> and has since been developed independently; the original code is kept in the
> [`baby-tritium-transport-upstream`](https://github.com/hhy2022/BABY-ClLiF-FESTIM/tree/baby-tritium-transport-upstream) branch.

A 2-D axisymmetric [FESTIM](https://github.com/festim-dev/FESTIM) model of tritium transport in BABY-1L,
the 1-litre molten-salt breeding experiment of the [LIBRA project](https://github.com/LIBRA-project).
The model is calibrated on the measured tritium release of runs 1, 2, 3 and 12 and then predicts runs 4
and 6.

Tritium is bred in a ClLiF salt by DT neutrons (and, in some runs, by a cyclotron operated next to the
vessel). It leaves the vessel along two paths, each collected in its own bubbler:

* **IV** (inner vessel): through the free surface of the salt, and with the sparge gas in the runs that
  have a sparger;
* **OV** (outer vessel): into the Inconel 625 wall and out through its outer surface, by recombination
  and by isotope exchange with the H₂ in the sweep gas.

On the way to the bubblers, part of each stream is held up in the collection lines and released later,
faster when the sweep gas contains more H₂. This holdup is a simple 0-D layer on top of the FESTIM
release. All Inconel properties are taken from the literature.

## Getting started

```bash
conda env create -f environment.yml
conda activate baby-tritium-transport-env
BABY_PY=$(which python) ./run_all.sh
```

`run_all.sh` reproduces all results, from downloading the data to the figures. A full run takes one
to two days on 12 cores; if it is stopped, running it again continues where it left off.

To solve the model at one parameter set instead (a few hours):

```bash
python experimental_data.py                                   # download the run data into data/
python baby_mesh.py                                           # build the mesh, baby_2d.msh
python baby_festim.py --params params/final.txt --out my_chain
python downstream.py params/final.txt my_chain                # holdup layer and comparison with the data
```

## How it works

The FESTIM model (`baby_festim.py`) solves the runs one after the other, 1 → 2 → 3 → 4 → 6, each
starting with the tritium the previous one left in the salt and the wall; run 12 starts clean. The
downstream holdup layer (`chain.py`) then turns the release from the vessel into the activity
collected in the IV and OV bubblers.

Calibration and the uncertainty ranges need thousands of model runs, which would take far too long with
FESTIM. For these steps a surrogate (`surrogate.py`) is used: it solves the same equations on a coarse
finite-volume grid of the vessel and takes about a second per run of the whole campaign.
`compare_surrogate_festim.py` checks it against FESTIM; the two agree to within 0.1 Bq along every
curve. All curves in the figures come from FESTIM.

The workflow in `run_all.sh`:

| step | script | time |
|---|---|---|
| download the run data | `experimental_data.py` | seconds |
| build the mesh | `baby_mesh.py` | seconds |
| fit the 17 free parameters to the training runs, using the surrogate | `calibrate.py` | ~1 h |
| solve FESTIM at the fitted point and fit the holdup parameters again on its release | `baby_festim.py`, `downstream.py --refit` | 3–10 h |
| check the surrogate against FESTIM | `compare_surrogate_festim.py` | seconds |
| threshold for the uncertainty ranges | `admissibility_threshold.py` | seconds |
| range of each parameter (profile likelihood) | `parameter_ranges.py` | several hours |
| range of each held-out prediction (prediction profile likelihood); the extremes are then solved with FESTIM | `prediction_ranges.py` | ~1 h + FESTIM |
| figures and summary table | `figures.py`, `summary.py` | minutes |

The figures are written to `figures/`, with everything they are made from in `figures/model_output/`:

| figure | content |
|---|---|
| `fig1_geometry` | the vessel, the sweep gas and the two tritium paths |
| `fig2_mechanism` | the model: FESTIM part, downstream holdup, collection |
| `fig3_training` | the training runs (1, 2, 3, 12) against the measurements |
| `fig4_prediction` | the held-out runs (4, 6) with the range of the prediction |
| `fig5_inventory` | where the tritium is during the campaign, run by run |
| `fig6_pathways_run3` | the tritium collected in run 3, split by where it comes from |
| `fig7_parameter_ranges` | the profile likelihood of every calibrated parameter |

## Likelihood and uncertainty

The data are the cumulative IV and OV activity of each run. The likelihood compares the activity
collected in each sample, with σ = √(0.3² + (0.1 × model)²) Bq, and the final cumulative value of each
line, with σ = √(0.3² + (0.05 × measured)²) Bq. The priors are log-normal: on the tritium breeding
ratio, around the OpenMC value with a 15 % width, and on the neutron rate of each training run, around
its foil measurement with the uncertainty of that measurement. Under pure helium the OV holdup can only
be released by the small H₂ impurity of the gas, so its release rate is limited to that at 10 ppm H₂.

A parameter set is considered consistent with the data when its −log posterior is within
1.92 χ²_red of the best fit (`admissibility_threshold.py`). The parameter and prediction ranges are the
extremes over all such sets.

## Files

| file | content |
|---|---|
| `experimental_data.py` | run conditions and measurements, read from LIBRA's `processed_data.json` of each run; vessel dimensions |
| `baby_mesh.py` | the 2-D axisymmetric mesh of the vessel (gmsh) |
| `baby_festim.py` | the FESTIM model |
| `surrogate.py` | the surrogate of the FESTIM model |
| `parameters.py` | reading and writing parameter files, and the inputs they give FESTIM |
| `chain.py` | the campaign chain: FESTIM or surrogate release, then the downstream holdup layer |
| `festim_output.py` | reads the output of a FESTIM run |
| `likelihood.py` | likelihood, priors and the calibration objective |
| `params/` | `final.txt`, the final calibration; `paper_point.txt`, an earlier one used as the starting point of `calibrate.py` |
| `data/` | cyclotron beam windows (`cyc_windows.json`, made by `logbook_photos/extracted/build_cyc_windows.py`) and the measured neutron rates of runs 4 and 12 (`measured_neutron_rates.json`, from LIBRA pull requests not merged yet); the downloaded run data are cached here |
