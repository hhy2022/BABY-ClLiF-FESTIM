# BABY-1L tritium transport

> This project started from
> [LIBRA-project/BABY-tritium-transport](https://github.com/LIBRA-project/BABY-tritium-transport)
> and has since been developed independently; the original code is kept in the
> [`baby-tritium-transport-upstream`](https://github.com/hhy2022/BABY-ClLiF-FESTIM/tree/baby-tritium-transport-upstream) branch.

A 2-D axisymmetric [FESTIM](https://github.com/festim-dev/FESTIM) model of tritium transport in BABY-1L,
the 1-liter molten-salt breeding experiment of the [LIBRA project](https://github.com/LIBRA-project).
The model is calibrated on the measured tritium release of runs 1, 2, 3 and 12 and then predicts runs 4
and 6.

Tritium is bred in a ClLiF salt by neutrons from a DT generator. In some runs, a cyclotron operating
next to the vessel also irradiated the salt and added to the tritium. The tritium leaves the vessel
along two paths, each collected in its own bubbler:

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

`run_all.sh` produces all results, from downloading the data to the figures. 
A full run takes one to two days on 12 cores. If it is stopped, running it again continues
where it left off. The number of cores and the output folder can be changed with `CORES` and
`FIG_DIR` (default `figures`), e.g. `CORES=8 FIG_DIR=my_figures BABY_PY=$(which python) ./run_all.sh`.

To calibrate and solve the model once, without the uncertainty ranges (a few hours), run from the
repository folder:

```bash
python experimental_data.py                                  # download the run data into data/
python baby_mesh.py                                          # build the mesh, baby_2d.msh
python calibrate.py --out my_run/calibrated.txt              # about 1 h
python baby_festim.py --params my_run/calibrated.txt --out my_run/festim
python downstream.py my_run/calibrated.txt my_run/festim     # holdup layer and comparison with the data
```

## How it works

The FESTIM model (`baby_festim.py`) solves the runs one after the other, 1 → 2 → 3 → 4 → 6, each
starting with the tritium the previous one left in the salt and the wall. Run 12 starts clean. The
downstream holdup layer (`chain.py`) then turns the release from the vessel into the activity
collected in the IV and OV bubblers.

Calibration and the uncertainty ranges need thousands of model runs, which would take far too long with
FESTIM. For these steps a surrogate (`surrogate.py`) is used: it solves the same equations on a coarse
finite-volume grid of the vessel and takes about a second per run of the whole campaign.
`compare_surrogate_festim.py` checks it against FESTIM. The two agree to within 0.1 Bq along every
curve. All curves in the figures come from FESTIM.

The workflow in `run_all.sh`:

| step | script | time |
|---|---|---|
| download the run data | `experimental_data.py` | seconds |
| build the mesh | `baby_mesh.py` | seconds |
| fit the free parameters to the training runs, using the surrogate | `calibrate.py` | ~1 h |
| solve FESTIM at the fitted point and fit the holdup parameters again on its release | `baby_festim.py`, `downstream.py --refit` | 3–10 h |
| check the surrogate against FESTIM | `compare_surrogate_festim.py` | seconds |
| threshold for the uncertainty ranges | `admissibility_threshold.py` | seconds |
| range of each parameter (profile likelihood) | `parameter_ranges.py` | several hours |
| range of each held-out prediction (prediction profile likelihood); the extremes are then solved with FESTIM | `prediction_ranges.py` | ~1 h + FESTIM |
| figures and summary table | `figures.py`, `summary.py` | minutes |


## Files

| file | content |
|---|---|
| `experimental_data.py` | run conditions and measurements, read from LIBRA's `processed_data.json` of each run |
| `baby_mesh.py` | the 2-D axisymmetric mesh of the vessel (gmsh) |
| `baby_festim.py` | the FESTIM model |
| `surrogate.py` | the surrogate of the FESTIM model |
| `parameters.py` | reading and writing parameter files, and the inputs they give FESTIM |
| `chain.py` | the campaign chain: FESTIM or surrogate release, then the downstream holdup layer |
| `festim_output.py` | reads the output of a FESTIM run |
| `likelihood.py` | likelihood, priors and the calibration objective |
| `params/start.txt` | the starting point of `calibrate.py`, rounded from an earlier calibration |
| `data/` | `cyc_windows.json`, the cyclotron beam windows of each run, from the cyclotron log and the run logbooks. `measured_neutron_rates.json`, the measured neutron rates of runs 4 and 12, from LIBRA pull requests not merged yet. |
