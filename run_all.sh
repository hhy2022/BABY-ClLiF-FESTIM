#!/usr/bin/env bash
# Reproduce all results: download the data, build the mesh, calibrate the model, run FESTIM, compute
# the parameter and prediction ranges, and make the figures.
#
#   BABY_PY=/path/to/python ./run_all.sh
#
# BABY_PY is the Python of the conda environment in environment.yml.  A full run takes one to two
# days on 12 cores, mostly FESTIM and the parameter ranges.  The figures and summary.txt are written
# to figures/, everything they are made from to figures/model_output/, and the progress to
# figures/model_output/run_all.log.  If the script is stopped, running it again picks up where it
# left off.
#
# Optional settings: CORES (default 12), START_PARAMS (starting point of the calibration, default
# params/start.txt) and FIG_DIR (output folder, default figures).
set -e
cd "$(dirname "$0")"
export BABY_PY=${BABY_PY:-python}
PY=$BABY_PY
CORES=${CORES:-12}
FIG=${FIG_DIR:-figures}
W=$FIG/model_output
START=${START_PARAMS:-params/start.txt}
CAL=$W/calibrated_surrogate.txt
NO_CYC=$W/calibrated_surrogate_no_cyclotron.txt
mkdir -p "$W"
exec >> "$W/run_all.log" 2>&1
log () { echo "$(date '+%F %T') $*"; }
trap 'log "stopped: a step failed (line $LINENO)"' ERR

# Run FESTIM at one parameter set, then add the holdup layer and compare with the data (downstream.py).
#   solve PARAMS OUT THREADS [RUNS] [refit]
# RUNS limits the chain to its first runs, without the comparison.  With "refit", the holdup
# parameters are first fitted again on the FESTIM release and saved to OUT/params_refit.txt.
solve () {
  local params=$1 out=$2 threads=$3 runs=$4 refit=$5
  mkdir -p "$out"
  OMP_NUM_THREADS=$threads $PY -u baby_festim.py --params "$params" --out "$out" \
    ${runs:+--runs "$runs"} >> "$out/festim.log" 2>&1
  [ -z "$runs" ] || return 0
  if [ -n "$refit" ] && [ ! -f "$out/params_refit.txt" ]; then
    $PY downstream.py "$params" "$out" --refit "$out/params_refit.txt"
  elif [ -n "$refit" ]; then
    $PY downstream.py "$out/params_refit.txt" "$out"
  else
    $PY downstream.py "$params" "$out"
  fi
}

# 1. Data and mesh.
$PY experimental_data.py
[ -f baby_2d.msh ] || { log "mesh"; $PY baby_mesh.py; }

# 2. Calibration on the training runs.  It uses the surrogate, which is fast enough for the
# thousands of model runs this needs (about 1 h).
[ -f "$CAL" ] || { log "calibration"; $PY calibrate.py --start "$START" --out "$CAL" --workers "$CORES"; }
log "calibration done: $(sed -n 2p "$CAL")"

# 3. FESTIM at the calibrated point, with the holdup parameters fitted again on its release.
# Two more chains, with the cyclotron source switched off (eta = 0), are used in the figures: one for
# run 3, and one for run 4 that starts from what the main chain left after run 3.
log "FESTIM at the calibrated point"
sed 's/^eta .*/eta = 0.0   # cyclotron off: figure control/' "$CAL" > "$NO_CYC"
solve "$NO_CYC" "$W/festim_no_cyclotron_run3" 2 1,2,3 &
control=$!
solve "$CAL" "$W/festim" 4 "" refit > "$W/festim_vs_data.txt"
for r in 1 2 3; do
  d=$W/festim_no_cyclotron_run4/run_$r
  [ -f "$d/run_complete" ] || { rm -rf "$d"; mkdir -p "$(dirname "$d")"; cp -r "$W/festim/run_$r" "$d"; }
done
solve "$NO_CYC" "$W/festim_no_cyclotron_run4" 4 1,2,3,4
wait $control
FINAL=$W/festim/params_refit.txt
log "FESTIM done: $(tail -1 "$W/festim_vs_data.txt")"

# 4. Check the surrogate against FESTIM, and set the threshold for the ranges.
$PY compare_surrogate_festim.py --params "$FINAL" --chain "$W/festim" > "$W/compare_surrogate_festim.txt"
log "surrogate vs FESTIM: $(tail -1 "$W/compare_surrogate_festim.txt")"
$PY admissibility_threshold.py "$FINAL" "$W/festim"
THR=$(cat "$W/festim/admissibility_threshold.txt")

# 5. Parameter and prediction ranges, with the surrogate.  The parameter sets at the ends of the
# prediction ranges are then run with FESTIM, four at a time.
[ -f "$W/parameter_ranges/parameter_ranges.txt" ] ||
  $PY parameter_ranges.py --params "$CAL" --threshold "$THR" --outdir "$W/parameter_ranges" --workers "$CORES"
log "parameter ranges done"
[ -f "$W/prediction_ranges/prediction_ranges.txt" ] ||
  $PY prediction_ranges.py --params "$CAL" --parameter-ranges "$W/parameter_ranges" \
    --threshold "$THR" --out "$W/prediction_ranges"
log "prediction ranges done; FESTIM at the extremes"
pids=()
for p in "$W"/prediction_ranges/params/*.txt; do
  solve "$p" "$W/prediction_ranges/festim/$(basename "$p" .txt)" 2 > /dev/null &
  pids+=($!)
  if [ ${#pids[@]} -eq 4 ]; then for pid in "${pids[@]}"; do wait $pid; done; pids=(); fi
done
for pid in "${pids[@]}"; do wait $pid; done

# 6. Figures and summary table.
$PY figures.py "$FIG"
$PY summary.py "$FIG" > "$FIG/summary.txt"
log "all done -> $FIG/"
