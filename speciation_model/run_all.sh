#!/usr/bin/env bash
# Reproduce the speciation model's results: calibrate with the twin, solve FESTIM at the calibrated
# point, fit the collection lines again on the FESTIM release, compare twin and FESTIM, make the figures.
#
#   BABY_PY=/path/to/python ./run_all.sh
#
# BABY_PY is the Python of the conda environment in ../environment.yml.  Calibration takes about an hour
# on 12 cores, FESTIM about 15 hours (two processes).  Everything goes to OUT_DIR (default out/release),
# the progress to OUT_DIR/run_all.log.  If the script is stopped, running it again picks up where it
# left off.  Optional: CORES (default 12), START_PARAMS (default params/start.txt), OUT_DIR.
set -e
cd "$(dirname "$0")"
PY=${BABY_PY:-python}
CORES=${CORES:-12}
W=${OUT_DIR:-out/release}
START=${START_PARAMS:-params/start.txt}
mkdir -p "$W"
exec >> "$W/run_all.log" 2>&1
log () { echo "$(date '+%F %T') $*"; }
trap 'log "stopped: a step failed (line $LINENO)"' ERR

# 1. Data and mesh (shared with the transport model in the folder above).
(cd .. && $PY experimental_data.py && { [ -f baby_2d.msh ] || $PY baby_mesh.py; })

# 2. Calibration on the training runs, with the twin.
[ -f "$W/calibrated.txt" ] || { log "calibration"; $PY calibrate.py --start "$START" --out "$W/calibrated.txt" --workers "$CORES"; }
log "calibration done: $(sed -n 2p "$W/calibrated.txt")"

# 3. FESTIM at the calibrated point: run 12 (clean start) next to the chain 1 -> 2 -> 3 -> 4 -> 6.
log "FESTIM"
OMP_NUM_THREADS=1 $PY -u festim_model.py --params "$W/calibrated.txt" --out "$W/festim" --runs 12 > "$W/festim_run12.log" 2>&1 &
p12=$!
OMP_NUM_THREADS=1 $PY -u festim_model.py --params "$W/calibrated.txt" --out "$W/festim" --runs 1,2,3,4,6 > "$W/festim_chain.log" 2>&1
wait $p12
log "FESTIM done"

# 4. Collection lines fitted again on the FESTIM release; scores; twin vs FESTIM.
[ -f "$W/final.txt" ] || $PY evaluate.py "$W/calibrated.txt" "$W/festim" --refit "$W/final.txt" > "$W/evaluation.txt"
$PY evaluate.py "$W/final.txt" "$W/festim" > "$W/evaluation.txt"
log "$(grep -- '-logL:' "$W/evaluation.txt")"

# 5. Figures.
$PY figures.py "$W/final.txt" "$W/festim" "$W/figures"
log "all done -> $W/"
