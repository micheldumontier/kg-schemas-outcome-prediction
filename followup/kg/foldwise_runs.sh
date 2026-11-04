#!/bin/bash
# Fold-wise standardisation and calibration runs (seed 0, 10 folds, the paper's flags) on the GPU machine.
# Usage: followup/kg/foldwise_runs.sh A|B|C|D   (two lanes of two runs each, so at most 2 GPU jobs at once)
# Env: PY (Python of repro/requirements.txt, default python), G (graphs, default ~/mimic-iv/derived/graphs_L24),
#      RES (results dir, default ~/mimic-iv/derived/foldwise/results), PROBS (patient-level probabilities for lanes C
#      and D, default ~/mimic-iv/derived/foldwise/probs, never copied off the machine), LOGS, EPOCHS (default 2000),
#      RUN_FOLDS (optional).
# The tabular runs below use followup/kg/tabular_mimic.py from the repository root, with PYTHONPATH=repro.
#   tabular_mimic.py <features_L24.parquet> $RES MIMIC24 --calibration                    (MIMIC24_<model>_cal)
#   tabular_mimic.py <features_L24.parquet> $RES MIMIC24 --models LR,GBM --suffix _probs --save_probs $PROBS
#   tabular_mimic.py <features_L24.parquet> $RES MIMIC24 --models GBM_recal --suffix '' --save_probs $PROBS
# Then aggregate_foldwise.py summarises $RES and calibration_slopes.py summarises $PROBS (aggregates only).
set -u
cd "$(dirname "$0")/../../repro"
G=${G:-~/mimic-iv/derived/graphs_L24}
PY=${PY:-python}
LOGS=${LOGS:-~/mimic-iv/derived/foldwise/logs}
RES=${RES:-~/mimic-iv/derived/foldwise/results}
EPOCHS=${EPOCHS:-2000}
PROBS=${PROBS:-~/mimic-iv/derived/foldwise/probs}   # lanes C and D: patient-level probabilities, never committed
mkdir -p "$LOGS"
N=$($PY -c "import joblib; print(len(joblib.load('$G/outcomes.joblib')))")
ZERO='synth_patient_|/p[0-9]+/|\^\^'
go() {
  run=$1; graph=$2; layers=$3; shift 3
  extra=(); [ -n "${RUN_FOLDS:-}" ] && extra=(--run_folds "$RUN_FOLDS")
  $PY -W ignore run_rgcn.py --run "$run" --prefix "$G/$graph" --outcomes "$G/outcomes.joblib" --num_patients "$N" \
    --layers "$layers" --emb_std 1.0 --zero_emb_regex "$ZERO" --epochs "$EPOCHS" --device cuda --seed 0 \
    --results_dir "$RES" "${extra[@]}" "$@" >> "$LOGS/$run.log" 2>&1
  echo "finished $run exit=$?" >> "$LOGS/lane_$LANE.log"
}
LANE=$1
case $LANE in
  A) go MIMIC24_caresm_ts_global_cal caresm_ts 4 --calibration
     go MIMIC24_sphn_ts_foldstd sphn_ts 3 --foldwise_std --calibration ;;
  B) go MIMIC24_caresm_ts_foldstd caresm_ts 4 --foldwise_std --calibration
     go MIMIC24_sphn_real_foldstd sphn_real 4 --both_directions --foldwise_std --calibration ;;
  C) go MIMIC24_caresm_ts_probs caresm_ts 4 --calibration --save_probs "$PROBS"
     go MIMIC24_sphn_ts_probs sphn_ts 3 --calibration --save_probs "$PROBS" ;;
  D) go MIMIC24_sphn_real_probs sphn_real 4 --both_directions --calibration --save_probs "$PROBS" ;;
esac
echo "lane $LANE done" >> "$LOGS/lane_$LANE.log"
