#!/bin/bash
# Semantic-enrichment pilot (plan A) on the A40: real SPHN 2026.1 graph at the 24 h landmark, seed 0,
# with random initial embeddings (sphn_real) and with text-label embeddings (sphn_real_text). Run from ~/neurovasc-repro.
G=~/mimic-iv/derived/graphs_L24
N=$(python -c "import joblib; print(len(joblib.load('$G/outcomes.joblib')))")
common=(--prefix $G/sphn_real --outcomes $G/outcomes.joblib --num_patients $N --layers 4 --both_directions
        --emb_std 1.0 --zero_emb_regex 'synth_patient_|/p[0-9]+/|\^\^' --epochs 2000 --device cuda --seed 0
        --results_dir ~/mimic-iv/derived/results)
python -W ignore run_rgcn.py --run MIMIC24_sphn_real "${common[@]}" > logs/MIMIC24_sphn_real.log 2>&1 &
python -W ignore run_rgcn.py --run MIMIC24_sphn_real_text --init_emb $G/sphn_real_init.npy "${common[@]}" \
    > logs/MIMIC24_sphn_real_text.log 2>&1 &
wait
echo PILOT_DONE
