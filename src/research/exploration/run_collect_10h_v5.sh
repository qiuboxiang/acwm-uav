#!/bin/bash
# v5 rerun of the single-scene 10 h collection: same scene, 4 workers x 170 episodes, same seeds and H.264 720p output as
# run_collect_10h.sh, but with the v5 agent. Extra policy flags (chosen from the dry-run comparison) are passed through.
# usage: run_collect_10h_v5.sh [scene] [episodes] [extra collect_data5.py flags...]
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
SCENE=${1:-gibson_Adrian}; N=${2:-170}; OUT=../../../data/collected_10h_v5; mkdir -p $OUT/logs
for w in 0 1 2 3; do
  nohup $PY collect_data5.py --scene $SCENE --out $OUT/w$w --episodes $N --seed0 $((100000 * (w + 1))) --video --stats $OUT/w$w.stats.jsonl ${@:3} > $OUT/logs/w$w.log 2>&1 &
done; wait
