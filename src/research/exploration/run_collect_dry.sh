#!/bin/bash
# collection-policy comparison in dry-run mode (poses + labels only, no 720p rendering), gibson_Adrian, 170 episodes each
# (= one worker's share of the 10 h run, same seeds as worker 0 of run_collect_10h.sh). usage: run_collect_dry.sh [variants...]
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OUT=../exploration_v5/collect_dry; mkdir -p $OUT; N=${N:-170}; S0=100000
run() {  # name, script, args
  rm -rf $OUT/$1; mkdir -p $OUT/$1; $PY $2 --scene gibson_Adrian --out $OUT/$1 --episodes $N --seed0 $S0 --dry ${@:3} > $OUT/$1.log 2>&1
}
for v in ${@:-v4 v5_ind v5_sess v5_sess_nodiv}; do
  case $v in
    v4) run v4 collect_data.py & ;;
    v5_ind) run v5_ind collect_data5.py --session_len 1 --div 1 --stats $OUT/v5_ind/stats.jsonl & ;;
    v5_sess) run v5_sess collect_data5.py --session_len 10 --div 1 --stats $OUT/v5_sess/stats.jsonl & ;;
    v5_sess_nodiv) run v5_sess_nodiv collect_data5.py --session_len 10 --div 0 --stats $OUT/v5_sess_nodiv/stats.jsonl & ;;
    v5_sess_wm) run v5_sess_wm collect_data5.py --session_len 10 --div 1 --wm 1 --stats $OUT/v5_sess_wm/stats.jsonl & ;;
    v5_sess_learned) run v5_sess_learned collect_data5.py --session_len 10 --div 1 --sel learned --value_model ../exploration_v5/fv/model_trainB.pt --stats $OUT/v5_sess_learned/stats.jsonl & ;;
  esac
done; wait
