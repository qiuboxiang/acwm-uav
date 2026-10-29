#!/bin/bash
# replay a collected dataset's poses for evaluation-only coverage / view statistics. usage: run_replay.sh <data_dir> <tag> [workers]
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${3:-3}; OUT=../exploration_v5/replay; mkdir -p $OUT
for k in $(seq 0 $((W-1))); do $PY analyze_collection.py replay --data $1 --scene gibson_Adrian --out $OUT/$2_$k.npz --workers $W --worker $k > $OUT/$2_$k.log 2>&1 & done; wait
