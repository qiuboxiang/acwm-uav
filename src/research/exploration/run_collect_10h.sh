#!/bin/bash
# 10 h (360k frames @10 fps) of H.264 720p video on one scene, 4 parallel workers (one Simulator each)
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE
SCENE=${1:-gibson_Adrian}; OUT=../../../data/collected_10h; mkdir -p $OUT/logs
for w in 0 1 2 3; do
  nohup $PY collect_data.py --scene $SCENE --out $OUT/w$w --episodes ${2:-170} --seed0 $((100000 * (w + 1))) --video > $OUT/logs/w$w.log 2>&1 &
done
