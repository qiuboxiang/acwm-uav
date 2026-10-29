#!/bin/bash
# proposal 3a: generate labelled frontier candidates (GT pose, render labels) on all 8 scenes, then train 2-fold scene-split models.
# fold A = gibson_Adrian replica_apartment_0 mp3d_17DRP5sb8fy hm3d_16tymPtM7uS ; fold B = the other four.
# model_trainB.pt (trained on fold B) is used on fold A scenes, model_trainA.pt on fold B scenes; gibson_Adrian never trains its own model.
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1; OUT=../exploration_v5/fv; mkdir -p $OUT/logs
A="gibson_Adrian replica_apartment_0 mp3d_17DRP5sb8fy hm3d_16tymPtM7uS"; B="gibson_Anaheim replica_frl_apartment_0 mp3d_1LXtFkjw3qL hm3d_12e6joG2B2T"
if [ "${SKIP_GEN:-0}" != 1 ]; then
  for s in $A $B; do $PY frontier_value.py gen --scene $s --seeds 3 --seed0 100 --steps 1500 --out $OUT/data > $OUT/logs/gen_$s.log 2>&1 & done; wait
fi
$PY frontier_value.py train --data $OUT/data --scenes ${B// /,} --out $OUT/model_trainB.pt > $OUT/logs/train_B.log 2>&1
$PY frontier_value.py train --data $OUT/data --scenes ${A// /,} --out $OUT/model_trainA.pt > $OUT/logs/train_A.log 2>&1
$PY frontier_value.py train --kind rank --data $OUT/data --scenes ${B// /,} --out $OUT/rank_trainB.pt > $OUT/logs/rank_B.log 2>&1
$PY frontier_value.py train --kind rank --data $OUT/data --scenes ${A// /,} --out $OUT/rank_trainA.pt > $OUT/logs/rank_A.log 2>&1
$PY frontier_value.py eval --data $OUT/data --scenes ${A// /,} --model $OUT/model_trainB.pt --rank_model $OUT/rank_trainB.pt > $OUT/logs/eval_A.log 2>&1
$PY frontier_value.py eval --data $OUT/data --scenes ${B// /,} --model $OUT/model_trainA.pt --rank_model $OUT/rank_trainA.pt > $OUT/logs/eval_B.log 2>&1
cat $OUT/logs/train_*.log $OUT/logs/rank_*.log $OUT/logs/eval_*.log
