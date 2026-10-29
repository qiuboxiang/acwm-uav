#!/bin/bash
# follow-up ablation (same protocol as run_v5_ablation.sh): v1 with the v4-report obstacle inflation, learned selection (3a value,
# 3b oracle imitation), privileged oracle selection (upper bound), and 4a forward-model novelty. Models are scene-fold cross-validated.
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OUT=${OUT:-../exploration_v5/ablation5}; mkdir -p $OUT/logs; FV=../exploration_v5/fv
H="plan=hier,pose=vo,map=logodds"
CFG=${CFG:-"hier+lo+vo learned|$H,sel=learned;hier+lo+vo rank|$H,sel=rank;hier+lo+vo oracle|$H,sel=oracle"}
for s in gibson_Adrian replica_apartment_0 mp3d_17DRP5sb8fy hm3d_16tymPtM7uS; do
  $PY explore5.py --scene $s --steps 3000 --dist_budget 100 --cov_stop 0.97 --seeds 3 --configs "$CFG" --value_model $FV/model_trainB.pt --rank_model $FV/rank_trainB.pt --out $OUT > $OUT/logs/$s.log 2>&1 &
done
for s in gibson_Anaheim replica_frl_apartment_0 mp3d_1LXtFkjw3qL hm3d_12e6joG2B2T; do
  $PY explore5.py --scene $s --steps 3000 --dist_budget 100 --cov_stop 0.97 --seeds 3 --configs "$CFG" --value_model $FV/model_trainA.pt --rank_model $FV/rank_trainA.pt --out $OUT > $OUT/logs/$s.log 2>&1 &
done; wait
