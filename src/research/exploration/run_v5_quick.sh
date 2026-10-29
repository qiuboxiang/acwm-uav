#!/bin/bash
# quick 4-scene sanity check of the v5 components (1000 actions, 1 seed)
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE; OUT=../exploration_v5/quick; mkdir -p $OUT/logs
CFG="v1|plan=v1,pose=corr,map=sticky;hier_sticky|plan=hier,pose=corr,map=sticky;hier_vo|plan=hier,pose=vo,map=logodds;hier_vo_div|plan=hier,pose=vo,map=logodds,div=1"
for s in replica_apartment_0 mp3d_17DRP5sb8fy hm3d_16tymPtM7uS gibson_Anaheim; do
  $PY explore5.py --scene $s --steps 1000 --seeds 1 --dist_budget 1e9 --configs "$CFG" --out $OUT > $OUT/logs/$s.log 2>&1 &
done; wait
