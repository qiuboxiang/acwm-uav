#!/bin/bash
# proposal 5: RGB-only (monocular metric depth), same 4 scenes x 2 seeds x 1000 actions as the v4 RGB-only comparison
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1; OUT=../exploration_v5/rgb; mkdir -p $OUT/logs
CFG="v1|plan=v1,pose=corr,map=sticky;hier+lo+vo|plan=hier,pose=vo,map=logodds;hier+lo+vo safe|plan=hier,pose=vo,map=logodds,safe_lat=0.35,safe_extra=0.45"
for s in gibson_Adrian gibson_Anaheim mp3d_17DRP5sb8fy replica_apartment_0; do
  $PY explore5.py --scene $s --sensor rgb --steps 1000 --seeds 2 --dist_budget 1e9 --cov_stop 2 --configs "$CFG" --out $OUT > $OUT/logs/$s.log 2>&1 &
done; wait
