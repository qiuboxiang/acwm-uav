#!/bin/bash
# no-return / dead-frontier variants, 4 scenes x 2 seeds x 1000 actions
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1; OUT=../exploration_v5/quick2; mkdir -p $OUT/logs
CFG="carve|plan=hier,pose=vo,map=logodds,noret=0,dead=0;carve+dead|plan=hier,pose=vo,map=logodds,noret=0,dead=1;unk|plan=hier,pose=vo,map=logodds,noret=1,dead=0;unk+dead|plan=hier,pose=vo,map=logodds,noret=1,dead=1"
for s in replica_apartment_0 mp3d_17DRP5sb8fy gibson_Anaheim gibson_Adrian hm3d_16tymPtM7uS replica_frl_apartment_0 mp3d_1LXtFkjw3qL hm3d_12e6joG2B2T; do
  $PY explore5.py --scene $s --steps 1000 --seeds 2 --dist_budget 1e9 --configs "$CFG" --out $OUT > $OUT/logs/$s.log 2>&1 &
done; wait
