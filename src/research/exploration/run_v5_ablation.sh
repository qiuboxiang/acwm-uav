#!/bin/bash
# v5 ablation, stress protocol (3000 actions / 100 m, stop at 97 %), 8 scenes x 3 paired seeds, RGB-D. One process per scene.
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OUT=${OUT:-../exploration_v5/ablation2}; mkdir -p $OUT/logs
CFG=${CFG:-"random|plan=random,pose=odom,map=sticky;v1|plan=v1,pose=corr,map=sticky;f3|plan=f3,pose=corr,map=sticky;v1+lo|plan=v1,pose=corr,map=logodds;v1+lo+vo|plan=v1,pose=vo,map=logodds;hier|plan=hier,pose=corr,map=sticky;hier+lo+vo|plan=hier,pose=vo,map=logodds;hier+lo+vo+div|plan=hier,pose=vo,map=logodds,div=1;hier+lo+vo ratio|plan=hier,pose=vo,map=logodds,sel=ratio;hier+lo+vo infl0|plan=hier,pose=vo,map=logodds,infl=0;hier+lo GT|plan=hier,pose=gt,map=logodds"}
for s in ${SCENES:-gibson_Adrian gibson_Anaheim replica_apartment_0 replica_frl_apartment_0 mp3d_17DRP5sb8fy mp3d_1LXtFkjw3qL hm3d_16tymPtM7uS hm3d_12e6joG2B2T}; do
  $PY explore5.py --scene $s --steps ${STEPS:-3000} --dist_budget ${DIST:-100} --cov_stop 0.97 --seeds ${SEEDS:-3} --configs "$CFG" --out $OUT > $OUT/logs/$s.log 2>&1 &
done; wait
