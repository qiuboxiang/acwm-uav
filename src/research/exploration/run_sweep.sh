#!/bin/bash
# lambda / pan-spacing / heading-cost sweep under a flight-distance budget (60 m, action cap 3000); RGB-D, odometry pose + pose-correction
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE
CFG="frontier1:odom:1,frontier2nopan:odom:1:lam=0,frontier2nopan:odom:1:lam=0.02,frontier2nopan:odom:1:lam=0.04,frontier2nopan:odom:1:lam=0.1,frontier2nopan:odom:1:lam=0.3,frontier2nopan:odom:1:lam=0.04:mu=0.3,frontier2:odom:1:lam=0.04:pan=3,frontier2:odom:1:lam=0.04:pan=6,frontier2:odom:1:lam=0.04:pan=10,frontier2:odom:1:lam=0.04:pan=6:mu=0.3"
run() { $PY explore4.py --scene $1 --sensor rgbd --steps 3000 --dist_budget 60 --seeds 3 --configs "$CFG" --video "" --out results_sweep > logs_sweep/$1.log 2>&1; }
for wave in "gibson_Adrian gibson_Anaheim replica_apartment_0 replica_frl_apartment_0" "mp3d_17DRP5sb8fy mp3d_1LXtFkjw3qL hm3d_16tymPtM7uS hm3d_12e6joG2B2T"; do
  for s in $wave; do run $s & done; wait
done
