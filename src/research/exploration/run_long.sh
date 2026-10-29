#!/bin/bash
# long-range evaluation to match the flight distances reported in other work (OPAL: 138-320 m to full coverage on HM3D):
# distance budget 300 m, action cap 15000, stop early at 97% coverage. RGB-D, odometry pose (+pose-corr for frontier), up/down enabled.
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE
CFG="random:odom:0:vert=1,frontier1:odom:1:vert=1,frontier2nopan:odom:1:lam=0.04:vert=1,frontier2:odom:1:lam=0.04:pan=6:vert=1,frontier2nopan:odom:1:lam=0.04"
run() { $PY explore4.py --scene $1 --sensor rgbd --steps 15000 --dist_budget 300 --cov_stop 0.97 --seeds 2 --configs "$CFG" --video "" --out results_long > logs_long/$1.log 2>&1; }
for wave in "gibson_Adrian gibson_Anaheim replica_apartment_0 replica_frl_apartment_0" "mp3d_17DRP5sb8fy mp3d_1LXtFkjw3qL hm3d_16tymPtM7uS hm3d_12e6joG2B2T"; do
  for s in $wave; do run $s & done; wait
done
