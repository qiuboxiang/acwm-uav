#!/bin/bash
# vertical-motion evaluation: each policy with and without up/down; RGB-D, odometry pose (+pose-corr for frontier), 1000 executed actions
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE
CFG="random:odom:0,random:odom:0:vert=1,frontier1:odom:1,frontier1:odom:1:vert=1,frontier2nopan:odom:1:lam=0.04,frontier2nopan:odom:1:lam=0.04:vert=1"
run() { $PY explore4.py --scene $1 --sensor rgbd --steps 1000 --seeds 3 --configs "$CFG" --video "frontier1:odom:1:vert=1" --out results_vert > logs_vert/$1.log 2>&1; }
for wave in "gibson_Adrian gibson_Anaheim" "replica_apartment_0 replica_frl_apartment_0" "mp3d_17DRP5sb8fy mp3d_1LXtFkjw3qL" "hm3d_16tymPtM7uS hm3d_12e6joG2B2T"; do
  for s in $wave; do run $s & done; wait
done
