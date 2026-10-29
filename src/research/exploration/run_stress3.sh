#!/bin/bash
# frontier3 (2-D band frontier) deadlock stress test, same protocol as run_stress.sh
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE; OUT=stress_f3; mkdir -p $OUT logs_$OUT
CFG="frontier3:odom:1:lam=0.04:vert=1,frontier3:odom:1:lam=0.04,frontier3:odom:1:lam=0.3:vert=1"
run() { $PY explore4.py --scene $1 --sensor rgbd --steps 3000 --dist_budget 100 --cov_stop 0.97 --seeds 3 --configs "$CFG" --video "" --out $OUT > logs_$OUT/$1.log 2>&1; }
for wave in "gibson_Adrian gibson_Anaheim replica_apartment_0 replica_frl_apartment_0" "mp3d_17DRP5sb8fy mp3d_1LXtFkjw3qL hm3d_16tymPtM7uS hm3d_12e6joG2B2T"; do
  for s in $wave; do run $s & done; wait
done
