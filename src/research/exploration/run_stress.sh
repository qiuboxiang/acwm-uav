#!/bin/bash
# stress test for deadlocks: 3000 actions / 100 m, start poses restricted to the main free-space component. usage: run_stress.sh <PLAN_DIL> <outdir>
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE PLAN_DIL=$1; OUT=$2; mkdir -p $OUT logs_$OUT
CFG="frontier1:odom:1:vert=1,frontier2nopan:odom:1:lam=0.04:vert=1"
run() { $PY explore4.py --scene $1 --sensor rgbd --steps 3000 --dist_budget 100 --cov_stop 0.97 --seeds 3 --configs "$CFG" --video "" --out $OUT > logs_$OUT/$1.log 2>&1; }
for wave in "gibson_Adrian gibson_Anaheim replica_apartment_0 replica_frl_apartment_0" "mp3d_17DRP5sb8fy mp3d_1LXtFkjw3qL hm3d_16tymPtM7uS hm3d_12e6joG2B2T"; do
  for s in $wave; do run $s & done; wait
done
