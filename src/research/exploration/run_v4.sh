#!/bin/bash
# RGB-D sweep: one process per scene (macOS habitat-sim: one Simulator per process), 4 concurrent
cd "$(dirname "$0")"; PY=/opt/anaconda3/envs/indooruav/bin/python; export KMP_DUPLICATE_LIB_OK=TRUE
for s in gibson_Adrian gibson_Anaheim replica_apartment_0 replica_frl_apartment_0 mp3d_17DRP5sb8fy mp3d_1LXtFkjw3qL hm3d_16tymPtM7uS hm3d_12e6joG2B2T; do echo $s; done | \
  xargs -P 4 -I{} sh -c "$PY explore4.py --scene {} --sensor rgbd --steps 1000 --seeds 3 --out results_v4 > logs4/{}__rgbd.log 2>&1"
