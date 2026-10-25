#!/bin/bash
# one process per scene (habitat-sim on macOS: one Simulator per process), 4 concurrent
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/indooruav/bin/python
for s in gibson_Adrian gibson_Anaheim replica_apartment_0 replica_frl_apartment_0 mp3d_17DRP5sb8fy mp3d_1LXtFkjw3qL hm3d_16tymPtM7uS hm3d_12e6joG2B2T; do
  echo $s
done | xargs -P 4 -I{} sh -c "$PY explore2.py --scene {} --steps 500 --seeds 3 --out results > logs/{}.log 2>&1"
