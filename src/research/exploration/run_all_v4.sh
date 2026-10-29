#!/bin/bash
cd "$(dirname "$0")"
./run_v4.sh          # RGB-D sweep (4 procs)
./run_v4_rgb.sh &    # RGB-only sweep (2 procs)
./run_fn.sh &        # FrontierNet baseline (3 procs)
wait
