#!/bin/bash
#echo "run nvidia-smi command to monitor gpu power"

TYPE=$1
BATCH=$2

python latency_bench.py $BATCH ${TYPE}_${BATCH} & ./pwr_meas.sh ${TYPE}_${BATCH} 


