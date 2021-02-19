#!/bin/bash
#echo "run nvidia-smi command to monitor gpu power"

TYPE=$1
python latency_bench.py ${TYPE} & ./pwr_meas.sh ${TYPE} 


