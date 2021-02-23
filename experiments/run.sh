#!/bin/bash
#echo "run nvidia-smi command to monitor gpu power"

TYPE=$1
BATCH=1
NUM_BATCH=10

while [ $BATCH -lt $NUM_BATCH ]
do
    COMB=${TYPE}_${BATCH}
    python latency_bench.py ${BATCH} ${COMB}
    sleep 10
    ((BATCH++))
done
# & ./pwr_meas.sh ${TYPE} 


