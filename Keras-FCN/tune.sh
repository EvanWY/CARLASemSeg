#!/bin/bash 
for i in `seq 0.5 0.01 0.6`;
do
    ROAD_TH=0.57
    VEH_TH=0.20
    echo "testing with ROAD_TH=$ROAD_TH, VEH_TH=$VEH_TH"
    echo "WY_echo" | grader "python run.py $ROAD_TH $VEH_TH" | grep "Car F score:" | (echo -n "road_th: $ROAD_TH | vehicle_th: $VEH_TH | " && cat) >> log_hyper_tune.txt
done 

#
# ROAD_TH 0.57 -> 0.50:0.64
# VEH_TH 0.20 -> 0.15:0.25
#

