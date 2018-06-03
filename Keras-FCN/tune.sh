#!/bin/bash 
for i in `seq 0 0.1 1`;
do
    ROAD_TH=0.57
    ROAD_FADE=0

    VEH_TH=0.40
    VEH_FADE=$i
    echo "testing with ROAD_TH=$ROAD_TH, VEH_TH=$VEH_TH"
    echo "WY_echo" | grader "python inference_client.py $ROAD_TH $VEH_TH $ROAD_FADE $VEH_FADE" | grep "Car F score:" | (echo -n "road_th: $ROAD_TH | road_fade: $ROAD_FADE | veh_th: $VEH_TH | veh_fade: $VEH_FADE | " && cat) >> log_hyper_tune.txt
done 

#
# ROAD_TH 0.57 -> 0.50:0.64
# VEH_TH 0.20 -> 0.15:0.25
#

