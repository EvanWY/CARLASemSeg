#!/bin/bash 
for i in `seq 0 0.1 1`;
do
    ROAD_TH=0.57
    ROAD_FADE=0

    VEH_TH=0.30
    VEH_FADE=$i
    echo "testing with ROAD_TH=$ROAD_TH, VEH_TH=$VEH_TH"
    echo "WY_echo" | grader "python inference_client.py $ROAD_TH $VEH_TH $ROAD_FADE $VEH_FADE" | grep "Car F score:" | (echo -n "road_th: $ROAD_TH | road_fade: $ROAD_FADE | veh_th: $VEH_TH | veh_fade: $VEH_FADE | " && cat) >> log_hyper_tune.txt
done 

#
# ROAD_TH 0.57 -> 0.50:0.64
# VEH_TH 0.20 -> 0.15:0.25
#

#road_th: 0.57 | road_fade: 0 | veh_th: 0.40 | veh_fade: 0.2 | Car F score: 0.780 | Car Precision: 0.675 | Car Recall: 0.812 | Road F score: 0.976 | Road Precision: 0.987 | Road Recall: 0.933 | Averaged F score: 0.878
#road_th: 0.57 | road_fade: 0 | veh_th: 0.40 | veh_fade: 0.3 | Car F score: 0.780 | Car Precision: 0.626 | Car Recall: 0.831 | Road F score: 0.976 | Road Precision: 0.987 | Road Recall: 0.933 | Averaged F score: 0.878
# veh_th 0.4, veh_fade 0.2-0.3, car_f 0.78


