import os
import numpy as np

for x in np.arange(0.0, 1.0, 0.1):
    road_th = x
    veh_th = 0.5
    log = os.popen('echo "WY_echo" | grader "python run.py {0} {1}"'.format(road_th, veh_th).read()
    for line in log.splitlines():
        if line.startswith("Car F score:"):
            with open("hyperparam_optimization.log", "a") as log_file:
                text = 'road_th: {0} | veh_th: {1} | '.format(road_th, veh_th) + line
                log_file.write(text)

for x in np.arange(0.0, 1.0, 0.1):
    road_th = 0.5
    veh_th = x
    log = os.popen('echo "WY_echo" | grader "python run.py {0} {1}"'.format(road_th, veh_th).read()
    for line in log.splitlines():
        if line.startswith("Car F score:"):
            with open("hyperparam_optimization.log", "a") as log_file:
                text = 'road_th: {0} | veh_th: {1} | '.format(road_th, veh_th) + line
                log_file.write(text)
        