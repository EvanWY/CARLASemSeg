import os
import numpy as np

def execute(road_th, veh_th):
    cmd = 'echo "WY_echo" | grader "python run.py {0} {1}"'.format(road_th, veh_th)
    print ('[running command]: ' + cmd)
    log = os.popen(cmd).read()
    for line in log.splitlines():
        if line.startswith("Car F score:"):
            with open("hyperparam_optimization.log", "a") as log_file:
                text = 'road_th: {0} | veh_th: {1} | '.format(road_th, veh_th) + line
                print ('[result]: ' + text)
                log_file.write(text)


for x in np.arange(0.0, 1.0, 0.1):
    execute(x, 0.5)

for x in np.arange(0.0, 1.0, 0.1):
    execute(0.5, x)
        