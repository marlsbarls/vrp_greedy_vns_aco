
'''----------------------------------------------

This file is used as entry point for all algorithm (ACO, VNS and Greedy).
It can handle every functionality that is documented below.

Usage: 

1) Uncomment the section you would like to execute.
2) Ensure you are using the correct conda environment (environment.yml is provided to create conda environment, if none exists)
3) Execute >python main.py 

-------------------------------------------------'''
# General imports - should stay active all the time.
from vns.src.algorithm.simulation import VNSSimulation
from vns.src.algorithm_solomon.simulation import VNSSimulationSolomon
from aco.execution_new import Execution
from greedy.surve_mobility.greedy_main import Greedy as Surve_Greedy
from greedy.solomon.solomon_greedy_main import Greedy as Solomon_Greedy
import sys
import os
from dotenv import load_dotenv
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR))

# Run preprocessing
# Attention: If you want to retrieve real time data, ensure to pass your API-Key through the environment variable HERE_API_KEY.
# # You can therefore use the preloaded dotenv-Package


# from vns.src.preprocessing.DataPreparation import DataPreparation
# from vns.src.preprocessing.preprocessing import Preprocessing


# test type: static or dynamic
test_type = 'dynamic'
# test_type = 'static'
# source: solomon or surve_mobility
source = 'surve_mobility'
# source = 'solomon' 

if source != 'surve_mobility' and source != 'solomon':
    sys.exit('Set variable source to "surve_mobility" or "solomon')
if test_type != 'dynamic' and test_type != 'static':
    sys.exit('Set variable source to "dynamic" or "static')



if source == 'solomon':
    # test_files = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
    # test_files = ['C201']
    test_files = ['C201']
    # test_files = ['R101', 'C101']

    ## run greedy
    greedy = Solomon_Greedy(test_files, test_type, source)
    greedy.run_greedy()

    # # run aco
    # if __name__ == '__main__':
    #     aco = Execution(test_files, test_type, source)
    #     aco.run_macs()

    # ## run vns
    # vns = VNSSimulationSolomon(test_files, test_type)
    # vns.run_solomon_simulation()



## run experiments


########################################################
# Contents from VNS main
########################################################


# Run solomon VNS
# Testcases & experiments should be defined in /src/algorithm_solomon/simulation.py

# import src.algorithm_solomon.simulation


# ---Specific experiments---
# Experiment to evaluate the relevance of availibility of charging stations
'''
import experiments.charging_avail_exp
'''

# Experiment to evaluate the relevance of simulated annealing
'''
import experiments.simulated_annealing_exp

'''






