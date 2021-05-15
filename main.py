'''----------------------------------------------

This file is used as entry point for the whole VNS.
It can handle every functionality that is documented below.

Usage: 

1) Uncomment the section you would like to execute.
2) Ensure you are using the correct conda environment (environment.yml is provided to create conda environment, if none exists)
3) Execute >python main.py 

-------------------------------------------------'''

# General imports - should stay active all the time.
import os
from dotenv import load_dotenv
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR))

# Run preprocessing
# Attention: If you want to retrieve real time data, ensure to pass your API-Key through the environment variable HERE_API_KEY.
# # You can therefore use the preloaded dotenv-Package

'''
from src.preprocessing.DataPreparation import DataPreparation
from src.preprocessing.preprocessing import Preprocessing
file_names = ["2020-07-07", "2020-07-08", "2020-08-14", "2020-08-15"]
for file in file_names:
    dp = DataPreparation(BASE_DIR, file)
    dp.rundatapreparation()
    pp = Preprocessing(BASE_DIR, file)
    pp.runpreprocessing()
'''

# Run chargery VNS
# Testcases & experiments should be defined in /src/algorithm/simulation.py

# import src.algorithm.simulation
import src.algorithm.greedy_test


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
