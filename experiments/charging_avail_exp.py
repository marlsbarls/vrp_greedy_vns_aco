'''
Contains experiment for varying availibility of charging stations
'''

import sys
import os
import numpy as np
import pandas as pd
from src.algorithm.SavingsAlgorithm import SavingsAlgorithm
from src.algorithm.improved_vns import run_vns, create_planning_df, total_cost
from src.helpers.DistanceMatrix import DistanceMatrix
import src.config.vns_config as cfg
from copy import deepcopy
from src.visualizations.TourPlanVisualizer import TourPlanVisualizer
from src.visualizations.TourVisualizer import TourVisualizer
from pathlib import Path
from src.preprocessing.preprocessing import Preprocessing

# Parameters

test_name = "charge_avail_exp_retry"
pd.set_option("display.max_colwidth", 10000)
dir_name = os.path.dirname(os.path.realpath('__file__'))
file_names = ["2020-07-07", "2020-08-15"]
target_folder = os.path.join(
    dir_name, "data", "results_preprocessing_charging_avail_exp")
configs = [[0.5, 1, 1, 1, 1], [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 1], [0.5, 0.75, 0.95, 1, 1]]
perform_preprocessing = True
validation_runs = 3


def build_config_string(config):
    config_string = ""
    for idx, prob in enumerate(config):
        config_string += "_p{i}_{probability}".format(
            i=str(idx+1), probability=str(prob*100))
    return config_string


# config_strings = [build_config_string(x) for x in configs]
# print(config_strings)


# Create Files for Experimentation (only executed if specified above):
if(perform_preprocessing):
    for file in file_names:
        for config in configs:
            pp = Preprocessing(dir_name, file, target_folder=target_folder, charge_avail_config=config,
                               charge_avail_config_string=build_config_string(config))
            pp.runpreprocessing()

# Run tests:
exp_id = 0
for file in file_names:
    result_df = pd.DataFrame(columns=[
        'config', 'initial_cost', 'final_cost', 'idle_time', 'tour_length', 'runtime', 'total_iterations', 'last_improvement', 'operator_performance', 'initial_tour', 'final_tour'])
    for config in configs:
        for val_id in range(validation_runs):
            config_string = build_config_string(config)
            order_file_path = os.path.join(
                target_folder, file + config_string + '_orders.csv')

            all_orders_df = pd.read_csv(order_file_path)
            order_ids = all_orders_df['order_id'].values.tolist()
            capacity = cfg.capacity

            planning_df = all_orders_df[['CUST_NO']].copy(deep=True)
            planning_df['VISITED'] = False
            planning_df['SCHEDULED_TIME'] = np.NaN
            planning_df['SCHEDULED_TOUR'] = np.NaN

            travel_time_matrix = DistanceMatrix.load_file(os.path.join(
                target_folder, file + config_string + '_travel_times'))
            service_time_matrix = DistanceMatrix.load_file(os.path.join(
                target_folder, file + config_string + '_service_times'))

            savings = SavingsAlgorithm(
                all_orders_df, travel_time_matrix, service_time_matrix)

            time = 480

            Path(os.path.join(dir_name, "experiments", "results",
                              file, test_name, "convergence")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(dir_name, "experiments", "results", file,
                              test_name, "tour_plans")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(dir_name, "experiments", "results", file,
                              test_name, "tour_visuals")).mkdir(parents=True, exist_ok=True)

            current_order_df, planning_df, current_tour = savings.initialization(
                time)

            final_tour, final_cost, final_idle_time, planning_df, initial_cost, initial_solution, final_tour_length, last_improvement, total_iterations, runtime, performance_counter, convergence = run_vns(file, current_tour, all_orders_df, np.zeros(
                len(current_tour), dtype=int), planning_df, 0, True, exp_id=exp_id, test_name=test_name)

            result_df.loc[len(result_df.index)] = [config_string, initial_cost, final_cost, final_idle_time, final_tour_length,
                                                   runtime, total_iterations, last_improvement, performance_counter, initial_solution, final_tour]
            result_df.to_csv(
                "%s/experiments/results/%s/%s/result.csv" % (dir_name, file, test_name))
            exp_id += 1
