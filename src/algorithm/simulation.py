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

pd.set_option("display.max_colwidth", 10000)
dir_name = os.path.dirname(os.path.realpath('__file__'))
file_names = ["2020-07-07"]
test_name = "dynamic_solution_retry"
test_type = "dynamic"  # static or dynamic
# only relevant if testtype = static: validation or experimentation
static_type = "experimentation"

for file in file_names:
    test_name = test_name
    file_name = os.path.join(
        dir_name, 'data', 'results_preprocessing', file + '_orders.csv')

    outputfile = open('output_chargery_%s.txt' % file, 'w')
    outputfile.write(f'File: {file}\n')

    all_orders_df = pd.read_csv(file_name)
    order_ids = all_orders_df['order_id'].values.tolist()
    capacity = cfg.capacity

    planning_df = all_orders_df[['CUST_NO']].copy(deep=True)
    planning_df['VISITED'] = False
    planning_df['SCHEDULED_TIME'] = np.NaN
    planning_df['SCHEDULED_TOUR'] = np.NaN

    travel_time_matrix = DistanceMatrix.load_file(os.path.join(
        dir_name, 'data', 'results_preprocessing', file + '_travel_times'))
    service_time_matrix = DistanceMatrix.load_file(os.path.join(
        dir_name, 'data', 'results_preprocessing', file + '_service_times'))

    time = 0
    savings = SavingsAlgorithm(
        all_orders_df, travel_time_matrix, service_time_matrix)

    if (test_type == "static"):
        time = 480
        Path(os.path.join(dir_name, "experiments", "results",
                          file, test_name, "convergence")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dir_name, "experiments", "results", file,
                          test_name, "tour_plans")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dir_name, "experiments", "results", file,
                          test_name, "tour_visuals")).mkdir(parents=True, exist_ok=True)
        result_df = pd.DataFrame(columns=[
            'costs_per_driver', 'costs_per_hour',  'initial_cost', 'final_cost', 'idle_time', 'tour_length', 'vehicle_number', 'runtime', 'total_iterations', 'last_improvement', 'operator_performance', 'initial_tour', 'final_tour'])
        current_order_df, planning_df, current_tour = savings.initialization(
            time)
        if(static_type == "experimentation"):
            # Set params to test
            fix_costs = [400, 0, 10, 100]  # cost per driver
            variable_costs = [9.5, 15, 30]  # cost per hour
            #insert_probabilities = [1, 0.97, 0.94]
            #sort_probabilities = [0, 0.5, 0.75, 1]
            #temperatures = [10, 100, 1000]

            exp_id = 0
            for cost_per_driver in fix_costs:
                for cost_per_hour in variable_costs:
                    # return object: "best_solution", "best_cost", "idle_time", "planning_df",
                    # "initial_cost", "initial_solution", "last_improvement", "total_iterations", "runtime", "operator_performance"
                    final_tour, final_cost, final_idle_time, planning_df, initial_cost, initial_solution, final_tour_length, last_improvement, total_iterations, runtime, performance_counter, convergence = run_vns(file, current_tour, all_orders_df, np.zeros(
                        len(current_tour), dtype=int), planning_df, 0, True, cost_per_driver=cost_per_driver, cost_per_hour=cost_per_hour, exp_id=exp_id, test_name=test_name)

                    result_df.loc[len(result_df.index)] = [
                        cost_per_driver, cost_per_hour, initial_cost, final_cost, final_idle_time, final_tour_length, len(final_tour), runtime, total_iterations, last_improvement, performance_counter, initial_solution, final_tour]
                    tp = TourPlanVisualizer(dir_name, file,
                                            travel_time_matrix, planning_df, final_tour, final_cost, final_idle_time, True, time=0, test_name=test_name, exp_id=exp_id)
                    tp.create_tour_plan()
                    # base_dir, tours, time_slice, file_name, is_exp, **exp_params
                    tv = TourVisualizer(dir_name, final_tour, time,
                                        file, True, test_name=test_name, exp_id=exp_id)
                    tv.run()
                    result_df.to_csv(
                        "%s/experiments/results/%s/%s/result.csv" % (dir_name, file, test_name))
                    exp_id += 1

        elif(static_type == "validation"):

            for exp_id in range(0, 10):
                # return object: "best_solution", "best_cost", "idle_time", "planning_df",
                # "initial_cost", "initial_solution", "last_improvement", "total_iterations", "runtime", "operator_performance"
                final_tour, final_cost, final_idle_time, planning_df, initial_cost, initial_solution, final_tour_length, last_improvement, total_iterations, runtime, performance_counter, convergence = run_vns(file, current_tour, all_orders_df, np.zeros(
                    len(current_tour), dtype=int), planning_df, 0, True, exp_id=exp_id, test_name=test_name)

                # planning_df.to_csv(os.path.join(dir_name, 'planning_df.csv'))
                # current_order_df.to_csv(os.path.join(
                #     dir_name, 'current_orders_df.csv'))

                # result_df.loc[len(result_df.index)] = [
                #     cfg.shaking['INSERT']['PROBABILITY'], cfg.shaking['SORT_LEN']['PROBABILITY'], final_tour, final_cost, last_improvement, performance_counter]

                result_df.loc[len(result_df.index)] = [
                    cfg.shaking['INSERT']['PROBABILITY'], cfg.shaking['SORT_LEN']['PROBABILITY'], initial_cost, final_cost, final_idle_time, final_tour_length, len(final_tour), runtime, total_iterations, last_improvement, performance_counter, initial_solution, final_tour]

                tp = TourPlanVisualizer(dir_name, file,
                                        travel_time_matrix, planning_df, final_tour, final_cost, final_idle_time, True, time=0, test_name=test_name, exp_id=exp_id)
                tp.create_tour_plan()
                # base_dir, tours, time_slice, file_name, is_exp, **exp_params
                tv = TourVisualizer(dir_name, final_tour, time,
                                    file, True, test_name=test_name, exp_id=exp_id)
                tv.run()

                result_df.to_csv(
                    "%s/experiments/results/%s/%s/result.csv" % (dir_name, file, test_name))

    elif(test_type == "dynamic"):
        Path(os.path.join(dir_name, "experiments", "results", file,
                          test_name, "convergence")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dir_name, "experiments", "results", file,
                          test_name, "tour_plans")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dir_name, "experiments", "results", file,
                          test_name, "tour_visuals")).mkdir(parents=True, exist_ok=True)
        result_df = pd.DataFrame(columns=[
            'interval', 'time',  'initial_cost', 'final_cost', 'idle_time', 'tour_length', 'runtime', 'total_iterations', 'last_improvement', 'operator_performance', 'initial_tour', 'final_tour'])
        intervals = [15, 30, 60]
        for interval in intervals:
            for time in range(0, 481, interval):
                outputfile.write(f'Time: {time} \n')
                if(time == 0):
                    current_order_df, planning_df, current_tour = savings.initialization(
                        time)
                    order_visibility = np.zeros(len(current_tour), dtype=int)
                else:
                    current_order_df, planning_df, current_tour, order_visibility = savings.insert_new(
                        final_tour, planning_df, time, interval)
                # return object: "best_solution", "best_cost", "idle_time", "planning_df",
                # "initial_cost", "initial_solution", "last_improvement", "total_iterations", "runtime", "operator_performance"
                final_tour, final_cost, final_idle_time, planning_df, initial_cost, initial_solution, final_tour_length, last_improvement, total_iterations, runtime, performance_counter, convergence = run_vns(
                    file, current_tour, all_orders_df, order_visibility, planning_df, interval, True, exp_id=str(interval) + "_" + str(time), test_name=test_name)

                result_df.loc[len(result_df.index)] = [
                    interval, time, initial_cost, final_cost, final_idle_time, final_tour_length, runtime, total_iterations, last_improvement, performance_counter, initial_solution, final_tour]

                tp = TourPlanVisualizer(dir_name, file,
                                        travel_time_matrix, planning_df, final_tour, final_cost, final_idle_time, True, exp_id=interval, time=time, test_name=test_name)
                tp.create_tour_plan()
                # base_dir, tours, time_slice, file_name, is_exp, **exp_params
                tv = TourVisualizer(dir_name, final_tour, time,
                                    file, True, test_name=test_name, exp_id=interval)
                tv.run()

                result_df.to_csv("%s/experiments/results/%s/%s/result.csv" %
                                 (dir_name, file, test_name))

    outputfile.close()
