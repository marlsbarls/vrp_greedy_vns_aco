import sys
import os
import numpy as np
import pandas as pd
from vns.src.algorithm.SavingsAlgorithm import SavingsAlgorithm
from vns.src.algorithm.improved_vns import run_vns, create_planning_df, total_cost
from vns.src.helpers.DistanceMatrix import DistanceMatrix
import vns.src.config.vns_config as cfg
from copy import deepcopy
from vns.src.visualizations.TourPlanVisualizer import TourPlanVisualizer
from vns.src.visualizations.TourVisualizer import TourVisualizer
from pathlib import Path
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 10000)
dir_name = os.path.dirname(os.path.realpath('__file__'))
file_names = ["2020-07-07"]
test_name = "simulated_annealing_smoothened_10_runs"


def smoothen_convergence(convergence_per_run):
    smoothened_convergence = []
    for current_iteration in range(len(convergence_per_run[0])):
        global_function_values_at_current_iteration = []
        for local_conv in convergence_per_run:
            global_function_values_at_current_iteration.append(
                local_conv[current_iteration])

        smoothened_convergence.append(
            np.mean(global_function_values_at_current_iteration))
    print(smoothened_convergence)
    return smoothened_convergence


for file in file_names:
    test_name = test_name
    file_name = os.path.join(
        dir_name, 'data', 'results_preprocessing', file + '_orders.csv')

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

    savings = SavingsAlgorithm(
        all_orders_df, travel_time_matrix, service_time_matrix)

    time = 480
    Path(os.path.join(dir_name, "experiments", "results",
                      file, test_name, "convergence")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dir_name, "experiments", "results", file,
                      test_name, "tour_plans")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dir_name, "experiments", "results", file,
                      test_name, "tour_visuals")).mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(columns=[
        'initial_temperature',  'initial_cost', 'final_cost', 'idle_time', 'tour_length', 'vehicle_number', 'runtime', 'total_iterations', 'last_improvement', 'operator_performance', 'initial_tour', 'final_tour'])
    current_order_df, planning_df, current_tour = savings.initialization(
        time)

    # Set params to test
    validation_runs = 10
    convergence_per_temp = {
        0: [],
        10: [],
        100: [],
        1000: []
    }
    exp_id = 0
    for temp, convergence in convergence_per_temp.items():
        for i in range(validation_runs):
            # return object: "best_solution", "best_cost", "idle_time", "planning_df",
            # "initial_cost", "initial_solution", "last_improvement", "total_iterations", "runtime", "operator_performance"
            final_tour, final_cost, final_idle_time, planning_df, initial_cost, initial_solution, final_tour_length, last_improvement, total_iterations, runtime, performance_counter, convergence = run_vns(file, current_tour, all_orders_df, np.zeros(
                len(current_tour), dtype=int), planning_df, 0, True, initial_temperature=temp, max_restarts=1, max_iterations_without_improvement=100, exp_id=exp_id, test_name=test_name)

            result_df.loc[len(result_df.index)] = [
                temp, initial_cost, final_cost, final_idle_time, final_tour_length, len(final_tour), runtime, total_iterations, last_improvement, performance_counter, initial_solution, final_tour]
            tp = TourPlanVisualizer(dir_name, file,
                                    travel_time_matrix, planning_df, final_tour, final_cost, final_idle_time, True, time=0, test_name=test_name, exp_id=exp_id)
            tp.create_tour_plan()
            # base_dir, tours, time_slice, file_name, is_exp, **exp_params
            tv = TourVisualizer(dir_name, final_tour, time,
                                file, True, test_name=test_name, exp_id=exp_id)
            tv.run()
            result_df.to_csv(
                "%s/experiments/results/%s/%s/result.csv" % (dir_name, file, test_name))
            convergence_per_temp[temp].append(convergence)
            exp_id += 1

            for t, c in convergence_per_temp.items():
                print(t, ":", len(c))

    # Plot convergence per temperature
    plt.clf()
    plt.figure(figsize=(20, 10))
    for temp, convergence in convergence_per_temp.items():
        plt.plot(smoothen_convergence(convergence),
                 label="Initial Temp: %s" % (temp))

    plt.title('Simulated annealing: Convergence per temperature')
    plt.ylabel('Costs')
    plt.xlabel('Iterations')
    plt.legend()

    plt.savefig("%s/experiments/results/%s/%s/convergencePerTemperature.png" %
                (dir_name, file, test_name))

    plt.close()
