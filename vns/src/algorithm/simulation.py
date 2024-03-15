import datetime
import sys
import os
import numpy as np
import pandas as pd
from vns.src.algorithm.SavingsAlgorithm import SavingsAlgorithm
from vns.src.algorithm.improved_vns import create_planning_df_new, run_vns, create_planning_df_new, total_cost
from vns.src.helpers.DistanceMatrix import DistanceMatrix
import vns.src.config.vns_config as cfg
from copy import deepcopy
from vns.src.visualizations.TourPlanVisualizer import TourPlanVisualizer
from vns.src.visualizations.TourVisualizer import TourVisualizer
from pathlib import Path
from datetime import date, datetime
import time



class VNSSimulation():
    def __init__(self, test_files, dynamic):
        self.test_files = test_files
        self.dyanmic = dynamic
        self.savings_consider_time = True
        # stop_event = iterations or time or both
        # self.stop_event = 'iterations'
        self.stop_event = 'time'
        self.set_parameters = {
            'static or dynamic': self.dyanmic,
            'considering time in saving': self.savings_consider_time,
            'stop_event': self.stop_event,
            'cost per hour': cfg.cost_per_hour,
            'cost per driver': cfg.cost_per_driver,
            'shaking': cfg.shaking,
            'other parameters': cfg.vns
        }

    def run_simulation(self):
        pd.set_option("display.max_colwidth", 10000)
        dir_name = os.path.dirname(os.path.realpath('__file__'))
        file_names = self.test_files
        test_name = "dynamic_solution"
        test_type = self.dyanmic  # static or dynamic
        # only relevant if testtype = static: validation or experimentation
        static_type = "validation"

        for file in file_names:
            self.set_parameters['file'] = file
            test_name = test_name
            file_name = os.path.join(
                dir_name, 'input_data', 'surve_mobility', 'orders', file + '_orders.csv')
            
            # today = date.today()
            # date_today = today.strftime("%b-%d-%Y")
            now = datetime.now()
            date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
            target_folder_results = os.path.join(
            dir_name, "results", "vns", 'surve_mobility', file, test_type, date_time)
            self.set_parameters['Result Folder'] = target_folder_results
            

            all_orders_df = pd.read_csv(file_name)
            order_ids = all_orders_df['order_id'].values.tolist()
            capacity = cfg.capacity

            if test_type == 'static':
                all_orders_df['AVAILABLETIME'] = 0
            elif test_type == 'dynamic':
                all_orders_df['AVAILABLETIME'] = all_orders_df['READYTIME']

            #### experiment
            all_orders_df['READYTIME'] = 0

            ###### end

            planning_df = all_orders_df[['CUST_NO']].copy(deep=True)
            planning_df['VISITED'] = False
            planning_df['SCHEDULED_TIME'] = np.NaN
            planning_df['SCHEDULED_TOUR'] = np.NaN

            travel_time_matrix = DistanceMatrix.load_file(os.path.join(
                dir_name, 'input_data', 'surve_mobility', file + '_travel_times'))
            service_time_matrix = DistanceMatrix.load_file(os.path.join(
                dir_name,'input_data', 'surve_mobility', file + '_service_times'))

            time_slice = 0
            savings = SavingsAlgorithm(
                all_orders_df, travel_time_matrix, service_time_matrix, self.savings_consider_time)

            # elif(test_type == "dynamic"):
            Path(target_folder_results).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(target_folder_results, "convergence")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(target_folder_results, "tour_plans")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(target_folder_results, "tour_visuals")).mkdir(parents=True, exist_ok=True)


            result_df = pd.DataFrame(columns=[
                'parameters', 'interval', 'time', 'initial_cost', 'final_cost', 'num drivers', 'idle_time', 'tour_length', 'runtime', 'total_iterations', 'last_improvement', 'operator_performance', 'initial_tour', 'final_tour', 'run_time'])

            if test_type == 'static':
                interval = 480
                # interval = 15
            if test_type == 'dynamic':
                interval = 15
            for exp_id in range(0, 1):
                for time_slice in range(0, 481, interval):
                    print(f'results_folder {target_folder_results}')
                    exp_id_result = exp_id
                    time_start = time.time()
                    if(time_slice == 0):
                        current_order_df, planning_df, current_tour = savings.initialization(
                            time_slice)
                        order_visibility = np.zeros(len(current_tour), dtype=int)
                    else:
                        current_order_df, planning_df, current_tour, order_visibility = savings.insert_new(
                            final_tour, planning_df, time_slice, interval)
                    # return object: "best_solution", "best_cost", "idle_time", "planning_df",
                    # "initial_cost", "initial_solution", "last_improvement", "total_iterations", "runtime", "operator_performance"
                    final_tour, final_cost, final_idle_time, planning_df, initial_cost, initial_solution, final_tour_length, last_improvement, total_iterations, runtime, performance_counter, convergence = run_vns(
                        file, current_tour, all_orders_df, order_visibility, planning_df, interval, target_folder_results, test_type, self.stop_event, True, exp_id=str(interval) + "_" + str(time_slice), test_name=test_name)

                    current_time = time.time()
                    run_time = (current_time - time_start)/60
                    result_df.loc[len(result_df.index)] = [
                        self.set_parameters, interval, time_slice, initial_cost, final_cost, len(final_tour), final_idle_time, final_tour_length, runtime, total_iterations, last_improvement, performance_counter, initial_solution, final_tour, run_time]

                    # tp = TourPlanVisualizer(dir_name, file, target_folder_results,
                                            # travel_time_matrix, planning_df, final_tour, final_cost, final_idle_time, True, exp_id=exp_id_result, time=time_slice, test_name=test_name)
                    # tp.create_tour_plan()
                    # base_dir, tours, time_slice, file_name, is_exp, **exp_params
                    # tv = TourVisualizer(dir_name, target_folder_results, final_tour, time_slice,
                    #                     file, True, test_name=test_name, exp_id=exp_id_result)
                    # tv.run()

                    result_df.to_csv(os.path.join(target_folder_results, 'result.csv'))
                    # print('')

                    if test_type == 'static':
                        break

            # if (test_type == "static"):
            #     time_slice = 0
            #     Path(target_folder_results).mkdir(parents=True, exist_ok=True)
            #     Path(os.path.join(target_folder_results, "convergence")).mkdir(parents=True, exist_ok=True)
            #     Path(os.path.join(target_folder_results, "tour_plans")).mkdir(parents=True, exist_ok=True)
            #     Path(os.path.join(target_folder_results, "tour_visuals")).mkdir(parents=True, exist_ok=True)

            #     # result_df = pd.DataFrame(columns=[
            #     #     'costs_per_driver', 'costs_per_hour', 'savings_considering_time' 'initial_cost', 'final_cost', 'idle_time', 'tour_length', 'vehicle_number', 'runtime', 'total_iterations', 'last_improvement', 'operator_performance', 'initial_tour', 'final_tour'])
            #     current_order_df, planning_df, current_tour = savings.initialization()
                # if(static_type == "experimentation"):
                #     # Set params to test
                #     # fix_costs = [400, 0, 10, 100]  # cost per driver
                #     # variable_costs = [9.5, 15, 30]  # cost per hour
                #     fix_costs = [400, 0, 10, 100]  # cost per driver
                #     variable_costs = [9.5, 15, 30]  # cost per hour
                #     #insert_probabilities = [1, 0.97, 0.94]
                #     #sort_probabilities = [0, 0.5, 0.75, 1]
                #     #temperatures = [10, 100, 1000]

                #     exp_id = 0
                #     for cost_per_driver in fix_costs:
                #         for cost_per_hour in variable_costs:
                #             # return object: "best_solution", "best_cost", "idle_time", "planning_df",
                #             # "initial_cost", "initial_solution", "last_improvement", "total_iterations", "runtime", "operator_performance"
                #             final_tour, final_cost, final_idle_time, planning_df, initial_cost, initial_solution, final_tour_length, last_improvement, total_iterations, runtime, performance_counter, convergence = run_vns(file, current_tour, all_orders_df, np.zeros(
                #                 len(current_tour), dtype=int), planning_df, 0, target_folder_results, True, cost_per_driver=cost_per_driver, cost_per_hour=cost_per_hour, exp_id=exp_id, test_name=test_name)

                #             result_df.loc[len(result_df.index)] = [
                #                 cost_per_driver, cost_per_hour, initial_cost, final_cost, final_idle_time, final_tour_length, len(final_tour), runtime, total_iterations, last_improvement, performance_counter, initial_solution, final_tour]
                #             tp = TourPlanVisualizer(dir_name, file,
                #                                     travel_time_matrix, planning_df, final_tour, final_cost, final_idle_time, True, time=0, test_name=test_name, exp_id=exp_id)
                #             tp.create_tour_plan()
                #             # base_dir, tours, time_slice, file_name, is_exp, **exp_params
                #             tv = TourVisualizer(dir_name, final_tour, time,
                #                                 file, True, test_name=test_name, exp_id=exp_id)
                #             tv.run()
                #             result_df.to_csv(os.path.join(target_folder_results, 'result.csv'))
                #             # result_df.to_csv(
                #             #     "%s/results/vns/surve_mobility/%s/%s/result.csv" % (dir_name, file, test_name))
                #             exp_id += 1

                # elif(static_type == "validation"):

                #     result_df = pd.DataFrame(columns=[
                #         'parameters', 'interval', 'time', 'initial_cost', 'final_cost', 'num drivers', 'idle_time', 'tour_length', 'runtime', 'total_iterations', 'last_improvement', 'operator_performance', 'initial_tour', 'final_tour', 'run_time'])

                #     # intervals = [15, 30, 60]
                #     interval = 15
                #     for exp_id in range(0, 5):
                #         for time_slice in range(0, 481, interval):
                #             exp_id_result = exp_id
                #             time_start = time.time()
                #             if(time_slice == 0):
                #                 current_order_df, planning_df, current_tour = savings.initialization(
                #                     time_slice)
                #                 order_visibility = np.zeros(len(current_tour), dtype=int)
                #             else:
                #                 current_order_df, planning_df, current_tour, order_visibility = savings.insert_new(
                #                     final_tour, planning_df, time_slice, interval)
                #             # return object: "best_solution", "best_cost", "idle_time", "planning_df",
                #             # "initial_cost", "initial_solution", "last_improvement", "total_iterations", "runtime", "operator_performance"
                #             final_tour, final_cost, final_idle_time, planning_df, initial_cost, initial_solution, final_tour_length, last_improvement, total_iterations, runtime, performance_counter, convergence = run_vns(
                #                 file, current_tour, all_orders_df, order_visibility, planning_df, interval, target_folder_results, self.stop_event, True, exp_id=str(interval) + "_" + str(time_slice), test_name=test_name)

                #             current_time = time.time()
                #             run_time = (current_time - time_start)/60
                #             result_df.loc[len(result_df.index)] = [
                #                 self.set_parameters, interval, time_slice, initial_cost, final_cost, len(final_tour), final_idle_time, final_tour_length, runtime, total_iterations, last_improvement, performance_counter, initial_solution, final_tour, run_time]

                #             # tp = TourPlanVisualizer(dir_name, file, target_folder_results,
                #             #                         travel_time_matrix, planning_df, final_tour, final_cost, final_idle_time, True, exp_id=exp_id_result, time=time_slice, test_name=test_name)
                #             # tp.create_tour_plan()
                #             # # base_dir, tours, time_slice, file_name, is_exp, **exp_params
                #             # tv = TourVisualizer(dir_name, target_folder_results, final_tour, time_slice,
                #             #                     file, True, test_name=test_name, exp_id=exp_id_result)
                #             # tv.run()

                #             result_df.to_csv(os.path.join(target_folder_results, 'result.csv'))

                #             print('')
