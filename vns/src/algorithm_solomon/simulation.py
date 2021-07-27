import sys
import os
import numpy as np
import pandas as pd
from vns.src.algorithm_solomon.SavingsAlgorithm import SavingsAlgorithm
from vns.src.algorithm_solomon.improved_vns_solomon import run_vns, total_distance
from vns.src.preprocessing.DataExtension import DataExtension
from vns.src.helpers.DistanceMatrix import DistanceMatrix
import vns.src.algorithm_solomon.vns_config as cfg
from copy import deepcopy
import vns.src.config.preprocessing_config as prep_cfg
from pathlib import Path
import re

class VNSSimulationSolomon():
    def __init__(self, test_files, dynamic):
        self.test_files = test_files
        self.dynamic = dynamic

    def run_solomon_simulation(self):
        pd.set_option("display.max_colwidth", 10000)

        dir_name = os.path.dirname(os.path.realpath('__file__'))
        file_names = self.test_files
        #file_names = ["C101"]
        #file_names = ["C101_10_percent_dynamicity"]
        test_name = "static_sol_test"
        test_type = self.dynamic # static or dynamic # caution: dynamic not yet implemented
        validation_runs = 10

        for file in file_names:
            test_name = test_name

            if (test_type == "static"):
                exp_id = 0
                result_df = pd.DataFrame(columns=['instance',
                                                'initial_cost', 'final_cost', 'last_improvement', 'total_iterations', 'runtime', 'performance_counter', 'final_tour'])
                # Check whether folder for experiments exists, create otherwise
                for i in range(validation_runs):
                    Path(os.path.join(dir_name, "experiments", "solomon", file, test_name,
                                    "convergence")).mkdir(parents=True, exist_ok=True)

                    file_name = os.path.join(
                        dir_name, 'input_data', 'solomon', 'orders', file + '_orders' +  '.csv')

                    all_orders_df = pd.read_csv(file_name)

                    xcoor = all_orders_df['XCOORD'].to_numpy()
                    ycoor = all_orders_df['YCOORD'].to_numpy()
                    capacity = all_orders_df['CAPACITY'].iloc[0]
                    servicetime = all_orders_df['SERVICETIME'].to_numpy()
                    readytime = all_orders_df['READYTIME'].to_numpy()
                    duetime = all_orders_df['DUETIME'].to_numpy()
                    cust_size = len(all_orders_df) - 1

                    dist_matrix = np.zeros((cust_size + 1, cust_size + 1))
                    for i in range(cust_size + 1):
                        for j in range(cust_size + 1):
                            dist_matrix[i][j] = np.sqrt(
                                (xcoor[i] - xcoor[j]) ** 2 + (ycoor[i] - ycoor[j]) ** 2)

                    savings = SavingsAlgorithm(
                        all_orders_df, dist_matrix, capacity, servicetime, readytime, duetime, dynamic=test_type)

                    Ini_tour = savings.initialization(480)

                    final_tour, final_cost, last_improvement, initial_cost, total_iterations, runtime, performance_counter = run_vns(file, Ini_tour, all_orders_df, capacity,
                                                                                                                                    np.zeros(len(Ini_tour), dtype=int), True, exp_id=exp_id, test_name=test_name)
                    ['instance', 'initial_cost', 'final_cost', 'last_improvement',
                        'total_iterations', 'runtime', 'performance_counter', 'final_tour']
                    result_df.loc[len(result_df.index)] = [file, initial_cost, final_cost,
                                                        last_improvement, total_iterations, runtime, performance_counter, final_tour]

                    exp_id = exp_id+1

                    target_folder = os.path.join(
                        dir_name, "results", "vns", "solomon", file, test_name)
                    Path(target_folder).mkdir(parents=True, exist_ok=True)
                    result_df.to_csv(
                        "%s/results/vns/solomon/%s/%s/result.csv" % (dir_name, file, test_name))

            else:
                # Todo:
                # Not implemented

                # Data Extension
                # Input data standard test instance
                intervals_order_reception = 28
                total_intervals = 32
                days_in_analysis = 232
                average_orders_chargery = 66.93
                dynamicity = 0.1
                lam = (dynamicity*100)/intervals_order_reception

                #  Run Data Extension, only run test files with identical shift duration at a time

                print('-----DATA EXTENSION STARTED, FILE', file, '-----')
                de = DataExtension(dir_name, file, dynamicity,
                                lam, average_orders_chargery)
                capacity, file_name = de.rundataextension()

                """ file_name = os.path.join(
                    dir_name, 'data', 'solomon_dynamic', file + '.csv') """

                all_orders_df = pd.read_csv(file_name)

                xcoor = all_orders_df['XCOORD'].to_numpy()
                ycoor = all_orders_df['YCOORD'].to_numpy()
                readytime = all_orders_df['AVAILABLETIME'].to_numpy()
                duetime = all_orders_df['DUETIME'].to_numpy()
                servicetime = all_orders_df['SERVICETIME'].to_numpy()
                # TODO: Change capacity, get it from data extension
                """ capacity = 200 """
                all_orders_df['CUST_NO'] = all_orders_df['CUST_NO'].astype('int32')
                # cust_size = len(all_orders_df)-1
                cust_size = len(all_orders_df)-1

                dist_matrix = np.zeros((cust_size + 1, cust_size + 1))
                for i in range(cust_size + 1):
                    for j in range(cust_size + 1):
                        dist_matrix[i][j] = np.sqrt(
                            (xcoor[i] - xcoor[j]) ** 2 + (ycoor[i] - ycoor[j]) ** 2)

                savings = SavingsAlgorithm(
                    all_orders_df, dist_matrix, capacity, servicetime, readytime, duetime, True)

                # Current_tour = savings.initialization()
                # print(Current_tour)
                # Current_tour = savings.insert_new(
                #     Current_tour, 30, interval_length)
                # print(Current_tour)

                interval_length = capacity / total_intervals

                for time in range(0, capacity + 1, interval_length):
                    if(time == 0):
                        Current_tour, planning_df = savings.initialization()
                    else:
                        Current_tour, planning_df = savings.insert_new(
                            Current_tour, time, interval_length)

                    final_tour, final_cost, last_improvement = run_vns(file, Current_tour, all_orders_df, capacity,
                                                                    np.zeros(len(Current_tour), dtype=int), True, exp_id=exp_id)

                    result_df.loc[len(result_df.index)] = [file, total_distance(
                        Current_tour, dist_matrix), final_tour, final_cost, last_improvement]

                    exp_id = exp_id + 1

                    result_df.to_csv(
                        "%s/results/vns/solomon/%s/%s/result.csv" % (dir_name, file, test_name))
