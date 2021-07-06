from greedy.surve_mobility.greedy_algorithm import GreedyAlgorithm
import sys
import os
import numpy as np
import pandas as pd
from vns.src.algorithm.SavingsAlgorithm import SavingsAlgorithm
from vns.src.algorithm.improved_vns import run_vns, create_planning_df, total_cost, get_travel_time_matrix
from vns.src.helpers.DistanceMatrix import DistanceMatrix
import vns.src.config.vns_config as cfg
from copy import deepcopy
from vns.src.visualizations.TourPlanVisualizer import TourPlanVisualizer
from vns.src.visualizations.TourVisualizer import TourVisualizer
from pathlib import Path
import shutil

#from scipy.spatial.distance import squareform, pdist

class Greedy:
    def __init__(self, test_files, test_type, source):
        # file names to be tested
        self.test_files = test_files
        # static or dynmic
        self.test_type = test_type
        # real data (surve mobility) or test data (solomon)
        self.source = source

    def run_greedy(self):
        # pd.set_option("display.max_colwidth", 10000)
        dir_name = os.path.dirname(os.path.realpath('__file__'))
        file_names = self.test_files
        # test_name = "dynamic_solution_retry"
        # test_type = "dynamic"  # static or dynamic
        # only relevant if testtype = static: validation or experimentation
        # static_type = "experimentation"

        for file in file_names:
            # test_name = test_name
            # file_name = os.path.join(
            #     dir_name, 'data', 'results_preprocessing', file + '_orders.csv')
            
            file_name = os.path.join(
                        dir_name, 'input_data', self.source, 'orders', file + '_orders.csv')

            # outputfile = open('output_chargery_%s.txt' % file, 'w')
            # outputfile.write(f'File: {file}\n')

            # Path(os.path.join(dir_name, 'results', "greedy", file, self.test_type,
            #                     )).mkdir(parents=True, exist_ok=True)
            # # Path(os.path.join(dir_name, 'vns', "experiments", "results", file,
            # #                     test_name, "tour_plans")).mkdir(parents=True, exist_ok=True)
            # # Path(os.path.join(dir_name, 'vns', "experiments", "results", file,
            # #                     test_name, "tour_visuals")).mkdir(parents=True, exist_ok=True)
            # result_df = pd.DataFrame(columns=['costs_per_driver', 'costs_per_hour',  'initial_cost', 'final_cost', 
            #                                   'idle_time', 'tour_length', 'vehicle_number', 'runtime', 'total_iterations', 
            #                                   'initial_tour', 'final_tour'])

            all_orders_df = pd.read_csv(file_name)

            travel_time_matrix_with_orderids = DistanceMatrix.load_file(os.path.join(
                dir_name, 'input_data', self.source, file + '_travel_times'))
            service_time_matrix = DistanceMatrix.load_file(os.path.join(
                dir_name, 'input_data', self.source, file + '_service_times'))

            travel_time_matrix = get_travel_time_matrix(len(all_orders_df)-1, all_orders_df['XCOORD'], all_orders_df['YCOORD'], all_orders_df['XCOORD_END'], all_orders_df['YCOORD_END'])

            print('')
            print(f'Current file: {file}')
            print('Running greedy algorithm')
            
            greedy = GreedyAlgorithm(self.test_type, self.source, all_orders_df, travel_time_matrix, travel_time_matrix_with_orderids, service_time_matrix)
            result_tuple = greedy.run_greedy()

            # result_file_path = os.path.join(dir_name, '/results/', file, '_', self.test_type, '.txt' )
            result_file_path = dir_name + '/results/greedy/' + file + '_' + self.test_type + '.txt' 
            f = open(result_file_path, 'w')
            separator = '\n'
            f.write(separator.join(result_tuple))
            f.close()
            


        