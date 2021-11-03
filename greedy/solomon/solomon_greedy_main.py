from math import dist
from greedy.solomon.solomon_greedy_algorithm import GreedyAlgorithm
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
        dir_name = os.path.dirname(os.path.realpath('__file__'))
        file_names = self.test_files

        for file in file_names:
            file_name = os.path.join(
                        dir_name, 'input_data', self.source, 'orders', file + '_orders.csv')
            all_orders_df = pd.read_csv(file_name)
            all_orders_df['CUST_NO'] = all_orders_df['CUST_NO'].apply(lambda x: x - 1)
            xcoor = all_orders_df['XCOORD'].to_numpy()
            ycoor = all_orders_df['YCOORD'].to_numpy()

            cust_size = len(all_orders_df)-1
            dist_matrix = np.zeros((cust_size + 1, cust_size + 1))
            for i in range(cust_size + 1):
                for j in range(cust_size + 1):
                    dist_matrix[i][j] = np.sqrt(
                        (xcoor[i] - xcoor[j]) ** 2 + (ycoor[i] - ycoor[j]) ** 2)
            service_times = all_orders_df['SERVICETIME'].to_numpy()

                      
            print(f'Current file: {file}')
            print(f'Running greedy algorithm {self.test_type}')
            greedy = GreedyAlgorithm(self.test_type, self.source, all_orders_df, dist_matrix, service_times)

            result_df = greedy.run_greedy()

            result_file_path = dir_name + '/results/greedy/solomon/' + file + '_' + self.test_type  + '.csv'
            result_df.to_csv(result_file_path)

            