import sys
import os
import numpy as np
import pandas as pd
from src.algorithm.SavingsAlgorithm import SavingsAlgorithm
from src.algorithm.improved_vns import run_vns, create_planning_df, total_cost, get_travel_time_matrix
from src.helpers.DistanceMatrix import DistanceMatrix
import src.config.vns_config as cfg
from copy import deepcopy
from src.visualizations.TourPlanVisualizer import TourPlanVisualizer
from src.visualizations.TourVisualizer import TourVisualizer
from pathlib import Path
from scipy.spatial.distance import squareform, pdist

# pd.set_option("display.max_colwidth", 10000)
dir_name = os.path.dirname(os.path.realpath('__file__'))
file_names = ["2020-07-07"]
# test_name = "dynamic_solution_retry"
# test_type = "dynamic"  # static or dynamic
# only relevant if testtype = static: validation or experimentation
# static_type = "experimentation"

for file in file_names:
    # test_name = test_name
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

    # travel_time_matrix = DistanceMatrix.load_file(os.path.join(
    #     dir_name, 'data', 'results_preprocessing', file + '_travel_times'))
    service_time_matrix = DistanceMatrix.load_file(os.path.join(
        dir_name, 'data', 'results_preprocessing', file + '_service_times'))

    travel_time_matrix = get_travel_time_matrix(len(all_orders_df)-1, all_orders_df['XCOORD'], all_orders_df['YCOORD'], all_orders_df['XCOORD_END'], all_orders_df['YCOORD_END'])


    time = 0


    
    order_sequence = create_order_sequence(travel_time_matrix, 0)




    time = 0