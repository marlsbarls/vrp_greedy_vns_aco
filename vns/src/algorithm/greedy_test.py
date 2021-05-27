from src.algorithm.greedy_final import GreedyAlgorithm
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

    # planning_df = all_orders_df[['CUST_NO']].copy(deep=True)
    # planning_df['VISITED'] = False
    # planning_df['SCHEDULED_TIME'] = np.NaN
    # planning_df['SCHEDULED_TOUR'] = np.NaN

    travel_time_matrix_with_orderids = DistanceMatrix.load_file(os.path.join(
        dir_name, 'data', 'results_preprocessing', file + '_travel_times'))
    service_time_matrix = DistanceMatrix.load_file(os.path.join(
        dir_name, 'data', 'results_preprocessing', file + '_service_times'))

    travel_time_matrix = get_travel_time_matrix(len(all_orders_df)-1, all_orders_df['XCOORD'], all_orders_df['YCOORD'], all_orders_df['XCOORD_END'], all_orders_df['YCOORD_END'])


    # time = 0
    # current_order_df = all_orders_df[(
    #     all_orders_df.READYTIME <= time)]

    greedy = GreedyAlgorithm(all_orders_df, travel_time_matrix, travel_time_matrix_with_orderids, service_time_matrix)
    greedy.run_greedy()
    
    # order_sequence = create_order_sequence(travel_time_matrix, 0)


    # def create_initial_solution(self, depot_id, time=480):
    #     tour_capacity_reached, Sub_tour = self.create_new_sub_tour(depot_id)
    #     visited = []
    #     tour = 0
    #     current_order = depot_id
    #     possible_travel_times = self.return_possible_travel_times()
    #     # tour_tabu_list = {}
    #     while len(visited) < len(self.current_order_df)-1:
    #         next = self.closest_order(current_order, possible_travel_times, Sub_tour)
    #         if not next:
    #             tour_capacity_reached, tour, Sub_tour, current_order = self.update_full_tour(tour_capacity_reached, tour,
    #                                                                                          depot_id, Sub_tour, True)

    #         temp_tour = deepcopy(Sub_tour[tour])
    #         temp_tour.insert(len(temp_tour)-1, next)
    #         time_check = time_checker(temp_tour, self.travel_time_matrix, self.service_time_matrix, self.readytime, self.duetime, self.order_ids)
    #         # if not next in tour_tabu_list:
    #         #     tour_tabu_list[next] = []
            
    #         if time_check:
    #             if self.calculate_time_demand(temp_tour) >= 0:
    #                 Sub_tour[tour].insert(len(Sub_tour[tour])-1, next)
    #                 visited.append(next)
    #                 current_order = next
    #                 possible_travel_times = self.remove_visited(current_order, possible_travel_times)
    #             else:
    #                 possible_travel_times = self.remove_impossible(possible_travel_times, current_order, next)
    #                 tour_capacity_reached, tour, Sub_tour, current_order = self.update_full_tour(tour_capacity_reached, tour,
    #                                                                                          depot_id, Sub_tour, True)
 
    #         else:
    #             # tour_tabu_list[next].append(tour)
    #             # possible_travel_times = self.remove_impossible(possible_travel_times, current_order, next)
    #             # free_tours = self.return_free_tours(tour_capacity_reached)
    #             # if not free_tours: 
    #             #     existing_tours = tour_capacity_reached.keys()
    #             #     # create a new tour
    #             #     tour = max(existing_tours) + 1
    #             #     tour_capacity_reached, Sub_tour = self.create_new_sub_tour(depot_id, tour_capacity_reached, Sub_tour)
    #             #     current_order = depot_id
    #             # else:
    #             #     tour_updated = False
    #             #     for free_tour in free_tours:
    #             #         if free_tour not in tour_tabu_list[next]:
    #             #             tour = free_tour
    #             #             current_order = Sub_tour[tour][-2]
    #             #             tour_updated = True
    #             #             break
                    
    #             #     if not tour_updated:
    #             #         existing_tours = tour_capacity_reached.keys()
    #             #         # create a new tour
    #             #         tour = max(existing_tours) + 1
    #             #         tour_capacity_reached, Sub_tour = self.create_new_sub_tour(depot_id, tour_capacity_reached, Sub_tour)
    #             #         current_order = depot_id

    #             possible_travel_times = self.remove_impossible(possible_travel_times, current_order, next)
    #             tour_capacity_reached, tour, Sub_tour, current_order = self.update_full_tour(tour_capacity_reached, tour,
    #                                                                                          depot_id, Sub_tour, False)

                    
        
    #     planning_df = create_planning_df(Sub_tour, self.current_order_df, self.travel_time_with_orderids, self.service_time_matrix, self.readytime, self.duetime)

    #     return Sub_tour, tour_capacity_reached, visited, tour, planning_df, current_order



    