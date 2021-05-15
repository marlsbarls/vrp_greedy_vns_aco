import numpy as np
from src.algorithm.improved_vns import create_planning_df
import src.config.vns_config as cfg
import src.config.preprocessing_config as config
import pandas as pd

from src.algorithm.SavingsAlgorithm import time_checker



class GreedyAlgorithm:
    def __init__(self, all_orders_df, travel_time_matrix, service_time_matrix, planning_df):
        self.all_orders_df = all_orders_df
        self.travel_time_matrix = travel_time_matrix
        self.service_time_matrix = service_time_matrix
        self.planning_df = planning_df
        self.demand = self.all_orders_df['DEMAND'].to_numpy()
        self.readytime = self.all_orders_df['READYTIME'].to_numpy()
        self.duetime = self.all_orders_df['DUETIME'].to_numpy()

    def closest_order(self, current_order):
        # should consider time and space 
        # maybe weigh distance more than time --> 2*distance + minutes
        shorest_travel_time = np.min(self.travel_time_matrix[current_order][np.nonzero(self.travel_time_matrix[current_order])])
        return list(self.travel_time_matrix[current_order]).index(shorest_travel_time)

    def remove_visited(self, current_order):
        for i in self.travel_time_matrix:
            i[current_order] = 0

    # def create_greedy_sequence(self, start):
    #     order_sequence = []
    #     order_sequence.append(start)
    #     current_order = start
    #     while len(order_sequence) < len(self.all_orders_df)+1:
    #         next = self.closest_order(self.travel_time_matrix, current_order)
    #         order_sequence.append(next)
    #         current_order = next
    #         self.remove_visited(self.travel_time_matrix, current_order)
    #     order_sequence.append(start)
    #     return order_sequence

    def return_free_tours(tour_capacity_reached):
        free_tours = [k for k,v in tour_capacity_reached.items() if v == False]
        return free_tours


    def create_greedy_sequence(self, start, current_order_df):
        Sub_tour = []
        Sub_tour.append(start)
        current_order = start
        visited = []
        tour = 0
        tour_capacity_reached = {0: False}
        while len(visited) < len(current_order_df):
            
            # next depends on which tour is currently being filled
            next = self.closest_order(self.travel_time_matrix, current_order)
            temp_tour = Sub_tour[tour] + [next]
            time_check = time_checker(temp_tour, current_order_df, self.travel_time_matrix, self.service_time_matrix)
            merge_demand = sum(self.demand[temp_tour])
            if (time_check and merge_demand < cfg.capacity):
                Sub_tour[tour].append(next)
            elif merge_demand > cfg.capacity:
                # ASSUMPTION: once we have one order that would go over capacity we declare tour as full
                # this makes sense because we will always choose best order next
                tour_capacity_reached[tour] = True
                free_tours = self.return_free_tours(tour_capacity_reached)
                if not free_tours: 
                    existing_tours = tour_capacity_reached.keys()
                    tour = max(existing_tours) + 1
                    Sub_tour[tour] = [next]
            

            visited.append(next)
            current_order = next
            self.remove_visited(self.travel_time_matrix, current_order)

        Sub_tour.append(start)
        return Sub_tour

    def run_greedy(self, time=0):
        current_order_df = self.all_orders_df[(
        self.all_orders_df.READYTIME <= time)]
        demand = current_order_df['DEMAND'].to_numpy()
        readytime = current_order_df['READYTIME'].to_numpy()
        duetime = current_order_df['DUETIME'].to_numpy()

        Sub_tour = []
        
        tour = self.create_greedy_sequence(0, current_order_df)

