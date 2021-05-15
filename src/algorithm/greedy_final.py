import numpy as np
from src.algorithm.improved_vns import create_planning_df, time_checker
import src.config.vns_config as cfg
import src.config.preprocessing_config as config
import pandas as pd
from copy import deepcopy
import src.config.preprocessing_config as prep_cfg




class GreedyAlgorithm:
    def __init__(self, all_orders_df, travel_time_matrix, service_time_matrix, planning_df):
        self.all_orders_df = all_orders_df
        self.travel_time_matrix = travel_time_matrix
        self.service_time_matrix = service_time_matrix
        self.planning_df = planning_df
        self.demand = self.all_orders_df['DEMAND'].to_numpy()
        self.readytime = self.all_orders_df['READYTIME'].to_numpy()
        self.duetime = self.all_orders_df['DUETIME'].to_numpy()
        self.order_ids = self.all_orders_df['order_id'].values.tolist()

    def return_possible_travel_time(self):
        # TODO
        # Wie Savingslist traveltime zwischen allen möglichen Punkten berechnen?
        return

    def closest_order(self, current_order):
        # TODO
        # should consider time and space 
        # maybe weigh distance more than time --> 2*distance + minutes
        # consider readytime 
        shorest_travel_time = np.min(self.travel_time_matrix[current_order][np.nonzero(self.travel_time_matrix[current_order])][1:])



        return list(self.travel_time_matrix[current_order]).index(shorest_travel_time)

    def remove_visited(self, current_order):
        for i in self.travel_time_matrix:
            i[current_order] = 0


    def return_free_tours(self, tour_capacity_reached):
        free_tours = [k for k,v in tour_capacity_reached.items() if v == False]
        return free_tours

    def create_new_sub_tour(self, depot_id, tour_capacity_reached = {}, Sub_tour = []):
        Sub_tour.append([depot_id, depot_id])
        tour_capacity_reached[len(Sub_tour)-1] = False
        return tour_capacity_reached, Sub_tour

    def calculate_time_demand(self, tour):
        time = 0
        time_capacity = cfg.capacity
        for i in range(1, len(tour)):
            traffic_phase = "off_peak" if time < prep_cfg.traffic_times["phase_transition"][
                "from_shift_start"] else "phase_transition" if time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"

            if(self.order_ids[tour[i-1]] != 'order_0'):
                if(self.readytime[tour[i-1]] <= (time + self.service_time_matrix[self.order_ids[tour[i-1]]+":"+traffic_phase])):
                    time = time + \
                        self.service_time_matrix[self.order_ids[tour[i-1]]+":"+traffic_phase] + \
                        self.travel_time_matrix[tour[i-1]][tour[i]]

                    time_capacity -= self.service_time_matrix[self.order_ids[tour[i-1]]+":"+traffic_phase] + \
                                     self.travel_time_matrix[tour[i-1]][tour[i]]
                else:
                    time = self.readytime[tour[i-1]] + \
                        self.travel_time_matrix[tour[i-1]][tour[i]]

                    idle_time = self.readytime[tour[i-1]] - time

                    time_capacity -= idle_time + self.travel_time_matrix[tour[i-1]][tour[i]]
            else:
                if(self.readytime[tour[i-1]] <= time):
                    time = time + \
                        self.travel_time_matrix[tour[i-1]][tour[i]]

                    time_capacity -= self.travel_time_matrix[tour[i-1]][tour[i]]

        return time_capacity 

    def create_initial_solution(self, depot_id, time=0):
        tour_capacity_reached, Sub_tour = self.create_new_sub_tour(depot_id)
        visited = []
        tour = 0
        current_order = depot_id
        # tour_capacity_reached = {0: False}
        current_order_df = self.all_orders_df[(
        self.all_orders_df.READYTIME <= time)]
        while len(visited) < len(current_order_df)-1:
            # next depends on which tour is currently being filled
            next = self.closest_order(current_order)
            temp_tour = deepcopy(Sub_tour[tour])
            temp_tour.insert(len(temp_tour)-1, next)
            time_check = time_checker(temp_tour, self.travel_time_matrix, self.service_time_matrix, self.readytime, self.duetime, self.order_ids)
            # think i dont need demand bc no orders have duetime after 480
            # merge_demand = sum(self.demand[temp_tour])

            if time_check:
                if self.calculate_time_demand(temp_tour) >= 0:
                    Sub_tour[tour].insert(len(Sub_tour[tour])-1, next)
                    visited.append(next)
                    current_order = next
                    self.remove_visited(current_order)
                else:
                    # ASSUMPTION: once we have one order that would go over capacity we declare tour as full
                    # this makes sense because we will always choose best order next
                    tour_capacity_reached[tour] = True 
                    free_tours = self.return_free_tours(tour_capacity_reached)
                    if not free_tours: 
                        existing_tours = tour_capacity_reached.keys()
                        # create a new tour
                        tour = max(existing_tours) + 1
                        tour_capacity_reached, Sub_tour = self.create_new_sub_tour(depot_id, tour_capacity_reached, Sub_tour)
                        current_order = depot_id
                    else:
                        tour = min(free_tours)
                        current_order = Sub_tour[tour][-2]
            else:
                free_tours = self.return_free_tours(tour_capacity_reached)
                if not free_tours: 
                    existing_tours = tour_capacity_reached.keys()
                    # create a new tour
                    tour = max(existing_tours) + 1
                    tour_capacity_reached, Sub_tour = self.create_new_sub_tour(depot_id, tour_capacity_reached, Sub_tour)
                    current_order = depot_id
                else:
                    tour = min(free_tours)
                    current_order = Sub_tour[tour][-2]

        return Sub_tour
                        
            
                
            
                
                
            



        Sub_tour.append(depot_id)
        return Sub_tour

    def insert_new(self):
        return 

    def run_greedy(self, time=0):

        # demand = current_order_df['DEMAND'].to_numpy()
        # readytime = current_order_df['READYTIME'].to_numpy()
        # duetime = current_order_df['DUETIME'].to_numpy()

        Sub_tour = []
        
        tour = self.create_initial_solution(0)

