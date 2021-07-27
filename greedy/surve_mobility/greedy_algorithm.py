from vns.src.algorithm.improved_vns import create_planning_df, time_checker, total_cost, total_distance, cal_total_time
import vns.src.config.vns_config as cfg
from copy import deepcopy
import vns.src.config.preprocessing_config as prep_cfg
import time
import os
import numpy as np
import sys


class GreedyAlgorithm:

    def __init__(self, test_type, source, all_orders_df, travel_time_matrix, travel_time_with_orderids, service_time_matrix):
        # static or dynmic
        self.test_type = test_type
        # real data (surve mobility) or test data (solomon)
        self.source = source
        
        self.all_orders_df = all_orders_df
        self.current_order_df = None
        self.travel_time_matrix = travel_time_matrix
        self.travel_time_with_orderids = travel_time_with_orderids
        self.service_time_matrix = service_time_matrix
        self.demand = self.all_orders_df['DEMAND'].to_numpy()
        self.readytime = self.all_orders_df['READYTIME'].to_numpy()
        if self.test_type == 'dynamic':
            self.available_time = self.readytime
            self.all_orders_df['AVAILABLETIME'] = self.all_orders_df['READYTIME']
        elif self.test_type == 'static':
            self.available_time = np.asarray([0] * len(self.readytime))
            self.all_orders_df['AVAILABLETIME'] = 0

        self.duetime = self.all_orders_df['DUETIME'].to_numpy()
        self.order_ids = self.all_orders_df['order_id'].values.tolist()
        self.depot_id = 0

    def return_possible_travel_times(self, Sub_tour=[]):
        current_orders = self.current_order_df['CUST_NO']
        all_orders = self.all_orders_df['CUST_NO']
        end_nodes = []
        if Sub_tour:
            for tour in Sub_tour:
                end_nodes.append(tour[-2])

        possible_travel_times = []
        for i in all_orders:
            for j in all_orders:
                if i != j and (i in current_orders or i in end_nodes) and (j in current_orders) and j != 0:
                    possible_travel_times.append([i, j, self.travel_time_matrix[i][j]]) 

        return possible_travel_times

    def get_service_time(self, order_no):
        # Always rush hour because traffic phase doesnt make a big difference
        traffic_phase = "rush_hour"
        service_time = self.service_time_matrix[self.order_ids[order_no]+":" + traffic_phase]
        return service_time

    def closest_order(self, current_order, possible_travel_times, planning_df):
        shortest_travel_time = sys.maxsize
        closest_order = None
        if current_order == self.depot_id:
            for i in possible_travel_times:
                if i[0] == current_order:
                    readytime_next_order = self.all_orders_df['READYTIME'][i[1]]
                    idle_time = readytime_next_order
                    if i[2] + idle_time < shortest_travel_time:
                        closest_order = i[1]
                        shortest_travel_time = i[2]
        else:
            for i in possible_travel_times:
                if i[0] == current_order:
                    readytime_next_order = self.all_orders_df['READYTIME'][i[1]]
                    ready_time_tour = planning_df['SCHEDULED_TIME'][current_order] + self.get_service_time(current_order) + self.travel_time_matrix[current_order][i[1]]
                    if ready_time_tour > readytime_next_order:
                        idle_time = 0
                    else:
                        idle_time = readytime_next_order - ready_time_tour

                    if i[2] + idle_time < shortest_travel_time:
                        closest_order = i[1]
                        shortest_travel_time = i[2]
                   

        return closest_order
        

    def remove_visited(self, current_order, possible_travel_times):
        k = [i for i in possible_travel_times if i[1] == current_order]
        if (len(k) != 0):
            possible_travel_times = [e for e in possible_travel_times if e not in k]
        
        return possible_travel_times

    def remove_impossible(self, possible_travel_times, current, next):
        possible_travel_times = [x for x in possible_travel_times if not (x[0] == current and x[1] == next)]
        return possible_travel_times

    def return_free_tours(self, tour_capacity_reached):
        free_tours = [k for k,v in tour_capacity_reached.items() if v == False]
        return free_tours

    def create_new_sub_tour(self, depot_id, tour_capacity_reached={}, Sub_tour=[]):
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

    def update_full_tour(self, tour_capacity_reached, tour, depot_id, Sub_tour, capacity_reached):
        # ASSUMPTION: once we have one order that would go over capacity or there is no more neighbours we declare tour as full
        if capacity_reached:
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

        return tour_capacity_reached, tour, Sub_tour, current_order


    def insert_new(self, depot_id, Sub_tour=[], tour_capacity_reached={}, visited=[], tour=0, planning_df=None, current_order=0, time=0, inital_solution=False):
        if inital_solution:
            Sub_tour = []
            tour_capacity_reached={}
            visited=[]
            tour=0
            planning_df=None
            current_order=0

        if not Sub_tour and not tour_capacity_reached:
            tour_capacity_reached, Sub_tour = self.create_new_sub_tour(depot_id=depot_id, tour_capacity_reached=tour_capacity_reached,
                                                                       Sub_tour=Sub_tour)

        planned = []
        possible_travel_times = self.return_possible_travel_times(Sub_tour)
        while len(planned) < len(self.current_order_df)-1:
            next = self.closest_order(current_order, possible_travel_times, planning_df)
            if not next:
                tour_capacity_reached, tour, Sub_tour, current_order = self.update_full_tour(tour_capacity_reached, tour,
                                                                                             depot_id, Sub_tour, True)
                continue
            temp_tour = deepcopy(Sub_tour[tour])
            temp_tour.insert(len(temp_tour)-1, next)
            time_check = time_checker(temp_tour, self.travel_time_matrix, self.service_time_matrix, self.readytime, self.duetime, self.order_ids)

            if time_check:
                if self.calculate_time_demand(temp_tour) >= 0:
                    Sub_tour[tour].insert(len(Sub_tour[tour])-1, next)
                    visited.append(next)
                    planned.append(next)
                    current_order = next
                    possible_travel_times = self.remove_visited(current_order, possible_travel_times)
                else:
                    possible_travel_times = self.remove_impossible(possible_travel_times, current_order, next)
                    tour_capacity_reached, tour, Sub_tour, current_order = self.update_full_tour(tour_capacity_reached, tour,
                                                                                             depot_id, Sub_tour, True)
            else:
                possible_travel_times = self.remove_impossible(possible_travel_times, current_order, next)
                tour_capacity_reached, tour, Sub_tour, current_order = self.update_full_tour(tour_capacity_reached, tour,
                                                                                             depot_id, Sub_tour, False)
            
            
            planning_df = create_planning_df(Sub_tour, self.all_orders_df, self.travel_time_with_orderids, self.service_time_matrix, self.readytime, self.duetime)

        return Sub_tour, tour_capacity_reached, visited, tour, planning_df, current_order

    def run_greedy(self):
        time_start = time.time()
        timer = 0
        print(f'Time: {timer}')
        depot_id = 0
        self.current_order_df = self.all_orders_df[(
            self.all_orders_df.AVAILABLETIME <= timer)]
  
        Sub_tour, tour_capacity_reached, visited, tour_id, planning_df, last_order = self.insert_new(depot_id=0, Sub_tour=[], time=timer, inital_solution=True)
        print(f'Total Cost: {total_cost(Sub_tour, self.travel_time_matrix, self.service_time_matrix, self.readytime, self.order_ids)}')
        print (f'Tour: {Sub_tour}')
        
        if self.test_type == 'dynamic':
            for timer in range(60, 481, 60):
                print(f'Time: {timer}')
                self.current_order_df = self.all_orders_df[(
                    self.all_orders_df.AVAILABLETIME <= timer) & (~self.all_orders_df['CUST_NO'].isin(visited))]
                Sub_tour, tour_capacity_reached, visited, tour_id, planning_df, last_order = self.insert_new(depot_id, Sub_tour, tour_capacity_reached, visited, tour_id, planning_df, last_order, timer)
                print(Sub_tour)
                print(total_cost(Sub_tour, self.travel_time_matrix, self.service_time_matrix, self.readytime, self.order_ids))

        time_end = time.time()
        print('----------FINAL RESULT----------------')
        total_costs = total_cost(Sub_tour, self.travel_time_matrix, self.service_time_matrix, self.readytime, self.order_ids)
        travel_time = cal_total_time(Sub_tour, self.travel_time_matrix, self.service_time_matrix, self.readytime, self.order_ids)
        print(f'Total Cost: {total_cost(Sub_tour, self.travel_time_matrix, self.service_time_matrix, self.readytime, self.order_ids)}')
        print(f'Tour:\n {Sub_tour}')
        print(f'total run time: {time_end-time_start}')
        print('end')

        
        result_tuple = ('path: '+str(Sub_tour), 'distance: '+str(total_distance(Sub_tour, self.travel_time_matrix)), 
                                              'vehicle_num: '+str(len(Sub_tour)), 'costs: '+str(total_costs), 
                                              'travel time: '+str(travel_time), str([]))
        
        return result_tuple

        

