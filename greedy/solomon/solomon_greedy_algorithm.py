from vns.src.algorithm_solomon.improved_vns_solomon import create_planning_df, time_checker, total_distance
import vns.src.config.vns_config as cfg
from copy import deepcopy
import vns.src.config.preprocessing_config as prep_cfg
import time
import numpy as np


class GreedyAlgorithm:

    def __init__(self, test_type, source, all_orders_df, dist_matrix, service_times):
        # static or dynmic
        self.test_type = test_type
        # real data (surve mobility) or test data (solomon)
        self.source = source
        
        self.all_orders_df = all_orders_df
        self.current_order_df = None
        self.dist_matrix = dist_matrix
        self.service_times = service_times
        self.demand = self.all_orders_df['DEMAND'].to_numpy()
        self.readytime = self.all_orders_df['READYTIME'].to_numpy()
        if self.test_type == 'dynamic':
            self.available_time = self.readytime
            self.all_orders_df['AVAILABLETIME'] = self.all_orders_df['READYTIME']
        elif self.test_type == 'static':
            self.available_time = np.asarray([0] * len(self.readytime))
            self.all_orders_df['AVAILABLETIME'] = 0
        self.duetime = self.all_orders_df['DUETIME'].to_numpy()
        self.order_ids = self.all_orders_df['CUST_NO'].values.tolist()
        self.depot_id = 0

        self.capacity = self.all_orders_df['CAPACITY'][0]

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
                    possible_travel_times.append([i, j, self.dist_matrix[i][j]]) 

        return possible_travel_times

    # def get_service_time(self, order_no):
    #     # Always rush hour because traffic phase doesnt make a big difference
    #     traffic_phase = "rush_hour"
    #     service_time = self.service_time_matrix[self.order_ids[order_no]+":" + traffic_phase]
    #     return service_time

    def closest_order(self, current_order, possible_travel_times, planning_df):
        shortest_travel_time = 10000000000
        closest_order = None
        if current_order == self.depot_id:
            for i in possible_travel_times:
                if i[0] == current_order and i[2] < shortest_travel_time:
                    closest_order = i[1]
                    shortest_travel_time = i[2]
        else:
            for i in possible_travel_times:
                if i[0] == current_order:
                    readytime_next_order = self.all_orders_df['READYTIME'][i[1]]
                    ready_time_tour = planning_df['SCHEDULED_TIME'][current_order] + self.service_times[current_order] + self.dist_matrix[current_order][i[1]]
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

    def create_new_sub_tour(self, depot_id, tour_capacity_reached = {}, Sub_tour = []):
        Sub_tour.append([depot_id, depot_id])
        tour_capacity_reached[len(Sub_tour)-1] = False
        return tour_capacity_reached, Sub_tour

    def return_tour_demand(self, tour):
        demand = 0
        for i in tour:
            demand += self.demand[i]
        return demand

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


    def insert_new(self, depot_id, Sub_tour = [], tour_capacity_reached = {}, visited = [], tour = 0, planning_df = None, current_order = 0, time = 0, inital_solution=False):
        if inital_solution:
            Sub_tour = []
            tour_capacity_reached = {}
            visited = []
            tour = 0
            planning_df = None
            current_order = 0        

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
            time_check = time_checker(temp_tour, self.dist_matrix, self.service_times, self.readytime, self.duetime)

            merge_demand = sum(self.demand[temp_tour])
            if time_check and merge_demand <= self.capacity:
                if self.return_tour_demand(temp_tour) <= self.capacity:
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
            
            
            planning_df = create_planning_df(Sub_tour, self.all_orders_df, self.dist_matrix, self.service_times, self.readytime)

        return Sub_tour, tour_capacity_reached, visited, tour, planning_df, current_order

    def run_greedy(self):
        time_start = time.time()
        depot_id = 0

        if(self.test_type == 'dynamic'):
            timer = 0
            print(f'Time: {timer}')
            self.current_order_df = self.all_orders_df[(
                self.all_orders_df.AVAILABLETIME <= timer)]
        else:
            self.current_order_df = self.all_orders_df
            timer = 480

        Sub_tour, tour_capacity_reached, visited, tour_id, planning_df, last_order = self.insert_new(depot_id = 0, time=timer, inital_solution=True)
        print(f'Total Cost: {total_distance(Sub_tour, self.dist_matrix)}')
        print (f'Tour: {Sub_tour}')
        
        if self.test_type == 'dynamic':
            for timer in range(123, 1236, 123):
                print(f'Time: {timer}')
                self.current_order_df = self.all_orders_df[(
                    self.all_orders_df.AVAILABLETIME <= timer) & (~self.all_orders_df['CUST_NO'].isin(visited))]
                Sub_tour, tour_capacity_reached, visited, tour_id, planning_df, last_order = self.insert_new(depot_id, Sub_tour, tour_capacity_reached, visited, tour_id, planning_df, last_order, timer)
                print(Sub_tour)
                print(total_distance(Sub_tour, self.dist_matrix))

        time_end = time.time()
        print('----------FINAL RESULT----------------')
        print(f'Total Cost: {total_distance(Sub_tour, self.dist_matrix)}')
        print(f'Tour:\n {Sub_tour}')
        print(f'number of vehicles: {len(Sub_tour)}')
        print(f'total run time: {time_end-time_start}')
        print('end')

        result_tuple = ('path: '+str(Sub_tour), 'distance: '+str(total_distance(Sub_tour, self.dist_matrix)), 
                        'vehicle_num: '+str(len(Sub_tour)),
                                    str([]))

        return result_tuple
        

