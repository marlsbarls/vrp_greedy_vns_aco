import numpy as np
import copy
import ast
import pandas as pd
import os
from math import radians, cos, sin, asin, sqrt
import vns.src.config.vns_config as cfg 
import vns.src.config.preprocessing_config as prep_cfg


class Node:
    # MOD: Arguments available time and end coordinates are passed and initialized
    def __init__(self, id:  int, x: float, y: float, demand: float, ready_time: float, due_time: float,
                 service_time: float, available_time: float, x_end: float, y_end: float):
        super()
        self.id = id

        if id == 0:
            self.is_depot = True
        else:
            self.is_depot = False

        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
        # # MOD: see above
        # if test_type == 'static':
        #     self.available_time = 0
        # elif test_type == 'dynamic':
        #     self.available_time = available_time

        self.available_time = available_time
        self.x_end = x_end
        self.y_end = y_end
        



class VrptwGraph:
    # MOD: Arguments path_handover, time_slice, source, minutes_per_km are passed and initialized
    def __init__(self, file_path, path_handover, time_slice, source, minutes_per_km, opt_time, service_time_matrix=None, order_ids=None, 
                 test_type='dynamic', rho=0.1):
        super()
         # MOD: Marlene
        self.test_type = test_type
        self.source = source
        if self.source == 'r':
            self.service_time_matrix = service_time_matrix
            self.order_ids = order_ids
        self.opt_time = opt_time
        # only relevant in _cal_nearest_next_index
        # original is False
        self.opt_time_next_index = False
        if self.opt_time == False:
            self.opt_time_next_index = False

        # MOD: see above
        self.minutes_per_km = minutes_per_km
        self.time_slice = time_slice
        self.file_path = file_path
        self.node_num, self.nodes, self.node_dist_mat, self.vehicle_num, self.vehicle_capacity, self.all_nodes \
            = self.create_from_file(file_path, path_handover)

        # Pheromone evaporation rate
        self.rho = rho

        # Creating a Pheromone Matrix
        self.nnh_travel_path, self.init_pheromone_val, _ = self.nearest_neighbor_heuristic()
        self.init_pheromone_val = 1 / (self.init_pheromone_val * self.node_num)

        self.pheromone_mat = np.ones((self.node_num, self.node_num)) * self.init_pheromone_val

        # Heuristic Information Matrix
        self.heuristic_info_mat = 1 / self.node_dist_mat

    def haversine(self, latitude_target, longitude_target, latitude_origin, longitude_origin):
        r = 6372.8
        d_latitude = radians(latitude_origin - latitude_target)
        d_longitude = radians(longitude_origin - longitude_target)
        latitude_target = radians(latitude_target)
        latitude_origin = radians(latitude_origin)

        a = sin(d_latitude / 2) ** 2 + cos(latitude_target) * cos(latitude_origin) * sin(d_longitude / 2) ** 2
        c = 2 * asin(sqrt(a))

        haversine_dist = r * c

        return haversine_dist

    @staticmethod
    def cal_total_travel_time(graph, path, service_time_matrix, order_ids, minutes_per_km=1):
        travel_time = 0
        current_time = 0
        vehicle_num = 0
        dist_dict = {}
        for i in range(0, len(path)-1):
            if path[i] == 0 and i != 0:
                dist_dict[vehicle_num] = travel_time
                current_time = 0
                vehicle_num += 1
                travel_time = 0
            dist = graph.node_dist_mat[path[i]][path[i+1]]*minutes_per_km
            if path[i] != 0:
                wait_time = max(graph.all_nodes[path[i+1]].ready_time - current_time - dist, 0)
                current_time += dist + wait_time
            else:
                wait_time = 0
                current_time = max(graph.all_nodes[path[i+1]].ready_time, dist)

            service_time = VrptwGraph.get_service_time(path[i+1], service_time_matrix, 
                                                            current_time, order_ids)
            current_time += service_time
            travel_time += dist + wait_time + service_time
            
        dist_dict[vehicle_num] = travel_time

        total_travel_time = sum(dist_dict.values())

        return total_travel_time
    
    def _cal_total_travel_distance(self, travel_path):
        distance = 0
        current_ind = travel_path[0]
        for next_ind in travel_path[1:]:
            distance += self.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind
        return distance

    def copy(self, init_pheromone_val):
        new_graph = copy.deepcopy(self)

        # pheromones
        new_graph.init_pheromone_val = init_pheromone_val
        new_graph.pheromone_mat = np.ones((new_graph.node_num, new_graph.node_num)) * init_pheromone_val

        return new_graph

    # MOD: Argument path_handover is passed to be able to address nodes included in handover file
    def create_from_file(self, file_path, path_handover):
        # Read the location of service points and customers from a file.
        self.node_list = []
        nodes = []

        # MOD: Marlene
        vehicle_num = cfg.num_vehicles
        vehicle_capacity = cfg.capacity

        order_df = pd.read_csv(file_path)
        for index, rows in order_df.iterrows():
            if self.source == 'r':
                if self.test_type == 'dynamic':
                    row_list = [str(rows.CUST_NO), str(rows.YCOORD), str(rows.XCOORD),  str(rows.DEMAND), str(rows.READYTIME), 
                            str(rows.DUETIME), str(rows.SERVICETIME), str(rows.READYTIME), str(rows.YCOORD_END), str(rows.XCOORD_END)]
                elif self.test_type == 'static':
                    row_list = [str(rows.CUST_NO), str(rows.YCOORD), str(rows.XCOORD),  str(rows.DEMAND), str(rows.READYTIME), 
                            str(rows.DUETIME), str(rows.SERVICETIME), str(0.0), str(rows.YCOORD_END), str(rows.XCOORD_END)]
                self.node_list.append(row_list)
            elif self.source == 't':
                if self.test_type == 'dynamic':
                    row_list = [str(int(rows.CUST_NO-1)), str(rows.YCOORD), str(rows.XCOORD),  str(rows.DEMAND), str(rows.READYTIME), 
                            str(rows.DUETIME), str(rows.SERVICETIME), str(rows.READYTIME), str(rows.YCOORD), str(rows.XCOORD)]
                elif self.test_type == 'static':
                    row_list = [str(int(rows.CUST_NO-1)), str(rows.YCOORD), str(rows.XCOORD),  str(rows.DEMAND), str(rows.READYTIME), 
                            str(rows.DUETIME), str(rows.SERVICETIME), str(0.0), str(rows.YCOORD), str(rows.XCOORD)]

                self.node_list.append(row_list)


        all_nodes = list(
            Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]),
                 float(item[6]), float(item[7]), float(item[8]), float(item[9])) for item in self.node_list)

        # MOD: In time slice 0, initiate nodes with available time 0,
        if self.time_slice == 0:
            nodes = list(Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]),
                              float(item[6]), float(item[7]), float(item[8]), float(item[9])) for item in self.node_list if float(item[7]) == 0)

        # MOD: After time slice 0, initiate nodes from handover file
        elif self.time_slice != 0:
            file = open(path_handover, 'r')
            lines = file.readlines()
            path_list = ast.literal_eval(lines[0])
            file.close()
            nodes = list(
                Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]),
                     float(item[6]), float(item[7]), float(item[8]), float(item[9])) for item in self.node_list if int(item[0]) in path_list)
        
        node_num = len(all_nodes)

        # Create distance matrix
        node_dist_mat = np.zeros((node_num, node_num))

        # MOD: For Chargery instances, choose Haversine distance for distance matrix
        for i in range(node_num):
            node_a = all_nodes[i]
            node_dist_mat[i][i] = 1e-8
            for j in range(i+1, node_num):
                node_b = all_nodes[j]
                if self.source == 't':
                    node_dist_mat[i][j] = self.calculate_dur_t(node_a, node_b)
                    node_dist_mat[j][i] = node_dist_mat[i][j]
                elif self.source == 'r':
                    node_dist_mat[i][j] = self.calculate_dur_r(node_a, node_b, orientation='ij')
                    node_dist_mat[j][i] = self.calculate_dur_r(node_a, node_b, orientation='ji')


        return node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity, all_nodes

    def calculate_dur_t(self, node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))

    # MOD: New method to calculate Haversine distance (retrieved from Preprocessing). An asymmetric matrix is generated,
    # by this, a location change can be taken into account
    def calculate_dur_r(self, node_a, node_b, orientation):
        if orientation == 'ij':
            return self.haversine(node_b.x, node_b.y, node_a.x_end, node_a.y_end) * cfg.minutes_per_kilometer
        elif orientation == 'ji':
            return self.haversine(node_a.x, node_a.y, node_b.x_end, node_b.y_end) * cfg.minutes_per_kilometer

    def local_update_pheromone(self, start_ind, end_ind):
        self.pheromone_mat[start_ind][end_ind] = (1-self.rho) * self.pheromone_mat[start_ind][end_ind] + \
                                                  self.rho * self.init_pheromone_val

    def global_update_pheromone(self, best_path, best_path_distance):
        '''
        Updating the pheromone matrix
        :return:
        '''
        self.pheromone_mat = (1-self.rho) * self.pheromone_mat

        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            self.pheromone_mat[current_ind][next_ind] += self.rho/best_path_distance
            current_ind = next_ind

    @staticmethod
    def get_service_time(next_index, service_time_matrix, time, order_ids):
        if next_index == 0:
            return 0
        else:
            traffic_phase = "off_peak" if time < prep_cfg.traffic_times["phase_transition"][
                    "from_shift_start"] else "phase_transition" if time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"
            service_time = service_time_matrix[order_ids[next_index]+":"+traffic_phase]
            return service_time

    def _get_total_travel_time(self, path, get_current_time=False):
        travel_time = 0
        current_time = 0
        vehicle_num = 0
        dist_dict = {}
        for i in range(0, len(path)-1):
            if path[i] == 0 and i != 0:
                dist_dict[vehicle_num] = travel_time
                current_time = 0
                vehicle_num += 1
                travel_time = 0
            dist = self.node_dist_mat[path[i]][path[i+1]]*self.minutes_per_km
            # no wait time if depot 
            if path[i] != 0:
                wait_time = max(self.all_nodes[path[i+1]].ready_time - current_time - dist, 0)
                current_time += dist + wait_time
            else:
                wait_time = 0
                current_time = max(self.all_nodes[path[i+1]].ready_time, dist)

            service_time = VrptwGraph.get_service_time(path[i+1], self.service_time_matrix, 
                                                            current_time, self.order_ids)
            current_time += service_time
            travel_time += dist + wait_time + service_time
            
        dist_dict[vehicle_num] = travel_time

        total_travel_time = sum(dist_dict.values())

        if get_current_time:
            return total_travel_time, current_time
        else:
            return total_travel_time
    # @staticmethod
    # def get_total_travel_time(path, dist_mat, ready_time, service_time_matrix, order_ids, minutes_per_km=1):
    #     travel_time = 0
    #     current_time = 0
    #     vehicle_num = 0
    #     dist_dict = {}
    #     # test_dict = {}
    #     # test_path = []
    #     for i in range(0, len(path)-1):
    #         if path[i] == 0 and i != 0:
    #             dist_dict[vehicle_num] = travel_time
    #             # test_dict[vehicle_num] = test_path
    #             current_time = 0
    #             vehicle_num += 1
    #             travel_time = 0
    #             # test_path = []
    #         dist = dist_mat[path[i]][path[i+1]]*minutes_per_km
    #         # no wait time if depot 
    #         if path[i] != 0:
    #             wait_time = max(ready_time[path[i+1]] - current_time - dist, 0)
    #             current_time += dist + wait_time
    #         else:
    #             wait_time = 0
    #             current_time = max(ready_time[path[i+1]], dist)

    #         service_time = VrptwGraph.get_service_time(path[i+1], service_time_matrix, 
    #                                                         current_time, order_ids)
    #         current_time += service_time
    #         travel_time += dist + wait_time + service_time
    #         # test_path.append(path[i])
            
    #     dist_dict[vehicle_num] = travel_time
    #     # test_dict[vehicle_num] = test_path

    #     total_travel_time = sum(dist_dict.values())

    #     return total_travel_time

    # def nearest_neighbor_heuristic(self, max_vehicle_num=None):
    #     index_to_visit = []

    #     # MOD: only get nodes with available time 0 for index to visit
    #     for i in range(1, self.node_num):
    #         if self.all_nodes[i].available_time == 0:
    #             index_to_visit.append(i)

    #     current_index = 0
    #     current_load = 0
    #     current_time = 0
    #     travel_distance = 0
    #     travel_path = [0]
    #     time_travelled = 0
    #     time_dict = {}
    #     veh_num = -1
    #     # test_dict = {}

    #     if max_vehicle_num is None:
    #         max_vehicle_num = self.node_num

    #     while len(index_to_visit) > 0 and max_vehicle_num > 0:
    #         nearest_next_index = self._cal_nearest_next_index(index_to_visit, current_index, current_load, current_time)
    #         if nearest_next_index is None:
    #             travel_distance += self.node_dist_mat[current_index][0]
    #             if self.opt_time:
    #                 time_travelled += self.node_dist_mat[current_index][0]*self.minutes_per_km
    #             current_load = 0
    #             current_time = 0
    #             travel_path.append(0)
    #             current_index = 0
                

    #             max_vehicle_num -= 1
    #         else:
    #             # MOD: Marlene
    #             travel_path.append(nearest_next_index)
    #             if self.opt_time:
    #                 # if current_index == 0:
    #                 #     if veh_num >= 0:
    #                 #         time_dict[veh_num] = time_travelled  
    #                 #         # test_dict[veh_num_test] = test_path
    #                 #     veh_num += 1
    #                 #     time_travelled = 0
    #                     # test_path = []
                    
    #             # current_load += self.all_nodes[nearest_next_index].demand + self.node_dist_mat[current_index][nearest_next_index]
    #             dist = self.node_dist_mat[current_index][nearest_next_index]

    #             # MOD: Marlene
                
    #             # wait time only when driver is not at depot
    #             if current_index != 0:
    #                 wait_time = max(self.all_nodes[nearest_next_index].ready_time - current_time - dist, 0)
    #                 current_time += dist*self.minutes_per_km + wait_time
    #                 service_time = VrptwGraph.get_service_time(nearest_next_index, self.service_time_matrix, 
    #                                                         current_time, self.order_ids)
    #                 current_time += service_time
    #             elif current_index == 0:

    #                 wait_time = 0
    #                 current_time = max(self.all_nodes[nearest_next_index].ready_time, dist*self.minutes_per_km)
    #                 service_time = VrptwGraph.get_service_time(nearest_next_index, self.service_time_matrix, 
    #                                                         current_time, self.order_ids)
    #                 current_time += service_time

    #             current_load += service_time + self.node_dist_mat[current_index][nearest_next_index]
                    

    #             index_to_visit.remove(nearest_next_index)

    #             travel_distance += self.node_dist_mat[current_index][nearest_next_index]
    #             if self.opt_time:
    #                     time_travelled += dist*self.minutes_per_km + wait_time + service_time
    #             travel_path.append(nearest_next_index)
    #             # test_path.append(nearest_next_index)
    #             current_index = nearest_next_index

    #     # 最后要回到depot
    #     # And finally, back to the depot.
    #     travel_distance += self.node_dist_mat[current_index][0]
    #     if self.opt_time:
    #         time_travelled += self.node_dist_mat[current_index][0]*self.minutes_per_km 
    #         time_dict[veh_num] = time_travelled
    #         total_time_travelled = sum(time_dict.values())
    #         # test_dict[veh_num_test] = test_path
    #     travel_path.append(0)
        
    #     vehicle_num = travel_path.count(0)-1
    #     if self.opt_time:
    #         test = self._get_total_travel_time(travel_path)
    #         # test2 = get_total_travel_time(travel_path, self.node_dist_mat, ready_time, service_time_matrix, order_ids, )
            
    #         return travel_path, total_time_travelled, vehicle_num
    #     elif not self.opt_time:
    #         return travel_path, travel_distance, vehicle_num

    def get_current_path_section(self, path):
        for idx, order in reversed(list(enumerate(path))):
            if order == 0:
                current_path_section = path[idx:]
                return current_path_section
    
    def _check_condition(self, travel_path, next_index, service_time_mat, order_id, path_section= True, minutes_per_km=1) -> bool:
        temp_travel_path = travel_path.copy()
        temp_travel_path.append(next_index)
        
        if path_section:
            temp_travel_path = self.get_current_path_section(temp_travel_path)
        temp_travel_path.append(0)
        tour = temp_travel_path

        time = 0
        counter = 0
        for i in range(1, len(tour)):
            traffic_phase = "off_peak" if time < prep_cfg.traffic_times["phase_transition"][
                "from_shift_start"] else "phase_transition" if time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"

            
            if(order_id[tour[i-1]] != 'order_0'):
                if(self.all_nodes[tour[i-1]].ready_time  <= (time + service_time_mat[order_id[tour[i-1]]+":"+traffic_phase])):
                    time = time + \
                        service_time_mat[order_id[tour[i-1]]+":"+traffic_phase] + \
                        self.node_dist_mat[tour[i-1]][tour[i]]*minutes_per_km
                else:
                    time = self.all_nodes[tour[i-1]].ready_time + \
                        self.node_dist_mat[tour[i-1]][tour[i]]*minutes_per_km
            else:
                if(self.all_nodes[tour[i-1]].ready_time <= time):
                    time = time + \
                        self.node_dist_mat[tour[i-1]][tour[i]]*minutes_per_km
            if (time <= self.all_nodes[tour[i]].due_time):
                counter += 1
            else:
                break
        
        orders_no_depot = [i for i in tour if i != 0]

        # if counter == len(orders_no_depot):
        if counter == len(tour) - 1:
            return True
        else:
            return False

    def nearest_neighbor_heuristic(self, max_vehicle_num=None):
        index_to_visit = []

        # MOD: only get nodes with available time 0 for index to visit
        for i in range(1, self.node_num):
            if self.all_nodes[i].available_time == 0:
                index_to_visit.append(i)

        current_index = 0
        travel_distance = 0
        total_travel_path = [0]
        current_tour = [0]

        if max_vehicle_num is None:
            max_vehicle_num = self.node_num

        while len(index_to_visit) > 0 and max_vehicle_num > 0:
            nearest_next_index = self._cal_nearest_next_index(index_to_visit, current_index, current_tour)
         
            if nearest_next_index is None:
                total_travel_path.append(0)
                current_index = 0
                max_vehicle_num -= 1
                current_tour = [0]
    
            else:
                total_travel_path.append(nearest_next_index)
                current_tour.append(nearest_next_index)
                current_load, current_time = self._get_total_travel_time(total_travel_path, True)
                index_to_visit.remove(nearest_next_index)
                current_index = nearest_next_index
   

        total_travel_path.append(0)
        travel_time = self._get_total_travel_time(total_travel_path)
        travel_distance = self._cal_total_travel_distance(total_travel_path) 

        vehicle_num = total_travel_path.count(0)-1
        if self.opt_time:
            return total_travel_path, travel_time, vehicle_num
        elif not self.opt_time:
            return total_travel_path, travel_distance, vehicle_num

    def _cal_nearest_next_index(self, index_to_visit, current_index, travel_path, minutes_per_km=1):
        '''
        next_index Find the nearest reachable next_index
        :param index_to_visit:
        :return:
        '''
        nearest_ind = None
        nearest_distance = None
        nearest_travel_time = None
        for next_index in index_to_visit:
            test_path = travel_path.copy()
            test_path.append(next_index)
            test_path.append(0)

            time_check = self._check_condition(travel_path, next_index, self.service_time_matrix, self.order_ids)
            if not time_check:
                continue
            
            travel_time = self.node_dist_mat[current_index][next_index] * minutes_per_km
            travel_distance = self.node_dist_mat[current_index][next_index]
            
            if not self.opt_time:
                if nearest_distance is None or travel_distance < nearest_distance:
                    nearest_distance = travel_distance
                    nearest_ind = next_index
            elif self.opt_time:
                if nearest_travel_time is None or travel_time < nearest_travel_time:
                    nearest_travel_time = self.node_dist_mat[current_index][next_index]
                    nearest_ind = next_index

        return nearest_ind


class PathMessage:
    def __init__(self, path, distance):
        if path is not None:
            self.path = copy.deepcopy(path)
            self.distance = copy.deepcopy(distance)
            self.used_vehicle_num = self.path.count(0) - 1
        else:
            self.path = None
            self.distance = None
            self.used_vehicle_num = None

    def get_path_info(self):
        return self.path, self.distance, self.used_vehicle_num
