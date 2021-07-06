import ast
from aco.vrptw_base import VrptwGraph, Node
import numpy as np
import pandas as pd
import vns.src.config.vns_config as cfg 


class UpdateProcess:
    def __init__(self, graph: VrptwGraph, path_handover, time_slice, interval_length, path_testfile, source,
                 minutes_per_km, service_time_matrix, order_ids, test_type):
        # MOD: Marlene
        self.service_time_matrix = service_time_matrix
        self.order_ids = order_ids
        self.test_type = test_type
        
        self.graph = graph
        self.path_handover = path_handover
        self.path_testfile = path_testfile
        self.source = source
        self.minutes_per_km = minutes_per_km
        self.time_slice = time_slice
        self.interval_length = interval_length
        self.interval_start_time = self.time_slice * self.interval_length
        self.time_passed = (self.time_slice + 1) * self.interval_length
        self.next_time_passed = (self.time_slice + 2) * self.interval_length
        self.all_nodes, self.all_nodes_num, self.node_dist_mat, self.vehicle_num, _,  = self.create_from_file(
            path_testfile)

        self.current_best_path, self.current_best_distance, self.current_best_vehicle_num = self.get_current_best()

        self.committed_nodes = []

    # Create node list in required format and create distance matrix (adopted from original MACS)
    def create_from_file(self, file_path):
        node_list = []

        # with open(file_path, 'rt') as f:
        #     count = 1
        #     for line in f:
        #         if count == 5:
        #             get num and capacity from cfg
        #             vehicle_num, vehicle_capacity = line.split()
        #             vehicle_num = int(vehicle_num)
        #             vehicle_capacity = int(vehicle_capacity)
                    
        #         elif count >= 9:
        #             node_list.append(line.split())
        #         count += 1

        vehicle_num = cfg.num_vehicles
        vehicle_capacity = cfg.capacity
        order_df = pd.read_csv(file_path)
        for index, rows in order_df.iterrows():
            if self.test_type == 'static':
                row_list = [str(rows.CUST_NO), str(rows.YCOORD), str(rows.XCOORD),  str(rows.DEMAND), str(0), 
                            str(rows.DUETIME), str(rows.SERVICETIME), str(0), str(rows.YCOORD_END), str(rows.XCOORD_END)]
                node_list.append(row_list)
            elif self.test_type == 'dynamic':
                row_list = [str(rows.CUST_NO), str(rows.YCOORD), str(rows.XCOORD),  str(rows.DEMAND), str(rows.READYTIME), 
                            str(rows.DUETIME), str(rows.SERVICETIME), str(rows.READYTIME), str(rows.YCOORD_END), str(rows.XCOORD_END)]
                node_list.append(row_list)

        all_nodes_list = list(
                Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]),
                     float(item[6]), float(item[7]), float(item[8]), float(item[9])) for item in node_list)

        all_nodes_num = len(all_nodes_list)

        node_dist_mat = np.zeros((all_nodes_num, all_nodes_num))
        for i in range(all_nodes_num):
            node_a = all_nodes_list[i]
            node_dist_mat[i][i] = 1e-8

            for j in range(i+1, all_nodes_num):
                node_b = all_nodes_list[j]

                if self.source == 't':
                    node_dist_mat[i][j] = self.graph.calculate_dur_t(node_a, node_b)
                    node_dist_mat[j][i] = node_dist_mat[i][j]

                elif self.source == 'r':
                    node_dist_mat[i][j] = self.graph.calculate_dur_r(node_a, node_b, orientation='ij')
                    node_dist_mat[j][i] = self.graph.calculate_dur_r(node_a, node_b, orientation='ji')

        return all_nodes_list, all_nodes_num, node_dist_mat, vehicle_num, vehicle_capacity

    # Read current best information from handover file
    def get_current_best(self):
        file = open(self.path_handover, 'r')
        lines = file.readlines()
        current_best_path = ast.literal_eval(lines[0])
        current_best_distance = float(lines[1])
        current_best_vehicle_num = int(lines[2])
        file.close()

        return current_best_path, current_best_distance, current_best_vehicle_num

    # Identify committed nodes
    def get_committed_nodes(self):
        committed_nodes = []
        travel_time = 0
        depot_counter = -1
        depot_idx = [index for index, value in enumerate(self.current_best_path) if value == 0]
        depot_num = len(depot_idx)

        for i in range(0, len(self.current_best_path)+1):

            if self.graph.all_nodes[self.current_best_path[i]].id == 0:
                depot_counter += 1

                if depot_counter == depot_num-1:
                    break

                else:
                    travel_time = 0
                    following_idx = self.current_best_path[depot_idx[depot_counter]+1]
                    current_idx = self.current_best_path[depot_idx[depot_counter]]
                    if self.graph.all_nodes[following_idx].available_time != 0:
                        travel_time += self.graph.all_nodes[following_idx].available_time
                    travel_time += self.node_dist_mat[current_idx][following_idx]

                    if travel_time < self.time_passed:
                        committed_nodes.append(self.current_best_path[i])

                    else:
                        continue

            else:
                following_idx = self.graph.all_nodes[self.current_best_path[i+1]].id
                current_idx = self.current_best_path[i]
                travel_time_previous = travel_time

                # MOD: Marlene 
                travel_time += self.node_dist_mat[current_idx][following_idx]
                service_time = VrptwGraph.get_service_time(current_idx, self.service_time_matrix, 
                                                           travel_time, self. order_ids)
                travel_time += service_time
                

                if travel_time <= self.next_time_passed:
                    committed_nodes.append(current_idx)

                elif travel_time >= self.time_passed >= travel_time_previous:
                    committed_nodes.append(current_idx)

                elif self.next_time_passed >= travel_time_previous >= self.time_passed:
                    committed_nodes.append(current_idx)

                else:
                    continue

        committed_nodes.append(0)

        return committed_nodes

    # Check for new nodes
    def check_new_nodes(self):
        new_nodes = []
        for i in range(1, len(self.all_nodes)):
            if self.all_nodes[i].available_time != 0 and self.interval_start_time <= self.all_nodes[i].available_time <\
                    self.time_passed:
                new_nodes.append(i)
        print('Time slice:', self.time_slice+1, ', new nodes:', new_nodes)

        return new_nodes

    # Identify possible insertion indices
    def insertion_idx(self):
        self.committed_nodes = self.get_committed_nodes()
        possible_insert_idx = []
        for i in range(0, len(self.current_best_path)):
            if self.all_nodes[self.current_best_path[i]].id in self.committed_nodes:
                continue

            else:
                possible_insert_idx.append(i)

        possible_insert_idx.append(len(self.current_best_path)-1)

        return possible_insert_idx

    # Check if conditions for insertion are fulfilled
    def check_condition(self, vehicle_travel_time, i, following_idx) -> bool:
        if vehicle_travel_time + self.all_nodes[following_idx].demand + self.node_dist_mat[i][following_idx] > self.\
                graph.vehicle_capacity:  # vehicle_travel_time==vehicle_load
            return False

        travel_time = self.node_dist_mat[i][following_idx]
        wait_time = max(self.all_nodes[following_idx].ready_time - vehicle_travel_time - travel_time, 0)

        # MOD: Marlene
        current_time = vehicle_travel_time + travel_time + wait_time
        service_time = VrptwGraph.get_service_time(following_idx, self.service_time_matrix, current_time, self.order_ids)

        # Checking to see if you can return to the depot after visiting a particular customer.
        if vehicle_travel_time + travel_time + wait_time + service_time + self.node_dist_mat[following_idx][0] > \
                self.all_nodes[0].due_time:
            return False

        # No service for customers outside of due time.
        if vehicle_travel_time + travel_time > self.all_nodes[following_idx].due_time:
            return False

        return True

    # Create shuttle tour if node can not included in existing tours
    def new_shuttle_tour(self, node, add_dist):
        self.current_best_path.append(node)
        self.current_best_path.append(0)
        self.current_best_vehicle_num = self.current_best_path.count(0) - 1
        add_dist += self.node_dist_mat[0][node] + self.node_dist_mat[node][0]

        return add_dist

    # Insertion of new nodes in current solution
    def insertion(self):
        new_nodes = self.check_new_nodes()
        if new_nodes:
            for node in new_nodes:
                add_dist = 0
                best_distance = 0
                best_ins_idx = None
                best_total_travel_time = 99999
                insertion = True
                total_addition = False
                ins_ind_list = self.insertion_idx()

                for ins_ind in ins_ind_list:
                    total_travel_time = 0
                    total_distance = 0
                    test_path = self.current_best_path.copy()
                    test_path.insert(ins_ind, node)
                    test_path_depot_idx = [index for index, value in enumerate(test_path) if value == 0]
                    depot_num = len(test_path_depot_idx)

                    for depot in test_path_depot_idx:
                        travel_time = 0
                        distance = 0

                        if test_path_depot_idx.index(depot) == depot_num-1 or insertion is False:
                            insertion = True
                            break

                        else:
                            cur_depot = depot+1
                            idx = test_path_depot_idx.index(depot)+1
                            next_depot = test_path_depot_idx[idx]+1
                            path_section = test_path[cur_depot:next_depot]
                            travel_time += self.node_dist_mat[0][path_section[0]]
                            distance += self.node_dist_mat[0][path_section[0]]
                            for i in path_section:
                                if i != 0:
                                    following_idx = path_section[path_section.index(i)+1]

                                    if self.check_condition(travel_time, i, following_idx) is False:
                                        insertion = False
                                        total_addition = False
                                        break

                                    else:
                                        total_addition = True
                                        
                                        # MOD: Marlene
                                        travel_time += self.node_dist_mat[i][following_idx]
                                        service_time = VrptwGraph.get_service_time(following_idx, self.service_time_matrix,
                                                                                   travel_time, self.order_ids)
                                        travel_time += service_time

                                        distance += self.node_dist_mat[i][following_idx]

                        if total_addition:
                            total_travel_time += travel_time
                            total_distance += distance

                        else:
                            total_travel_time = 99999

                    if total_travel_time < best_total_travel_time:
                        best_total_travel_time = total_travel_time
                        best_ins_idx = ins_ind
                        best_distance = total_distance

                if best_ins_idx is None:
                    add_dist += self.new_shuttle_tour(node, add_dist)
                    self.current_best_distance += add_dist

                else:
                    self.current_best_path.insert(best_ins_idx, node)
                    self.current_best_vehicle_num = self.current_best_path.count(0)-1
                    self.current_best_distance = best_distance

    # Carry out insertion and print result to handover file
    def print_result_to_file(self):
        self.insertion()

        # MOD: Marlene
        cost = self.current_best_distance * cfg.cost_per_minute + self.current_best_vehicle_num + cfg.cost_per_driver

        result_tuple = (str(self.current_best_path), str(self.current_best_distance), str(
            self.current_best_vehicle_num), str(self.committed_nodes), str(cost))
        file = open(self.path_handover, 'w')
        separator = '\n'
        file.write(separator.join(result_tuple))
        file.close()

    # Run Update Process through starting print method
    def runupdateprocess(self):
        self.print_result_to_file()
        print('-----UPDATE PROCESS FINALIZED-----')
