import numpy as np
import copy
from aco.vrptw_base import VrptwGraph
from threading import Event
import ast
import time
import vns.src.config.preprocessing_config as prep_cfg
import vns.src.config.vns_config as cfg


class Ant:
    # MOD: Arguments path handover and time slice passed and initialized
    def __init__(self, path_handover, time_slice, graph: VrptwGraph, source, opt_time, service_time_matrix = None, 
                 order_ids = None, start_index=0, min_per_km=1):
        super()
        self.graph = graph
        self.current_index = start_index
        self.vehicle_load = 0
        self.vehicle_travel_time = 0

        # MOD: see above
        self.time_slice = time_slice
        self.path_handover = path_handover

        # MOD: Marlene
        self.source = source
        if self.source == 'r':
            self.service_time_matrix = service_time_matrix
            self.order_ids = order_ids
        self.opt_time = opt_time
        self.min_per_km = min_per_km
        
        # MOD: For index to visit, distinguish between time slice 0 and others
        if self.time_slice == 0:
            self.index_to_visit = []

            for i in range(len(self.graph.all_nodes)):
                if self.graph.all_nodes[i].available_time == 0:
                    self.index_to_visit.append(self.graph.all_nodes[i].id)


            self.index_to_visit.remove(start_index)

        else:
            self.file = open(path_handover, "r")
            self.lines = self.file.readlines()
            while len(self.lines) < 4:
                self.file.close()
                time.sleep(1)
                self.file = open(path_handover, "r")
                self.lines = self.file.readlines()

            self.index_to_visit = ast.literal_eval(self.lines[0])
            self.committed_path = ast.literal_eval(self.lines[3])
            self.file.close()
            self.index_to_visit = [elem for elem in self.index_to_visit if elem != 0]
            index_to_visit1 = [elem for elem in self.index_to_visit if elem in self.committed_path]
            index_to_visit2 = [elem for elem in self.index_to_visit if elem not in self.committed_path]
            index_to_visit2.sort()
            self.index_to_visit = index_to_visit1 + index_to_visit2

        self.travel_path = [start_index]
        self.arrival_time = [0]
        self.total_travel_distance = 0

        self.total_travel_time = 0

        # self.check_condition(1, self.service_time_matrix, self.order_ids, path_section=False)


    def clear(self):
        self.travel_path.clear()
        self.index_to_visit.clear()
    
    def get_current_path_section(self, path):
        for idx, order in reversed(list(enumerate(path))):
            if order == 0:
                
                current_path_section = path[idx:]
                return current_path_section

    def move_to_next_index(self, next_index):
        # 更新蚂蚁路径
        # Updating ant paths
        self.travel_path.append(next_index)
        self.total_travel_distance = Ant.cal_total_travel_distance(self.graph, self.travel_path)

        dist = self.graph.node_dist_mat[self.current_index][next_index]
        test = self.vehicle_travel_time + dist
        travel_time, current_time = Ant.cal_total_travel_time(self.graph, self.travel_path, self.service_time_matrix, self.order_ids, get_current_time=True)
        service_time = VrptwGraph.get_service_time(next_index, self.service_time_matrix, self.vehicle_travel_time, self.order_ids)
        self.arrival_time.append(current_time - service_time)

        self.total_travel_time = travel_time

        path_section = self.get_current_path_section(self.travel_path)

        self.vehicle_load = current_time
        self.vehicle_travel_time = Ant.cal_total_travel_time(self.graph, path_section, self.service_time_matrix, self.order_ids)

        if not self.graph.all_nodes[next_index].is_depot:
            self.index_to_visit.remove(next_index)

        self.current_index = next_index

    def index_to_visit_empty(self):
        return len(self.index_to_visit) == 0

    def get_active_vehicles_num(self):
        return self.travel_path.count(0)-1


    # def check_condition_test(self, next_index, path_section = False, minutes_per_km=1) -> bool:
    #     temp_travel_path = self.travel_path.copy()
    #     temp_travel_path.append(next_index)
    #     temp_travel_time = 0
    #     current_time = 0

    #     if path_section:
    #         temp_travel_path = self.get_current_path_section(temp_travel_path)
    #     temp_travel_path.append(0)

    #     for i in range(0, len(temp_travel_path)-1):
    #         dist = self.graph.node_dist_mat[temp_travel_path[i]][temp_travel_path[i+1]]*minutes_per_km
    #         if temp_travel_path[i] != 0:
    #             wait_time = max(self.graph.all_nodes[temp_travel_path[i+1]].ready_time - current_time - dist, 0)
    #             current_time += dist + wait_time
    #         else:
    #             wait_time = 0
    #             current_time = max(self.graph.all_nodes[temp_travel_path[i+1]].ready_time, dist)

    #         service_time = VrptwGraph.get_service_time(temp_travel_path[i+1], self.service_time_matrix, 
    #                                                         current_time, self.order_ids)

    #         if self.graph.all_nodes[temp_travel_path[i+1]].due_time < current_time:
    #             return False

    #         current_time += service_time
            

    #         temp_travel_time += dist + wait_time + service_time
            
    #     # dist_depot = self.graph.node_dist_mat[temp_travel_path[i+1]][0]*minutes_per_km
    #     if current_time > cfg.capacity:
    #         return False

    #     # think this can be deleted
    #     if temp_travel_time > cfg.capacity:
    #         return False

    #     return True


    # def check_condition(self, next_index, path_section = False, minutes_per_km=1) -> bool:
    #     temp_travel_path = self.travel_path.copy()
    #     temp_travel_path.append(next_index)
    #     temp_travel_time = 0
    #     current_time = 0

    #     if path_section:
    #         temp_travel_path = self.get_current_path_section(temp_travel_path)
    #     temp_travel_path.append(0)

    #     for i in range(0, len(temp_travel_path)-1):
    #         dist = self.graph.node_dist_mat[temp_travel_path[i]][temp_travel_path[i+1]]*minutes_per_km
    #         if temp_travel_path[i] != 0:
    #             wait_time = max(self.graph.all_nodes[temp_travel_path[i+1]].ready_time - current_time - dist, 0)
    #             current_time += dist + wait_time
    #         else:
    #             wait_time = 0
    #             current_time = max(self.graph.all_nodes[temp_travel_path[i+1]].ready_time, dist)

    #         service_time = VrptwGraph.get_service_time(temp_travel_path[i+1], self.service_time_matrix, 
    #                                                         current_time, self.order_ids)

    #         if self.graph.all_nodes[temp_travel_path[i+1]].due_time < current_time:
    #             return False

    #         current_time += service_time
            

    #         temp_travel_time += dist + wait_time + service_time
            
    #     # dist_depot = self.graph.node_dist_mat[temp_travel_path[i+1]][0]*minutes_per_km
    #     if current_time > cfg.capacity:
    #         return False

    #     # think this can be deleted
    #     if temp_travel_time > cfg.capacity:
    #         return False

    #     return True

    def check_condition(self, next_index, service_time_mat, order_id, path_section = True, minutes_per_km=1) -> bool:
        temp_travel_path = self.travel_path.copy()
        temp_travel_path.append(next_index)
   

        if path_section:
            temp_travel_path = self.get_current_path_section(temp_travel_path)
        temp_travel_path.append(0)

        tour = temp_travel_path

        # test = [0, 132, 2, 129, 63, 134, 74, 128, 130, 3, 120, 8, 126, 68, 67, 136, 137, 79, 95, 142, 143, 87, 131, 108, 133, 80, 6, 0, 124, 105, 114, 115, 116, 66, 64, 7, 4, 93, 94, 83, 9, 1, 90, 101, 125, 76, 113, 140, 112, 75, 5, 138, 0, 65, 98, 109, 62, 61, 97, 127, 86, 118, 85, 88, 73, 122, 102, 119, 123, 117, 82, 104, 77, 91, 103, 106, 100, 141, 139, 121, 70, 72, 99, 0, 135, 92, 81, 84, 96, 89, 58, 60, 13, 36, 41, 37, 20, 23, 44, 54, 32, 47, 27, 21, 24, 29, 34, 15, 38, 25, 0, 111, 78, 110, 17, 43, 35, 53, 16, 39, 11, 42, 52, 10, 28, 31, 26, 55, 45, 48, 59, 14, 40, 33, 50, 0, 22, 12, 46, 30, 18, 19, 51, 56, 57, 71, 107, 69, 49, 0]
        # test = [0, 132, 2, 129, 63, 134, 74, 128, 130, 3, 120, 8, 126, 68, 67, 136, 137, 79, 95, 142, 143, 87, 131, 108, 133, 80, 6, 0]
        # test = [0, 35, 38, 43, 15, 31, 42, 29, 61, 11, 80, 81, 9, 10, 12, 13, 53, 105, 6, 25, 0]
        # test = [0, 77, 78, 83, 85, 0]
        # [0, 49, 55, 47, 48, 40, 46, 39, 50, 63, 64, 42, 0, 32, 29, 22, 24, 25, 41, 8, 9, 12, 52, 93, 82, 86, 44, 71, 27, 0, 65, 68, 77, 80, 75, 76, 83, 96, 92, 117, 101, 56, 26, 124, 0, 33, 36, 69, 70, 66, 62, 79, 2, 18, 13, 10, 19, 17, 7, 6, 11, 112, 0, 21, 51, 28, 54, 3, 4, 20, 15, 16, 1, 14, 73, 116, 5, 74, 134, 113, 88, 0, 61, 37, 57, 38, 35, 30, 0, 85, 90, 43, 67, 102, 87, 109, 59, 0, 78, 81, 84, 91, 97, 108, 100, 98, 60, 53, 127, 58, 107, 0, 115, 72, 89, 45, 106, 0, 95, 99, 105, 23, 103, 119, 125, 110, 121, 111, 0, 133, 135, 136, 31, 0, 94, 34, 118, 120, 122, 104, 114, 137, 128, 0, 126, 130, 129, 123, 0, 131, 132, 0]
        # tour = test
        time = 0
        counter = 0
        for i in range(1, len(tour)):
            traffic_phase = "off_peak" if time < prep_cfg.traffic_times["phase_transition"][
                "from_shift_start"] else "phase_transition" if time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"

            # if tour[i] == 0:
            #     time = 0
            #     continue
            
            if(order_id[tour[i-1]] != 'order_0'):
                if(self.graph.all_nodes[tour[i-1]].ready_time  <= (time + service_time_mat[order_id[tour[i-1]]+":"+traffic_phase])):
                    time = time + \
                        service_time_mat[order_id[tour[i-1]]+":"+traffic_phase] + \
                        self.graph.node_dist_mat[tour[i-1]][tour[i]]*minutes_per_km
                else:
                    time = self.graph.all_nodes[tour[i-1]].ready_time + \
                        self.graph.node_dist_mat[tour[i-1]][tour[i]]*minutes_per_km
            else:
                if(self.graph.all_nodes[tour[i-1]].ready_time <= time):
                    time = time + \
                        self.graph.node_dist_mat[tour[i-1]][tour[i]]*minutes_per_km
            if (time <= self.graph.all_nodes[tour[i]].due_time):
                counter += 1
            else:
                break
        
        orders_no_depot = [i for i in tour if i != 0]

        # if counter == len(orders_no_depot):
        if counter == len(tour) - 1:
            return True
        else:
            return False

    def check_condition_check_ant(self, path_section, next_index, service_time_mat, order_id, minutes_per_km=1) -> bool:
        temp_travel_path = path_section.copy()
        temp_travel_path.append(next_index)

        temp_travel_path.append(0)

        tour = temp_travel_path

        time = 0
        counter = 0
        for i in range(1, len(tour)):
            traffic_phase = "off_peak" if time < prep_cfg.traffic_times["phase_transition"][
                "from_shift_start"] else "phase_transition" if time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"
            
            if(order_id[tour[i-1]] != 'order_0'):
                if(self.graph.all_nodes[tour[i-1]].ready_time  <= (time + service_time_mat[order_id[tour[i-1]]+":"+traffic_phase])):
                    time = time + \
                        service_time_mat[order_id[tour[i-1]]+":"+traffic_phase] + \
                        self.graph.node_dist_mat[tour[i-1]][tour[i]]*minutes_per_km
                else:
                    time = self.graph.all_nodes[tour[i-1]].ready_time + \
                        self.graph.node_dist_mat[tour[i-1]][tour[i]]*minutes_per_km
            else:
                if(self.graph.all_nodes[tour[i-1]].ready_time <= time):
                    time = time + \
                        self.graph.node_dist_mat[tour[i-1]][tour[i]]*minutes_per_km
            if (time <= self.graph.all_nodes[tour[i]].due_time):
                counter += 1
            else:
                break
        

        if counter == len(tour) - 1:
            return True
        else:
            return False
  
    

    
    def check_feasibility(self, minutes_per_km = 1):
        path = self.travel_path
        travel_time = 0
        current_time = 0
        vehicle_num = 0
        dist_dict = {}
        total_idle_time = 0

        for i in range(0, len(path)-1):
            if path[i] == 0 and i != 0:
                dist_dict[vehicle_num] = travel_time
                current_time = 0
                vehicle_num += 1
                travel_time = 0
            dist = self.graph.node_dist_mat[path[i]][path[i+1]]*minutes_per_km
            if path[i] != 0:
                wait_time = max(self.graph.all_nodes[path[i+1]].ready_time - current_time - dist, 0)
                current_time += dist + wait_time
                total_idle_time += wait_time
            else:
                wait_time = 0
                current_time = max(self.graph.all_nodes[path[i+1]].ready_time, dist)

            service_time = VrptwGraph.get_service_time(path[i+1], self.service_time_matrix, 
                                                            current_time, self.order_ids)
            if current_time > self.graph.all_nodes[path[i+1]].due_time:
                return False
            current_time += service_time
            travel_time += dist + wait_time + service_time
            
        if current_time > 480:
            return False
        else:
            return True

    # MOD: Major changes to ensure that for each new tour, the committed nodes are appended first, followed by all
    # other possible nodes. The committed path is one of the key elements to this method.
    def cal_next_index_meet_constrains(self):
        """
        找出所有从当前位置（ant.current_index）可达的customer
        Find all the customers that can be reached from the current position (ant.current_index).
        :return:
        """
        next_index_meet_constrains = []
        if self.time_slice == 0:
            for next_ind in self.index_to_visit:
                if self.check_condition(next_ind, self.service_time_matrix, self.order_ids, path_section=True):
                    next_index_meet_constrains.append(next_ind)

        else:
            for next_ind in self.index_to_visit:
                if self.current_index == 0 and next_ind in self.committed_path:
                    if self.committed_path[self.committed_path.index(next_ind) -1] == 0:
                        next_index_meet_constrains.append(next_ind)
                    elif self.committed_path[self.committed_path.index(next_ind) -1] != 0:
                        continue

                elif self.current_index == 0 and next_ind not in self.committed_path:
                    continue

                elif self.current_index in self.committed_path and next_ind in self.committed_path and self.current_index != 0 and next_ind != 0:
                    if self.committed_path.index(self.current_index) < len(self.committed_path)-1:
                        if self.committed_path[self.committed_path.index(self.current_index)+1] == 0:
                            continue

                        elif next_ind == self.committed_path[self.committed_path.index(self.current_index)+1]:
                            next_index_meet_constrains.append(next_ind)
                            break

                        elif next_ind != self.committed_path[self.committed_path.index(self.current_index)+1]:
                            continue

                elif self.current_index in self.committed_path and next_ind not in self.committed_path:
                    if self.committed_path.index(self.current_index) < len(self.committed_path)-1:
                        if self.committed_path[self.committed_path.index(self.current_index)+1] == 0 and self.check_condition(next_ind, self.service_time_matrix, self.order_ids, path_section=True):
                            next_index_meet_constrains.append(next_ind)

                        elif self.committed_path[self.committed_path.index(self.current_index)+1] != 0 or self.check_condition(next_ind, self.service_time_matrix, self.order_ids, path_section=True) is False:
                            continue

                elif self.current_index not in self.committed_path and next_ind in self.committed_path:
                    continue

                elif self.current_index not in self.committed_path and next_ind not in self.committed_path:
                    if self.check_condition(next_ind, self.service_time_matrix, self.order_ids, path_section=True):
                        next_index_meet_constrains.append(next_ind)
                    elif self.check_condition(next_ind, self.service_time_matrix, self.order_ids, path_section=True) is False:
                        continue

        return next_index_meet_constrains

    def cal_nearest_next_index(self, next_index_list):
        """
        从待选的customers中选择，离当前位置（ant.current_index）最近的customer
        Select the closest customer to the current position (ant.current_index) from the list of customers to be
        selected.
        :param next_index_list:
        :return:
        """
        current_ind = self.current_index

        nearest_ind = next_index_list[0]
        min_dist = self.graph.node_dist_mat[current_ind][next_index_list[0]]

        for next_ind in next_index_list[1:]:
            dist = self.graph.node_dist_mat[current_ind][next_ind]
            if dist < min_dist:
                min_dist = dist
                nearest_ind = next_ind

        return nearest_ind

    @staticmethod
    def cal_total_travel_distance(graph: VrptwGraph, travel_path):
        distance = 0
        current_ind = travel_path[0]
        for next_ind in travel_path[1:]:
            distance += graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind
        return distance
    
    ## updated travel time function (validated in vrptw_base)
    @staticmethod
    def cal_total_travel_time(graph: VrptwGraph, path, service_time_matrix, order_ids, minutes_per_km=1, get_idle_time=False, get_current_time=False):
        travel_time = 0
        current_time = 0
        vehicle_num = 0
        dist_dict = {}
        total_idle_time = 0

        for i in range(0, len(path)-1):
            # if path[i] == 0:
            #     if vehicle_num == 9:
            #         print('')
            if path[i] == 0 and i != 0:
                dist_dict[vehicle_num] = travel_time
                # test_dict[vehicle_num] = test_path
                current_time = 0
                vehicle_num += 1
                travel_time = 0
                # test_path = []
            dist = graph.node_dist_mat[path[i]][path[i+1]]*minutes_per_km
            if path[i] != 0:
                wait_time = max(graph.all_nodes[path[i+1]].ready_time - current_time - dist, 0)
                current_time += dist + wait_time
                total_idle_time += wait_time
            else:
                wait_time = 0
                current_time = max(graph.all_nodes[path[i+1]].ready_time, dist)

            service_time = VrptwGraph.get_service_time(path[i+1], service_time_matrix, 
                                                            current_time, order_ids)
            current_time += service_time
            travel_time += dist + wait_time + service_time
            
        dist_dict[vehicle_num] = travel_time

        total_travel_time = sum(dist_dict.values())

        if not get_idle_time and not get_current_time:
            return total_travel_time
        elif get_idle_time:
            return total_travel_time, total_idle_time
        elif get_current_time:
            return total_travel_time, current_time

    def try_insert_on_path(self, node_id, stop_event: Event):
        """
        尝试性地将node_id插入当前的travel_path中
        Experimentally insert the node_id into the current travel_path
        插入的位置不能违反载重，时间，行驶距离的限制
        The inserted position must not violate load, time and distance limits.
        如果有多个位置，则找出最优的位置
        If there is more than one position, find the optimal position.
        :param node_id:
        :return:
        """
        best_insert_index = None
        if not self.opt_time:
            best_distance = None
        elif self.opt_time:
            best_time = None

        for insert_index in range(len(self.travel_path)):

            if stop_event.is_set():
                # print('[try_insert_on_path]: receive stop event')
                return

            if self.graph.all_nodes[self.travel_path[insert_index]].is_depot:
                continue

            # MOD: If node is committed node, nothing can be inserted
            elif self.time_slice != 0:
                if self.travel_path[insert_index] in self.committed_path:
                    continue

            # 找出insert_index的前面的最近的depot
            # Find the most recent depot in front of insert_index
            front_depot_index = insert_index
            while front_depot_index >= 0 and not self.graph.all_nodes[self.travel_path[front_depot_index]].is_depot:
                front_depot_index -= 1
            front_depot_index = max(front_depot_index, 0)

            # check_ant从front_depot_index出发
            # check_ant from front_depot_index
            
          
            check_ant = Ant(path_handover=self.path_handover, time_slice=self.time_slice, graph=self.graph, 
                            source=self.source, service_time_matrix=self.service_time_matrix, 
                            order_ids=self.order_ids, start_index=self.travel_path[front_depot_index], opt_time=self.opt_time)


            # 让check_ant 走过 path中下标从front_depot_index开始到insert_index-1的点
            # Have check_ant walk past the point in path where the index starts at
            # the front_depot_index and ends at insert_index-1.
            for i in range(front_depot_index+1, insert_index):
                check_ant.move_to_next_index(self.travel_path[i])

            # 开始尝试性地对排序后的index_to_visit中的结点进行访问
            # Begin experimental access to nodes in the sorted index_to_visit
            # if check_ant.check_condition(node_id, self.service_time_matrix, self.order_ids):
            if check_ant.check_condition_check_ant(check_ant.travel_path, node_id, self.service_time_matrix, self.order_ids):
                check_ant.move_to_next_index(node_id)
            else:
                continue

            # 如果可以到node_id，则要保证vehicle可以行驶回到depot
            # If you can go to node_id, make sure the vehicle can travel back to depot.
            for next_ind in self.travel_path[insert_index:]:

                if stop_event.is_set():
                    return

                if check_ant.check_condition_check_ant(check_ant.travel_path, next_ind, self.service_time_matrix, self.order_ids):
                # if check_ant.check_condition(node_id, self.service_time_matrix, self.order_ids):
                    check_ant.move_to_next_index(next_ind)

                    # 如果回到了depot
                    # If you go back to the depot.
                    if self.graph.all_nodes[next_ind].is_depot:
                        temp_front_index = self.travel_path[insert_index-1]
                        temp_back_index = self.travel_path[insert_index]

                        if not self.opt_time:
                            check_ant_distance = self.total_travel_distance - self.graph.node_dist_mat[temp_front_index][
                                temp_back_index] + self.graph.node_dist_mat[temp_front_index][node_id] + \
                                self.graph.node_dist_mat[node_id][temp_back_index]

                            if best_distance is None or check_ant_distance < best_distance:
                                best_distance = check_ant_distance
                                best_insert_index = insert_index
                            break
                        elif self.opt_time:
                            check_ant_time= self.total_travel_time - self.graph.node_dist_mat[temp_front_index][
                                temp_back_index] + self.graph.node_dist_mat[temp_front_index][node_id] + \
                                self.graph.node_dist_mat[node_id][temp_back_index]

                            
                            if best_time is None or check_ant_time < best_time:
                                best_time = check_ant_time
                                best_insert_index = insert_index
                            break


                # 如果不可以回到depot，则返回上一层
                # If you can't go back to depot, go back to the previous level.
                else:
                    break

        return best_insert_index

    def insertion_procedure(self, stop_even: Event):
        """
        为每个未访问的结点尝试性地找到一个合适的位置，插入到当前的travel_path
        Try to find a suitable location for each unvisited node to insert into the current travel_path
        插入的位置不能违反载重，时间，行驶距离的限制
        The inserted position must not violate load, time and distance limits.
        :return:
        """

        if self.index_to_visit_empty():
            return

        success_to_insert = True
        # 直到未访问的结点中没有一个结点可以插入成功
        # Until none of the unvisited nodes can be inserted successfully
        while success_to_insert:

            success_to_insert = False
            # 获取未访问的结点 Get unvisited nodes
            ind_to_visit = np.array(copy.deepcopy(self.index_to_visit))

            # 获取为访问客户点的demand，降序排序
            # Get the demand for accessing client points, in descending order.
            demand = np.zeros(len(ind_to_visit))
            for i, ind in zip(range(len(ind_to_visit)), ind_to_visit):
                demand[i] = self.graph.all_nodes[ind].demand
            arg_ind = np.argsort(demand)[::-1]
            ind_to_visit = ind_to_visit[arg_ind]

            for node_id in ind_to_visit:
                if stop_even.is_set():
                    # print('[insertion_procedure]: receive stop event')
                    return

                best_insert_index = self.try_insert_on_path(node_id, stop_even)
                if best_insert_index is not None:
                    self.travel_path.insert(best_insert_index, node_id)
                    self.index_to_visit.remove(node_id)
                    success_to_insert = True

            del demand
            del ind_to_visit
        if self.index_to_visit_empty():
            print('[insertion_procedure]: success in insertion')

        if not self.opt_time:
            self.total_travel_distance = Ant.cal_total_travel_distance(self.graph, self.travel_path)
        elif self.opt_time:
            self.total_travel_time = Ant.cal_total_travel_time(self.graph, self.travel_path, 
                                                               self.service_time_matrix, self.order_ids)

    # MOD: path_handover and time_slice passed to use by check ant
    @staticmethod
    def local_search_once(graph: VrptwGraph, travel_path: list, travel_distance: float, i_start, stop_event: Event, path_handover, time_slice,
                          source, opt_time, service_time_matrix, order_ids):

        # 找出path中所有的depot的位置
        # Find the location of all depots in the path.
        depot_ind = []
        for ind in range(len(travel_path)):
            if graph.all_nodes[travel_path[ind]].is_depot:
                depot_ind.append(ind)

        # 将self.travel_path分成多段，每段以depot开始，以depot结束，称为route
        # Divide self.travel_path into multiple segments, each segment starts with a
        # depot and ends with a depot, called a route.
        for i in range(i_start, len(depot_ind)):
            for j in range(i + 1, len(depot_ind)):

                if stop_event.is_set():
                    return None, None, None

                for start_a in range(depot_ind[i - 1] + 1, depot_ind[i]):
                    for end_a in range(start_a, min(depot_ind[i], start_a + 6)):
                        for start_b in range(depot_ind[j - 1] + 1, depot_ind[j]):
                            for end_b in range(start_b, min(depot_ind[j], start_b + 6)):
                                if start_a == end_a and start_b == end_b:
                                    continue
                                new_path = []
                                new_path.extend(travel_path[:start_a])
                                new_path.extend(travel_path[start_b:end_b + 1])
                                new_path.extend(travel_path[end_a:start_b])
                                new_path.extend(travel_path[start_a:end_a])
                                new_path.extend(travel_path[end_b + 1:])

                                depot_before_start_a = depot_ind[i - 1]

                                depot_before_start_b = depot_ind[j - 1] + (end_b - start_b) - (end_a - start_a) + 1
                                if not graph.all_nodes[new_path[depot_before_start_b]].is_depot:
                                    raise RuntimeError('error')

                                # 判断发生改变的route a是否是feasible的
                                # Determine whether the changed route a is feasible or not.
                                success_route_a = False
                                check_ant = Ant(path_handover=path_handover, time_slice=time_slice, graph=graph, 
                                                    source=source, service_time_matrix=service_time_matrix, order_ids=order_ids,
                                                     start_index=new_path[depot_before_start_a], opt_time=opt_time)
                                for ind in new_path[depot_before_start_a + 1:]:
                                    if check_ant.check_condition(ind, service_time_matrix, order_ids, path_section=True):
                                        check_ant.move_to_next_index(ind)
                                        if graph.all_nodes[ind].is_depot:
                                            success_route_a = True
                                            break
                                    else:
                                        break

                                check_ant.clear()
                                del check_ant

                                # 判断发生改变的route b是否是feasible的
                                # Determine whether the changed route b is feasible or not.
                                success_route_b = False
                                check_ant = Ant(path_handover=path_handover, time_slice=time_slice, graph=graph, 
                                source=source, service_time_matrix=service_time_matrix, 
                                order_ids=order_ids, start_index=new_path[depot_before_start_b], opt_time=opt_time)
                                for ind in new_path[depot_before_start_b + 1:]:
                                    if check_ant.check_condition(ind, service_time_matrix, order_ids, path_section=True):
                                        check_ant.move_to_next_index(ind)
                                        if graph.all_nodes[ind].is_depot:
                                            success_route_b = True
                                            break
                                    else:
                                        break
                                check_ant.clear()
                                del check_ant

                                if success_route_a and success_route_b:
                                    if not opt_time:
                                        new_path_distance = Ant.cal_total_travel_distance(graph, new_path)
                                        if new_path_distance < travel_distance:
                                            # print('success to search')

                                            # 删除路径中连在一起的depot中的一个
                                            # Deletes one of the concatenated depots in the path.
                                            for temp_ind in range(1, len(new_path)):
                                                if graph.all_nodes[new_path[temp_ind]].is_depot and graph.all_nodes[
                                                        new_path[temp_ind - 1]].is_depot:
                                                    new_path.pop(temp_ind)
                                                    break
                                            return new_path, new_path_distance, i
                                    elif opt_time:
                                        new_path_time = Ant.cal_total_travel_time(graph, new_path, service_time_matrix, order_ids)
                                        if new_path_time < travel_distance:
                                            # print('success to search')

                                            # 删除路径中连在一起的depot中的一个
                                            # Deletes one of the concatenated depots in the path.
                                            for temp_ind in range(1, len(new_path)):
                                                if graph.all_nodes[new_path[temp_ind]].is_depot and graph.all_nodes[
                                                        new_path[temp_ind - 1]].is_depot:
                                                    new_path.pop(temp_ind)
                                                    break
                                            return new_path, new_path_time, i

                                else:
                                    new_path.clear()

        return None, None, None

    def local_search_procedure(self, stop_event: Event):
        """
        对当前的已经访问完graph中所有节点的travel_path使用cross进行局部搜索
        Use cross to perform a local search on travel_path for all nodes in the graph that are currently accessed.
        :return:
        """
        new_path = copy.deepcopy(self.travel_path)
        if self.opt_time:
            new_path_time = self.total_travel_time
        elif not self.opt_time:
            new_path_distance = self.total_travel_distance
        times = 10
        count = 0
        i_start = 1
        while count < times:
            if not self.opt_time:
                temp_path, temp_distance, temp_i = Ant.local_search_once(self.graph, new_path, new_path_distance, i_start,
                                                                        stop_event, self.path_handover, self.time_slice,
                                                                        self.source, self.service_time_matrix, self.order_ids)
            elif self.opt_time:
                temp_path, temp_time, temp_i = Ant.local_search_once(self.graph, new_path, new_path_time, i_start,
                                                    stop_event, self.path_handover, self.time_slice,
                                                    self.source, self.service_time_matrix, self.order_ids)
            if temp_path is not None:
                count += 1
                if not self.opt_time:
                    del new_path, new_path_distance
                    new_path = temp_path
                    new_path_distance = temp_distance
                if self.opt_time:
                    del new_path, new_path_time
                    new_path = temp_path
                    new_path_time = temp_time

                # 设置i_start Set i_start
                i_start = (i_start + 1) % (new_path.count(0)-1)
                i_start = max(i_start, 1)
            else:
                break

        self.travel_path = new_path
        if not self.opt_time:
            self.total_travel_distance = new_path_distance
            print('[local_search_procedure]: local search finished', self.total_travel_distance)
        elif self.opt_time:
            self.total_travel_time = new_path_time
            print('[local_search_procedure]: local search finished', self.total_travel_time)
