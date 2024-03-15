from vns.src.algorithm.improved_vns import idle_time, total_cost
import numpy as np
import pandas as pd
import random
from aco.vprtw_aco_figure import VrptwAcoFigure
from aco.vrptw_base import VrptwGraph, PathMessage
from aco.ant import Ant
import vns.src.config.vns_config as cfg
import vns.src.config.preprocessing_config as prep_cfg
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import time
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
import ast
import shutil
import os


# MOD: Arguments source, path_handover, path_map, folder_name_result, alpha and time_slice passed from execution file
class MultipleAntColonySystem:
    def __init__(self, graph: VrptwGraph, source, path_handover, path_map, folder_name_result, result_df, parameter, 
                 exp_id, opt_time, service_time_matrix=None, order_ids=None, ants_num=10, alpha=1,
                 beta=1, q0=0.1, time_slice=0, whether_or_not_to_show_figure=True, min_per_km=1):
        super()
        # graph Location of nodes, service hours information
        self.graph = graph
        # ants_num Number of ants
        self.ants_num = ants_num
        # vehicle_capacity Indicates the maximum load per vehicle
        self.max_load = graph.vehicle_capacity
        # beta Importance of inspiring information
        self.beta = beta
        # q0 Indicates the probability of directly selecting the next point with the
        # highest probability.
        self.q0 = q0
        # best path
        # MOD: Marlene
        self.opt_time = opt_time
        if not self.opt_time:
            self.best_path_distance = None
        elif self.opt_time:
            self.best_path_travel_time = None
        self.best_path = None
        self.best_vehicle_num = None
        self.min_per_km = min_per_km
        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

        # MOD: Added parameters
        self.alpha = alpha
        self.time_slice = time_slice
        self.path_handover = path_handover
        self.source = source
        self.path_map = path_map
        self.folder_name_result = folder_name_result
 
        self.service_time_matrix = service_time_matrix
        self.order_ids = order_ids
        self.opt_time = opt_time
        # self.time_iterations = 0
        # self.vehicle_iterations = 0
        # self.macs_iterations = 0
        self.result_df = result_df
        self.parameter = parameter
        self.iteration_run_time = 0
        self.exp_id = exp_id
        # self.improvement_counter = improvement_counter
    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
        轮盘赌
        :param index_to_visit: a list of N index (list or tuple)
        :param transition_prob:
        :return: selected index
        """
        # calculate N and max fitness value
        N = len(index_to_visit)

        # normalize
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob/sum_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]

    @staticmethod
    def new_active_ant(ant: Ant, vehicle_num: int, local_search: bool, IN: np.numarray, q0: float, alpha: int,
                       beta: int, stop_event: Event, service_time_matrix, order_ids):
        """
        vehicle_num vehicle num，acs_time acs_vehicle
        acs_time，travel distance
        acs_vehicle vehicle num best path macs
        :param ant:
        :param vehicle_num:
        :param local_search:
        :param IN:
        :param q0:
        :param alpha:
        :param beta:
        :param stop_event:
        :return:
        """

        # print('[new_active_ant]: start, start_index %d' % ant.travel_path[0])
        # new_active_ant，vehicle_num，vehicle_num+1 depot, vehicle depot
        # In new_active_ant, you can use up to vehicle_num cars, i.e. you can contain up to vehicle_num+1 depot nodes,
        # only vehicle depots are left since the departure node is used up.
        unused_depot_count = vehicle_num

        # depot
        # If there are still unvisited nodes and you can go back to the depot, the
        while not ant.index_to_visit_empty() and unused_depot_count > 0:
            if stop_event.is_set():
                # print('[new_active_ant]: receive stop event')
                return

            # Calculate all the next nodes that meet the limits of load etc.
            next_index_meet_constrains = ant.cal_next_index_meet_constrains()

            # depot
            # If there is no next node that satisfies the limit, then back to depot
            if len(next_index_meet_constrains) == 0:
                ant.move_to_next_index(0)
                unused_depot_count -= 1
                continue

            # Start calculating the next node that meets the limit, selecting the probability of each node
            length = len(next_index_meet_constrains)
            ready_time = np.zeros(length)
            due_time = np.zeros(length)

            for i in range(length):
                ready_time[i] = ant.graph.all_nodes[next_index_meet_constrains[i]].ready_time
                due_time[i] = ant.graph.all_nodes[next_index_meet_constrains[i]].due_time

            delivery_time = np.maximum(
                ant.vehicle_travel_time + ant.graph.node_dist_mat[ant.current_index][next_index_meet_constrains],
                ready_time)
            delta_time = delivery_time - ant.vehicle_travel_time
            distance = delta_time * (due_time - ant.vehicle_travel_time)

            distance = np.maximum(1.0, distance-IN[next_index_meet_constrains])
            closeness = 1/distance

            np.seterr(divide='ignore', invalid='ignore')

            # MOD: alpha added to calculation of transition probability
            transition_prob = np.power(ant.graph.pheromone_mat[ant.current_index][next_index_meet_constrains], alpha) *\
                              np.power(closeness, beta)
            transition_prob = transition_prob / np.sum(transition_prob)

            # closeness
            # Directly select the node with the largest closeness according to probability.
            if np.random.rand() <= q0:
                max_prob_index = np.argmax(transition_prob)
                next_index = next_index_meet_constrains[max_prob_index]
            else:
                # Using the roulette wheel
                next_index = MultipleAntColonySystem.stochastic_accept(next_index_meet_constrains, transition_prob)

            # Updating the pheromone matrix
            ant.graph.local_update_pheromone(ant.current_index, next_index)
            ant.move_to_next_index(next_index)

        # depot
        # If you finish all the points, you need to go back to the depot.
        if ant.index_to_visit_empty():
            ant.graph.local_update_pheromone(ant.current_index, 0)
            ant.move_to_next_index(0)

        # path
        # Insertion of unvisited points to ensure that the path is feasible
        ant.insertion_procedure(stop_event)

        # ant.index_to_visit_empty()==True就是feasible的意思
        # ant.index_to_visit_empty() == True means FEASIBLE!
        if local_search is True and ant.index_to_visit_empty():
            ant.local_search_procedure(stop_event)

    # MOD: Arguments alpha, path_handover and time_slice added
    @staticmethod
    def acs_time(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, alpha: int, beta: int,
                 global_path_queue: Queue, path_found_queue: Queue, stop_event: Event, path_handover, time_slice, 
                 source, opt_time:bool, service_time_matrix=None, order_ids=None):
        """
        acs_time，travel distance
        For acs_time, you need to visit all the nodes (where the path is possible) and try to find a path with a shorter
        travel distance.
        :param new_graph:
        :param vehicle_num:
        :param ants_num:
        :param q0:
        :param alpha:
        :param beta:
        :param global_path_queue:
        :param path_found_queue:
        :param stop_event:
        :param path_handover:
        :param time_slice
        :return:
        """

        # vehicle_num，path vehicle_num+1 depot
        # vehicle_num best_path一
        # Use at most vehicle_num, i.e. find the shortest path with at most vehicle_num+1 depot in the path,
        # and set vehicle_num to match the current best_path.
        print('[acs_time]: start, vehicle_num %d' % vehicle_num)
        # Initialized pheromone matrix
        global_best_path = None
        if not opt_time:
            global_best_distance = None
        elif opt_time:
            global_best_time = None

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        while True:
            print('[acs_time]: new iteration')

            if stop_event.is_set():
                print('[acs_time]: receive stop event')
                return

            for k in range(ants_num):
                
                ant = Ant(path_handover=path_handover, time_slice=time_slice, 
                            graph=new_graph, source=source, service_time_matrix=service_time_matrix, 
                            order_ids=order_ids, start_index=0, opt_time=opt_time) 
             
              
                # MOD: Local Search set to False, argument alpha added
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, False,
                                          np.zeros(new_graph.node_num), q0, alpha, beta, stop_event, service_time_matrix, order_ids)
                ants_thread.append(thread)
                ants.append(ant)

            # result
            # Here you can use the result method and wait for the thread to run out.
            for thread in ants_thread:
                thread.result()

            if not opt_time:
                ant_best_travel_distance = None
            elif opt_time:
                ant_best_travel_time = None
            ant_best_path = None
            # feasible
            # Determine if the path found by the ant is FEASIBLE and better than the global one
            for ant in ants:
                if stop_event.is_set():
                    print('[acs_time]: receive stop event')
                    return

                # best path
                # Get the current best path
                if not opt_time:
                    if not global_path_queue.empty():
                        info = global_path_queue.get()
                        while not global_path_queue.empty():
                            info = global_path_queue.get()
                        global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

                # The shortest path calculated by the path ant.
                    if ant.index_to_visit_empty() and (ant_best_travel_distance is None or ant.total_travel_distance <
                                                    ant_best_travel_distance):
                        ant_best_travel_distance = ant.total_travel_distance
                        ant_best_path = ant.travel_path
                elif opt_time:
                    if not global_path_queue.empty():
                        info = global_path_queue.get()
                        while not global_path_queue.empty():
                            info = global_path_queue.get()
                        global_best_path, global_best_time, global_used_vehicle_num = info.get_path_info()

                    if ant.index_to_visit_empty() and (ant_best_travel_time is None or ant.total_travel_time <
                                                    ant_best_travel_time):
                        ant_best_travel_time = ant.total_travel_time
                        ant_best_path = ant.travel_path

            # Perform a global update of pheromones here
            if not opt_time:
                new_graph.global_update_pheromone(global_best_path, global_best_distance)
                
            # macs
            # sends the calculated current best path to macs.
                if ant_best_travel_distance is not None and ant_best_travel_distance < global_best_distance:
                    print('[acs_time]: ants\' local search found a improved feasible path, send path info to macs')
                    path_found_queue.put(PathMessage(ant_best_path, ant_best_travel_distance))

            elif opt_time:
                new_graph.global_update_pheromone(global_best_path, global_best_time)

                if ant_best_travel_time is not None and ant_best_travel_time < global_best_time:
                    print('[acs_time]: ants\' local search found a improved feasible path, send path info to macs')
                    path_found_queue.put(PathMessage(ant_best_path, ant_best_travel_time))

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    # MOD: Arguments alpha, path_handover and time_slice added
    @staticmethod
    def acs_vehicle(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, alpha: int, beta: int,
                    global_path_queue: Queue, path_found_queue: Queue, stop_event: Event, path_handover, time_slice,
                    source, opt_time: bool, service_time_matrix=None, order_ids=None):
        """
        对于acs_vehicle来说，所使用的vehicle num会比当前所找到的best path所使用的车辆数少一辆，要使用更少的车辆，尽量去访问结点，
        如果访问完了所有的结点（路径是可行的），就将通知macs
        For acs_vehicle, the vehicle num used will be one less vehicle than the number of vehicles used by the best
        path currently found.
        :param new_graph:
        :param vehicle_num:
        :param ants_num:
        :param q0:
        :param alpha:
        :param beta:
        :param global_path_queue:
        :param path_found_queue:
        :param stop_event:
        :param path_handover:
        :param time_slice
        :return:
        """
        # vehicle_num best_path
        # vehicle_num is set to one less than the current best_path.
        print('[acs_vehicle]: start, vehicle_num %d' % vehicle_num)
        global_best_path = None
        if not opt_time:
            global_best_distance = None
        elif opt_time:
            global_best_time = None

        # nearest_neighbor_heuristic path distance
        # Initialize path and distance using the nearest_neighbor_heuristic algorithm.
        # MOD: Only execute NN for initial solution. Use path and distance from handover afterwards
        if time_slice == 0:
            if not opt_time:
                current_path, current_path_distance, _ = new_graph.nearest_neighbor_heuristic(max_vehicle_num=vehicle_num)
            # path Find the unvisited nodes in the current path.
            elif opt_time:
                current_path, current_path_time, _ = new_graph.nearest_neighbor_heuristic(max_vehicle_num=vehicle_num)

            current_index_to_visit = list(range(new_graph.node_num))

            for ind in set(current_path):
                current_index_to_visit.remove(ind)

        else:
            f = open(path_handover, 'r')
            lines = f.readlines()
            current_path = ast.literal_eval(lines[0])
            if not opt_time:
                current_path_distance = float(lines[1])
            elif opt_time:
                current_path_time = float(lines[1])
            f.close()

            current_index_to_visit = current_path.copy()
            current_index_to_visit.sort()
            current_index_to_visit = [elem for elem in current_index_to_visit if elem != 0]
            current_index_to_visit.insert(0, 0)

            for ind in set(current_path):
                current_index_to_visit.remove(ind)

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        IN = np.zeros(new_graph.node_num)
        while True:
            print('[acs_vehicle]: new iteration')

            if stop_event.is_set():
                print('[acs_vehicle]: receive stop event')
                return

            # MOD: Argument alpha added
            for k in range(ants_num):
                ant = Ant(path_handover=path_handover, time_slice=time_slice, 
                            graph=new_graph, source=source, service_time_matrix=service_time_matrix, 
                            order_ids=order_ids, start_index=0, opt_time=opt_time)
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, False, IN, q0,
                                          alpha, beta, stop_event, service_time_matrix, order_ids)

                ants_thread.append(thread)
                ants.append(ant)

            # result
            # Here you can use the result method and wait for the thread to run out.
            for thread in ants_thread:
                thread.result()

            for ant in ants:

                if stop_event.is_set():
                    print('[acs_vehicle]: receive stop event')
                    return

                IN[ant.index_to_visit] = IN[ant.index_to_visit]+1

                # current_path，vehicle_num
                # The path found by the ant is compared with current_path to see if more nodes can be accessed using
                # the vehicle_num car
                if len(ant.index_to_visit) < len(current_index_to_visit):
                    current_path = copy.deepcopy(ant.travel_path)
                    current_index_to_visit = copy.deepcopy(ant.index_to_visit)
                    if not opt_time:
                        current_path_distance = ant.total_travel_distance
                    if opt_time:
                        current_path_time = ant.total_travel_time
                    # And set IN to 0
                    IN = np.zeros(new_graph.node_num)

                    # feasible，macs_vrptw If this path is feasible, it should be sent to macs_vrptw
                    if ant.index_to_visit_empty():
                        print('[acs_vehicle]: found a feasible path, send path info to macs')
                        if not opt_time:
                            path_found_queue.put(PathMessage(ant.travel_path, ant.total_travel_distance))
                        elif opt_time:
                            path_found_queue.put(PathMessage(ant.travel_path, ant.total_travel_time))

            # new_graph，global
            # Update pheromones in new_graph, global
            if not opt_time:
                new_graph.global_update_pheromone(current_path, current_path_distance)
            if opt_time:
                new_graph.global_update_pheromone(current_path, current_path_time)

            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                print('[acs_vehicle]: receive global path info')
                if not opt_time:
                    global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()
                if opt_time:
                    global_best_path, global_best_time, global_used_vehicle_num = info.get_path_info()

            if not opt_time:
                new_graph.global_update_pheromone(global_best_path, global_best_distance)
            if opt_time:
                new_graph.global_update_pheromone(global_best_path, global_best_time)

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    # MOD: Argument total_given_time passed from execution file
    def run_multiple_ant_colony_system(self, total_given_time, file_to_write_path=None):
        print('run_multiple_ant_colony_system')
        '''
        开启另外的线程来跑multiple_ant_colony_system， 使用主线程来绘图
        Open another thread to run multiple_ant_colony_system, use the main thread to plot
        :return:
        '''
        path_queue_for_figure = MPQueue()
        # MOD: Argument total_given_time added
        multiple_ant_colony_system_thread = Process(target=self._multiple_ant_colony_system, args=(
            path_queue_for_figure, total_given_time, file_to_write_path))
        multiple_ant_colony_system_thread.start()

        # Whether to show figure
        if self.whether_or_not_to_show_figure:
            # MOD: self.time_slice, self.time_slice, self.graph.all_nodes (instead of self.graph.nodes), self.path_map,
            # self.graph.file_path and self.folder_name_result added
            figure = VrptwAcoFigure(source=self.source, time_slice=self.time_slice, nodes=self.graph.all_nodes, 
                                    path_queue=path_queue_for_figure, path_map=self.path_map, 
                                    file_path=self.graph.file_path, folder_name_result=self.folder_name_result, 
                                    opt_time=self.opt_time)
            figure.run()
        multiple_ant_colony_system_thread.join()

        return
    
    def total_cost_vns(self, tours, travel_time_mat, service_time, order_ids):
        total_cost = 0
        for tour in tours:
            current_time = 0
            travel_time = 0
            for i in range(len(tour) - 1):
                traffic_phase = "off_peak" if current_time < prep_cfg.traffic_times["phase_transition"][
                "from_shift_start"] else "phase_transition" if current_time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"
                if i == 0:
                    travel_time += travel_time_mat[tour[i]][tour[i + 1]]
                    current_time += max(travel_time_mat[tour[i]][tour[i + 1]], self.graph.all_nodes[tour[i+1]].ready_time)
                else:
                    travel_time += max(service_time[order_ids[tour[i]]+":" +
                                            traffic_phase] + travel_time_mat[tour[i]][tour[i + 1]], 
                                            self.graph.all_nodes[tour[i+1]].ready_time-current_time)
                    current_time += max(service_time[order_ids[tour[i]]+":" +
                                            traffic_phase] + travel_time_mat[tour[i]][tour[i + 1]], 
                                            self.graph.all_nodes[tour[i+1]].ready_time-current_time)
            tour_cost = (travel_time*cfg.cost_per_minute) + \
                cfg.cost_per_driver
            total_cost += tour_cost
        return total_cost

    def _calculate_costs_vns(self):
        # adjust tour format to vns 
        sub_tours = [[0,0]] * self.best_vehicle_num
        current_tour = 0
        counter = 0
        for order in self.best_path:
            if order != 0:
                sub_tours[current_tour] = sub_tours[current_tour][:-1]+[order]+[sub_tours[current_tour][-1]]
            if order == 0 and counter != 0:
                current_tour +=1
            counter += 1
        
        # calculate total costs
        # total_costs = self.total_cost_original(sub_tours, self.graph.node_dist_mat, self.service_time_matrix, self.order_ids)
        total_cost_exp, list_dist = self.total_cost_vns(sub_tours, self.graph.node_dist_mat, self.service_time_matrix, self.order_ids)
        
        return total_cost_exp, list_dist

    def _calculate_costs_new(self):
        total_cost = 0
        travel_time = 0
        current_time = 0
        vehicle_num = 0
        dist_dict = {}

        for i in range(0, len(self.best_path)-1):
            if self.best_path[i] == 0 and i != 0:
                dist_dict[vehicle_num] = travel_time
                current_time = 0
                vehicle_num += 1
                travel_time = 0
            dist = self.graph.node_dist_mat[self.best_path[i]][self.best_path[i+1]]*self.min_per_km
            # no wait time if depot 
            if self.best_path[i] != 0:
                wait_time = max(self.graph.all_nodes[self.best_path[i+1]].ready_time - current_time - dist, 0)
                current_time += dist + wait_time
            else:
                wait_time = 0
                current_time = max(self.graph.all_nodes[self.best_path[i+1]].ready_time, dist)

            service_time = VrptwGraph.get_service_time(self.best_path[i+1], self.service_time_matrix, 
                                                            current_time, self.order_ids)
            current_time += service_time
            travel_time += dist + wait_time + service_time
            
        dist_dict[vehicle_num] = travel_time
        num_drivers = len(dist_dict) 

        total_travel_time = sum(dist_dict.values())

        total_cost = total_travel_time*cfg.cost_per_minute + num_drivers*cfg.cost_per_driver

        return total_cost


    # MOD: Argument total_given_time added
    def _multiple_ant_colony_system(self, path_queue_for_figure: MPQueue, total_given_time, file_to_write_path=None):
        print('_multiple_ant_colony_system')
        '''
        acs_time acs_vehicle Call acs_time and acs_vehicle for path exploration.
        :param path_queue_for_figure:
        :return:
        '''
        if file_to_write_path is not None:
            file_to_write = open(file_to_write_path, 'w')
        else:
            file_to_write = None

        start_time_total = time.time()

        # Here we need two queues, time_what_to_do and vehicle_what_to_do, to tell acs_time and acs_vehicle what the
        # current best path is, or to stop them.
        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        # Another queue, path_found_queue, receives the feasible path calculated by acs_time and acs_vehicle to be
        # better than the best path.
        path_found_queue = Queue()

        # Initialization using the nearest neighbor algorithm
        # MOD: Only execute NN for initial solution. Use path and distance from handover afterwards
        if self.time_slice == 0:
            if not self.opt_time:
                self.best_path, self.best_path_distance, self.best_vehicle_num = self.graph.nearest_neighbor_heuristic()
            if self.opt_time:
                self.best_path, self.best_path_travel_time, self.best_vehicle_num = self.graph.nearest_neighbor_heuristic()

        else:
            f = open(self.path_handover, 'r')
            self.lines = f.readlines()
            self.best_path = ast.literal_eval(self.lines[0])
            self.best_vehicle_num = int(self.lines[2])
            if not self.opt_time:
                self.best_path_distance = float(self.lines[1])
            elif self.opt_time:
                self.best_path_travel_time = float(self.lines[1])
            f.close()

        if not self.opt_time:
            path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))
        elif self.opt_time:
            path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_travel_time))

        while True:
            print('[multiple_ant_colony_system]: new iteration')
            # MOD: start_time_found_improved_solution = time.time() removed, total time is significant time information

            # Information about the current best path, put in queue to inform acs_time and acs_vehicle what the current
            # best_path is.
            # MOD: 
            if not self.opt_time:
                global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
                global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))
            elif self.opt_time:
                global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_travel_time))
                global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_travel_time))
            stop_event = Event()

            # acs_vehicle tries to explore with self.best_vehicle_num-1 vehicles to access more nodes
            graph_for_acs_vehicle = self.graph.copy(self.graph.init_pheromone_val)

            # MOD: If time slice different from 0, perform global pheromone update for current best solution
            if self.time_slice != 0:
                if not self.opt_time:
                    graph_for_acs_vehicle.global_update_pheromone(self.best_path, self.best_path_distance)
                elif self.opt_time:
                    graph_for_acs_vehicle.global_update_pheromone(self.best_path, self.best_path_travel_time)

            # MOD: Arguments self.alpha, self.path_handover and self.time_slice added
            acs_vehicle_thread = Thread(target=MultipleAntColonySystem.acs_vehicle,
                                    args=(graph_for_acs_vehicle, self.best_vehicle_num - 1, self.ants_num, self.q0,
                                            self.alpha, self.beta, global_path_to_acs_vehicle, path_found_queue,
                                            stop_event, self.path_handover, self.time_slice, self.source, self.opt_time, 
                                            self.service_time_matrix, 
                                            self.order_ids))

            # acs_time tries to explore with self.best_vehicle_num vehicles to find a shorter path.
            graph_for_acs_time = self.graph.copy(self.graph.init_pheromone_val)

            # MOD: If time slice different from 0, perform global pheromone update for current best solution
            if self.time_slice != 0:
                if not self.opt_time:
                    graph_for_acs_vehicle.global_update_pheromone(self.best_path, self.best_path_distance)
                elif self.opt_time:
                    graph_for_acs_vehicle.global_update_pheromone(self.best_path, self.best_path_travel_time)

            

            # MOD: Arguments self.alpha, self.path_handover and self.time_slice added
            acs_time_thread = Thread(target=MultipleAntColonySystem.acs_time,
                                        args=(graph_for_acs_time, self.best_vehicle_num, self.ants_num, 
                                            self.q0, self.beta, self.alpha, global_path_to_acs_time, 
                                            path_found_queue, stop_event, self.path_handover, self.time_slice, 
                                            self.source, self.opt_time, self.service_time_matrix, self.order_ids))

            # Start acs_vehicle_thread and acs_time_thread and send them to macs when they find a FEASIBLE path that is
            # better than the best path.
            print('[macs]: start acs_vehicle and acs_time')
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():
                # Quit the program if no better results are found within the specified time period.
                # MOD: total_given_time decisive for the termination of the program (before: start_time_found_improved_
                # solution)
                if time.time() - start_time_total > 60 * total_given_time:
                    self.iteration_run_time = (time.time() - start_time_total)/60
                    stop_event.set()
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(file_to_write, 'time is up: cannot find a better solution in given time(%d minutes)' % total_given_time)
                    self.print_and_write_in_file(file_to_write, 'it takes %0.3f seconds from multiple_ant_colony_system running' % (time.time()-start_time_total))
                    self.print_and_write_in_file(file_to_write, 'the best path have found is:')
                    self.print_and_write_in_file(file_to_write, self.best_path)
                    total_cost = self._calculate_costs_new()
                    if not self.opt_time:
                        travel_time, idle_time = Ant.cal_total_travel_time(self.graph, self.best_path, self.service_time_matrix, self.order_ids, get_idle_time=True)
                        self.print_and_write_in_file(file_to_write, 'best path distance is %f, best vehicle_num is %d' % (self.best_path_distance, self.best_vehicle_num))
                        result_tuple = (str(self.best_path), str(self.best_path_distance), str(self.best_vehicle_num),
                                        str([]))
                        result_tuple_costs = ('path: '+str(self.best_path), 'distance: '+str(self.best_path_distance), 
                                              'vehicle_num: '+str(self.best_vehicle_num), 'costs: '+str(total_cost), 
                                              'travel time: '+str(travel_time), str([]))
                        
                        # tour_feasible = self.check_feasibility(self.best_path)                        
                        if self.time_slice == 0:
                            self.result_df.loc[self.time_slice] = [
                                self.parameter, cfg.cost_per_driver, cfg.cost_per_hour, total_cost, idle_time, self.best_path_distance, 
                                self.best_vehicle_num, self.iteration_run_time, self.best_path]
                            self.result_df.to_csv(os.path.join(self.folder_name_result, 'result.csv'), mode='a')
                        else:
                            result_df = pd.read_csv(os.path.join(self.folder_name_result, 'result.csv'), usecols=[
                                'parameters', 'cost per driver', 'cost per hour', 'final_cost', 'idle_time', 'tour_length', 'vehicle_number', 'runtime', 'final_tour'])
                            result_df.loc[len(result_df.index)] = [
                                self.parameter, cfg.cost_per_driver, cfg.cost_per_hour, total_cost, idle_time, self.best_path_distance, 
                                self.best_vehicle_num, self.iteration_run_time, self.best_path]
                            result_df.to_csv(os.path.join(self.folder_name_result, 'result.csv'))
                    elif self.opt_time:
                        travel_time, idle_time = Ant.cal_total_travel_time(self.graph, self.best_path, self.service_time_matrix, self.order_ids, get_idle_time=True)
                        distance = Ant.cal_total_travel_distance(self.graph, self.best_path)
                        cost_test = self.best_path_travel_time * cfg.cost_per_minute + self.best_vehicle_num * cfg.cost_per_driver
                        self.print_and_write_in_file(file_to_write, 'best path travel time is %f, best vehicle_num is %d' % (self.best_path_travel_time, self.best_vehicle_num))
                        result_tuple = (str(self.best_path), str(travel_time), str(self.best_vehicle_num),
                                        str([]))
                        result_tuple_costs = ('path: '+str(self.best_path), 'distance: '+str(distance), 
                                              'vehicle_num: '+str(self.best_vehicle_num), 'costs: '+str(total_cost), 
                                              'travel time: '+str(travel_time), str([]))

                        travel_time_compariosn = f'travel time from function: {travel_time}, travel time from algo {self.best_path_travel_time}'                        
                        # tour_feasible = self.check_feasibility(self.best_path)                        
                        if self.time_slice == 0:
                            self.result_df.loc[self.time_slice] = [
                                self.parameter, cfg.cost_per_driver, cfg.cost_per_hour, travel_time_compariosn, total_cost, idle_time, distance, 
                                self.best_vehicle_num, self.iteration_run_time, self.best_path]
                            self.result_df.to_csv(os.path.join(self.folder_name_result, 'result.csv'), mode='a')
                        else:
                            result_df = pd.read_csv(os.path.join(self.folder_name_result, 'result.csv'), usecols=[
                                'parameters', 'cost per driver', 'cost per hour', 'travel_time_comp', 'final_cost', 'idle_time', 'tour_length', 'vehicle_number', 'runtime', 'final_tour'])
                            result_df.loc[len(result_df.index)] = [
                                self.parameter, cfg.cost_per_driver, cfg.cost_per_hour, travel_time_compariosn, total_cost, idle_time, distance, 
                                self.best_vehicle_num, self.iteration_run_time, self.best_path]
                            result_df.to_csv(os.path.join(self.folder_name_result, 'result.csv'))
                        print(f'time slice: {self.time_slice}')
                        print(travel_time_compariosn)
                        
                    # MOD: Save current best results to file and copy to handover file
                    f = open(self.path_handover, 'w')
                    separator = '\n'
                    f.write(separator.join(result_tuple))
                    f.close()
                    if not self.opt_time:
                        optimization = 'distance'
                    elif self.opt_time:
                        optimization = 'time'
                    file_name_results = 'result-a_'+str(self.ants_num)+'-t_'+str(total_given_time)+'-o_'+str(optimization)+'.txt'
                    path_results = os.path.join(self.folder_name_result, file_name_results.split('.')[0] + '_' + str(self.time_slice) + '.txt')
                    # shutil.copy(self.path_handover, path_results)
                    f = open(path_results, 'w')
                    separator = '\n'
                    f.write(separator.join(result_tuple_costs))
                    f.close()

                    self.print_and_write_in_file(file_to_write, '*' * 50)

                    # Pass in None as an end flag.
                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(None, None))

                    if file_to_write is not None:
                        file_to_write.flush()
                        file_to_write.close()

                    return 

                if path_found_queue.empty():
                    continue

                path_info = path_found_queue.get()
                print('[macs]: receive found path info')
                if not self.opt_time:
                    found_path, found_path_distance, found_path_used_vehicle_num = path_info.get_path_info()
                    while not path_found_queue.empty():
                        path, distance, vehicle_num = path_found_queue.get().get_path_info()

                        if distance < found_path_distance:
                            found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                        if vehicle_num < found_path_used_vehicle_num:
                            found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                    # If the found path (which is feasible) is a shorter distance away, update the current information
                    # about the best path.
                    if found_path_distance < self.best_path_distance:
                        # Search for better results, update start_time
                        # MOD: start_time_found_improved_solution = time.time() removed, total time is significant time
                        # information
                        self.print_and_write_in_file(file_to_write, '*' * 50)
                        self.print_and_write_in_file(file_to_write, '[macs]: distance of found path (%f) better than best path\'s (%f)' % (found_path_distance, self.best_path_distance))
                        # MOD: Difference to previous solution added
                        self.print_and_write_in_file(file_to_write, 'difference to previous solution: {:.20f}'.format(self.best_path_distance-found_path_distance))
                        self.print_and_write_in_file(file_to_write, 'it takes %0.3f second from multiple_ant_colony_system running' % (time.time()-start_time_total))
                        self.print_and_write_in_file(file_to_write, '*' * 50)
                        if file_to_write is not None:
                            file_to_write.flush()

                        self.best_path = found_path
                        self.best_vehicle_num = found_path_used_vehicle_num
                        self.best_path_distance = found_path_distance

                        # If you need to draw a drawing, the best path to find is sent to the plotter
                        if self.whether_or_not_to_show_figure:
                            path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                        # Notify the acs_vehicle and acs_time threads that the best_path and best_path_distance are
                        # currently found.
                        global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
                        global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))

                elif self.opt_time:
                    found_path, found_path_time, found_path_used_vehicle_num = path_info.get_path_info()
                    while not path_found_queue.empty():
                        path, travel_time, vehicle_num = path_found_queue.get().get_path_info()

                        if travel_time < found_path_time:
                            found_path, found_path_time, found_path_used_vehicle_num = path, travel_time, vehicle_num

                        if vehicle_num < found_path_used_vehicle_num:
                            found_path, found_path_time, found_path_used_vehicle_num = path, travel_time, vehicle_num

                    # If the found path (which is feasible) is a shorter distance away, update the current information
                    # about the best path.
                    if found_path_time < self.best_path_travel_time:
                        # Search for better results, update start_time
                        # MOD: start_time_found_improved_solution = time.time() removed, total time is significant time
                        # information
                        self.print_and_write_in_file(file_to_write, '*' * 50)
                        self.print_and_write_in_file(file_to_write, '[macs]: travel time of found path (%f) better than best path\'s (%f)' % (found_path_time, self.best_path_travel_time))
                        # self.improvement_counter +=1
                        # MOD: Difference to previous solution added
                        self.print_and_write_in_file(file_to_write, 'difference to previous solution: {:.20f}'.format(self.best_path_travel_time-found_path_time))
                        self.print_and_write_in_file(file_to_write, 'it takes %0.3f second from multiple_ant_colony_system running' % (time.time()-start_time_total))
                        self.print_and_write_in_file(file_to_write, '*' * 50)
                        if file_to_write is not None:
                            file_to_write.flush()

                        self.best_path = found_path
                        self.best_vehicle_num = found_path_used_vehicle_num
                        self.best_path_travel_time = found_path_time

                        # If you need to draw a drawing, the best path to find is sent to the plotter
                        if self.whether_or_not_to_show_figure:
                            path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_travel_time))

                        # Notify the acs_vehicle and acs_time threads that the best_path and best_path_distance are
                        # currently found.
                        global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_travel_time))
                        global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_travel_time))


                # If, the two threads find the path using fewer vehicles, stop the two threads and start the next
                # iteration
                # Sends stop messages to aces_time and aces_vehicle.
                if not self.opt_time:
                    if found_path_used_vehicle_num < best_vehicle_num:
                        # Search for better results, update start_time
                        # MOD: start_time_found_improved_solution = time.time() removed, total time is significant time
                        # information
                        self.print_and_write_in_file(file_to_write, '*' * 50)
                        self.print_and_write_in_file(file_to_write, '[macs]: vehicle num of found path (%d) better than best path\'s (%d), found path distance is %f' % (found_path_used_vehicle_num, best_vehicle_num, found_path_distance))
                        self.print_and_write_in_file(file_to_write, 'it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))
                        self.print_and_write_in_file(file_to_write, '*' * 50)
                        if file_to_write is not None:
                            file_to_write.flush()

                        self.best_path = found_path
                        self.best_vehicle_num = found_path_used_vehicle_num
                        self.best_path_distance = found_path_distance

                        if self.whether_or_not_to_show_figure:
                            path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                        # Stop the acs_time and acs_vehicle threads.
                        print('[macs]: send stop info to acs_time and acs_vehicle')
                        # Notify the acs_vehicle and acs_time threads that the best_path and best_path_distance are
                        # currently found.
                        stop_event.set()
                elif self.opt_time:
                    if found_path_used_vehicle_num < best_vehicle_num:
                        # Search for better results, update start_time
                        # MOD: start_time_found_improved_solution = time.time() removed, total time is significant time
                        # information
                        self.print_and_write_in_file(file_to_write, '*' * 50)
                        self.print_and_write_in_file(file_to_write, '[macs]: vehicle num of found path (%d) better than best path\'s (%d), found path travel time is %f' % (found_path_used_vehicle_num, best_vehicle_num, found_path_time))
                        self.print_and_write_in_file(file_to_write, 'it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))
                        self.print_and_write_in_file(file_to_write, '*' * 50)
                        if file_to_write is not None:
                            file_to_write.flush()

                        self.best_path = found_path
                        self.best_vehicle_num = found_path_used_vehicle_num
                        self.best_path_travel_time = found_path_time

                        if self.whether_or_not_to_show_figure:
                            path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_travel_time))
                            # path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_travel_time, self.graph, self.service_time_matrix, self.order_ids))

                        # Stop the acs_time and acs_vehicle threads.
                        print('[macs]: send stop info to acs_time and acs_vehicle')
                        # Notify the acs_vehicle and acs_time threads that the best_path and best_path_time are
                        # currently found.
                        stop_event.set()

    @staticmethod
    def print_and_write_in_file(file_to_write=None, message='default message'):
        if file_to_write is None:
            print(message)
        else:
            print(message)
            file_to_write.write(str(message)+'\n')

    def check_feasibility(self, path, minutes_per_km = 1):
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


