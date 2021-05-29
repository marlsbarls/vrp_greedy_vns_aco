import numpy as np
import random
from aco.vprtw_aco_figure import VrptwAcoFigure
from aco.vrptw_base import VrptwGraph, PathMessage
from aco.ant import Ant
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
    def __init__(self, graph: VrptwGraph, source, path_handover, path_map, folder_name_result, service_time_matrix, order_ids, ants_num=10, alpha=1,
                 beta=1, q0=0.1, time_slice=0, whether_or_not_to_show_figure=True):
        super()
        # graph 结点的位置、服务时间信息 Location of nodes, service hours information
        self.graph = graph
        # ants_num 蚂蚁数量 Number of ants
        self.ants_num = ants_num
        # vehicle_capacity 表示每辆车的最大载重 Indicates the maximum load per vehicle
        self.max_load = graph.vehicle_capacity
        # beta 启发性信息重要性 Importance of inspiring information
        self.beta = beta
        # q0 表示直接选择概率最大的下一点的概率 Indicates the probability of directly selecting the next point with the
        # highest probability.
        self.q0 = q0
        # best path
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

        # MOD: Added parameters
        self.alpha = alpha
        self.time_slice = time_slice
        self.path_handover = path_handover
        self.source = source
        self.path_map = path_map
        self.folder_name_result = folder_name_result

        # MOD: Marlene
        self.service_time_matrix = service_time_matrix
        self.order_ids = order_ids

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
                       beta: int, stop_event: Event):
        """
        按照指定的vehicle_num在地图上进行探索，所使用的vehicle num不能多于指定的数量，acs_time和acs_vehicle都会使用到这个方法
        对于acs_time来说，需要访问完所有的结点（路径是可行的），尽量找到travel distance更短的路径
        对于acs_vehicle来说，所使用的vehicle num会比当前所找到的best path所使用的车辆数少一辆，要使用更少的车辆，尽量去访问结点，如果访问完了所有的结点（路径是可行的），就将通知macs
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
        # 在new_active_ant中，最多可以使用vehicle_num个车，即最多可以包含vehicle_num+1个depot结点，由于出发结点用掉了一个，所以只剩下vehicle个depot
        # In new_active_ant, you can use up to vehicle_num cars, i.e. you can contain up to vehicle_num+1 depot nodes,
        # only vehicle depots are left since the departure node is used up.
        unused_depot_count = vehicle_num

        # 如果还有未访问的结点，并且还可以回到depot中
        # If there are still unvisited nodes and you can go back to the depot, the
        while not ant.index_to_visit_empty() and unused_depot_count > 0:
            if stop_event.is_set():
                # print('[new_active_ant]: receive stop event')
                return

            # 计算所有满足载重等限制的下一个结点
            # Calculate all the next nodes that meet the limits of load etc.
            next_index_meet_constrains = ant.cal_next_index_meet_constrains()

            # 如果没有满足限制的下一个结点，则回到depot中
            # If there is no next node that satisfies the limit, then back to depot
            if len(next_index_meet_constrains) == 0:
                ant.move_to_next_index(0)
                unused_depot_count -= 1
                continue

            # 开始计算满足限制的下一个结点，选择各个结点的概率
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

            # 按照概率直接选择closeness最大的结点
            # Directly select the node with the largest closeness according to probability.
            if np.random.rand() <= q0:
                max_prob_index = np.argmax(transition_prob)
                next_index = next_index_meet_constrains[max_prob_index]
            else:
                # 使用轮盘赌算法
                # Using the roulette wheel
                next_index = MultipleAntColonySystem.stochastic_accept(next_index_meet_constrains, transition_prob)

            # 更新信息素矩阵
            # Updating the pheromone matrix
            ant.graph.local_update_pheromone(ant.current_index, next_index)
            ant.move_to_next_index(next_index)

        # 如果走完所有的点了，需要回到depot
        # If you finish all the points, you need to go back to the depot.
        if ant.index_to_visit_empty():
            ant.graph.local_update_pheromone(ant.current_index, 0)
            ant.move_to_next_index(0)

        # 对未访问的点进行插入，保证path是可行的
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
                 service_time_matrix, order_ids):
        """
        对于acs_time来说，需要访问完所有的结点（路径是可行的），尽量找到travel distance更短的路径
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

        # 最多可以使用vehicle_num辆车，即在path中最多包含vehicle_num+1个depot中，找到路程最短的路径，
        # vehicle_num设置为与当前的best_path一致
        # Use at most vehicle_num, i.e. find the shortest path with at most vehicle_num+1 depot in the path,
        # and set vehicle_num to match the current best_path.
        print('[acs_time]: start, vehicle_num %d' % vehicle_num)
        # 初始化信息素矩阵
        # Initialized pheromone matrix
        global_best_path = None
        global_best_distance = None

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        while True:
            print('[acs_time]: new iteration')

            if stop_event.is_set():
                print('[acs_time]: receive stop event')
                return

            for k in range(ants_num):
                ant = Ant(path_handover, time_slice, service_time_matrix, order_ids, new_graph, 0)
                # MOD: Local Search set to False, argument alpha added
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, False,
                                          np.zeros(new_graph.node_num), q0, alpha, beta, stop_event)
                ants_thread.append(thread)
                ants.append(ant)

            # 这里可以使用result方法，等待线程跑完
            # Here you can use the result method and wait for the thread to run out.
            for thread in ants_thread:
                thread.result()

            ant_best_travel_distance = None
            ant_best_path = None
            # 判断蚂蚁找出来的路径是否是feasible的，并且比全局的路径要好
            # Determine if the path found by the ant is FEASIBLE and better than the global one
            for ant in ants:
                if stop_event.is_set():
                    print('[acs_time]: receive stop event')
                    return

                # 获取当前的best path
                # Get the current best path
                if not global_path_queue.empty():
                    info = global_path_queue.get()
                    while not global_path_queue.empty():
                        info = global_path_queue.get()
                    global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

                # 路径蚂蚁计算得到的最短路径
                # The shortest path calculated by the path ant.
                if ant.index_to_visit_empty() and (ant_best_travel_distance is None or ant.total_travel_distance <
                                                   ant_best_travel_distance):
                    ant_best_travel_distance = ant.total_travel_distance
                    ant_best_path = ant.travel_path

            # 在这里执行信息素的全局更新
            # Perform a global update of pheromones here
            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            # 向macs发送计算得到的当前的最佳路径
            # sends the calculated current best path to macs.
            if ant_best_travel_distance is not None and ant_best_travel_distance < global_best_distance:
                print('[acs_time]: ants\' local search found a improved feasible path, send path info to macs')
                path_found_queue.put(PathMessage(ant_best_path, ant_best_travel_distance))

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    # MOD: Arguments alpha, path_handover and time_slice added
    @staticmethod
    def acs_vehicle(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, alpha: int, beta: int,
                    global_path_queue: Queue, path_found_queue: Queue, stop_event: Event, path_handover, time_slice,
                    service_time_matrix, order_ids):
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
        # vehicle_num设置为比当前的best_path少一个
        # vehicle_num is set to one less than the current best_path.
        print('[acs_vehicle]: start, vehicle_num %d' % vehicle_num)
        global_best_path = None
        global_best_distance = None

        # 使用nearest_neighbor_heuristic算法初始化path 和distance
        # Initialize path and distance using the nearest_neighbor_heuristic algorithm.
        # MOD: Only execute NN for initial solution. Use path and distance from handover afterwards
        if time_slice == 0:
            current_path, current_path_distance, _ = new_graph.nearest_neighbor_heuristic(max_vehicle_num=vehicle_num)
            # 找出当前path中未访问的结点 Find the unvisited nodes in the current path.

            current_index_to_visit = list(range(new_graph.node_num))

            for ind in set(current_path):
                current_index_to_visit.remove(ind)

        else:
            f = open(path_handover, 'r')
            lines = f.readlines()
            current_path = ast.literal_eval(lines[0])
            current_path_distance = float(lines[1])
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
                ant = Ant(path_handover, time_slice, service_time_matrix, order_ids, new_graph, 0)
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, False, IN, q0,
                                          alpha, beta, stop_event)

                ants_thread.append(thread)
                ants.append(ant)

            # 这里可以使用result方法，等待线程跑完
            # Here you can use the result method and wait for the thread to run out.
            for thread in ants_thread:
                thread.result()

            for ant in ants:

                if stop_event.is_set():
                    print('[acs_vehicle]: receive stop event')
                    return

                IN[ant.index_to_visit] = IN[ant.index_to_visit]+1

                # 蚂蚁找出来的路径与current_path进行比较，是否能使用vehicle_num辆车访问到更多的结点
                # The path found by the ant is compared with current_path to see if more nodes can be accessed using
                # the vehicle_num car
                if len(ant.index_to_visit) < len(current_index_to_visit):
                    current_path = copy.deepcopy(ant.travel_path)
                    current_index_to_visit = copy.deepcopy(ant.index_to_visit)
                    current_path_distance = ant.total_travel_distance
                    # 并且将IN设置为0 #
                    # And set IN to 0
                    IN = np.zeros(new_graph.node_num)

                    # 如果这一条路径是feasible的话，就要发到macs_vrptw中 If this path is feasible, it should be sent to macs_vrptw
                    if ant.index_to_visit_empty():
                        print('[acs_vehicle]: found a feasible path, send path info to macs')
                        path_found_queue.put(PathMessage(ant.travel_path, ant.total_travel_distance))

            # 更新new_graph中的信息素，global
            # Update pheromones in new_graph, global
            new_graph.global_update_pheromone(current_path, current_path_distance)

            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                print('[acs_vehicle]: receive global path info')
                global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

            new_graph.global_update_pheromone(global_best_path, global_best_distance)

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

        # 是否要展示figure
        # Whether to show figure
        if self.whether_or_not_to_show_figure:
            # MOD: self.time_slice, self.time_slice, self.graph.all_nodes (instead of self.graph.nodes), self.path_map,
            # self.graph.file_path and self.folder_name_result added
            figure = VrptwAcoFigure(self.source, self.time_slice, self.graph.all_nodes, path_queue_for_figure,
                                    self.path_map, self.graph.file_path, self.folder_name_result)
            figure.run()
        multiple_ant_colony_system_thread.join()

        return

    # MOD: Argument total_given_time added
    def _multiple_ant_colony_system(self, path_queue_for_figure: MPQueue, total_given_time, file_to_write_path=None):
        print('_multiple_ant_colony_system')
        '''
        调用acs_time 和 acs_vehicle进行路径的探索 Call acs_time and acs_vehicle for path exploration.
        :param path_queue_for_figure:
        :return:
        '''
        if file_to_write_path is not None:
            file_to_write = open(file_to_write_path, 'w')
        else:
            file_to_write = None

        start_time_total = time.time()

        # 在这里需要两个队列，time_what_to_do、vehicle_what_to_do， 用来告诉acs_time、acs_vehicle这两个线程，当前的best
        # path是什么，或者让他们停止计算
        # Here we need two queues, time_what_to_do and vehicle_what_to_do, to tell acs_time and acs_vehicle what the
        # current best path is, or to stop them.
        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        # 另外的一个队列， path_found_queue就是接收acs_time 和acs_vehicle计算出来的比best path还要好的feasible path
        # Another queue, path_found_queue, receives the feasible path calculated by acs_time and acs_vehicle to be
        # better than the best path.
        path_found_queue = Queue()

        # 使用近邻点算法初始化
        # Initialization using the nearest neighbor algorithm

        # MOD: Only execute NN for initial solution. Use path and distance from handover afterwards
        if self.time_slice == 0:
            self.best_path, self.best_path_distance, self.best_vehicle_num = self.graph.nearest_neighbor_heuristic()

        else:
            f = open(self.path_handover, 'r')
            self.lines = f.readlines()
            self.best_path = ast.literal_eval(self.lines[0])
            self.best_path_distance = float(self.lines[1])
            self.best_vehicle_num = int(self.lines[2])
            f.close()

        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

        while True:
            print('[multiple_ant_colony_system]: new iteration')
            # MOD: start_time_found_improved_solution = time.time() removed, total time is significant time information

            # 当前best path的信息，放在queue中以通知acs_time和acs_vehicle当前的best_path是什么
            # Information about the current best path, put in queue to inform acs_time and acs_vehicle what the current
            # best_path is.
            global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
            global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))
            stop_event = Event()

            # acs_vehicle，尝试以self.best_vehicle_num-1辆车去探索，访问更多的结点
            # acs_vehicle tries to explore with self.best_vehicle_num-1 vehicles to access more nodes
            graph_for_acs_vehicle = self.graph.copy(self.graph.init_pheromone_val)

            # MOD: If time slice different from 0, perform global pheromone update for current best solution
            if self.time_slice != 0:
                graph_for_acs_vehicle.global_update_pheromone(self.best_path, self.best_path_distance)

            # MOD: Arguments self.alpha, self.path_handover and self.time_slice added
            acs_vehicle_thread = Thread(target=MultipleAntColonySystem.acs_vehicle,
                                        args=(graph_for_acs_vehicle, self.best_vehicle_num - 1, self.ants_num, self.q0,
                                              self.alpha, self.beta, global_path_to_acs_vehicle, path_found_queue,
                                              stop_event, self.path_handover, self.time_slice, self.service_time_matrix, self.order_ids))

            # acs_time 尝试以self.best_vehicle_num辆车去探索，找到更短的路径
            # acs_time tries to explore with self.best_vehicle_num vehicles to find a shorter path.
            graph_for_acs_time = self.graph.copy(self.graph.init_pheromone_val)

            # MOD: If time slice different from 0, perform global pheromone update for current best solution
            if self.time_slice != 0:
                graph_for_acs_time.global_update_pheromone(self.best_path, self.best_path_distance)

            # MOD: Arguments self.alpha, self.path_handover and self.time_slice added
            acs_time_thread = Thread(target=MultipleAntColonySystem.acs_time,
                                     args=(graph_for_acs_time, self.best_vehicle_num, self.ants_num, self.q0, self.beta,
                                           self.alpha, global_path_to_acs_time, path_found_queue, stop_event,
                                           self.path_handover, self.time_slice, self.service_time_matrix, self.order_ids))

            # 启动acs_vehicle_thread和acs_time_thread，当他们找到feasible、且是比best path好的路径时，就会发送到macs中来
            # Start acs_vehicle_thread and acs_time_thread and send them to macs when they find a FEASIBLE path that is
            # better than the best path.
            print('[macs]: start acs_vehicle and acs_time')
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():

                # 如果在指定时间内没有搜索到更好的结果，则退出程序
                # Quit the program if no better results are found within the specified time period.

                # MOD: total_given_time decisive for the termination of the program (before: start_time_found_improved_
                # solution)
                if time.time() - start_time_total > 60 * total_given_time:
                    stop_event.set()
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(file_to_write, 'time is up: cannot find a better solution in given time(%d minutes)' % total_given_time)
                    self.print_and_write_in_file(file_to_write, 'it takes %0.3f seconds from multiple_ant_colony_system running' % (time.time()-start_time_total))
                    self.print_and_write_in_file(file_to_write, 'the best path have found is:')
                    self.print_and_write_in_file(file_to_write, self.best_path)
                    self.print_and_write_in_file(file_to_write, 'best path distance is %f, best vehicle_num is %d' % (self.best_path_distance, self.best_vehicle_num))
                    self.print_and_write_in_file(file_to_write, '*' * 50)

                    # MOD: Save current best results to file and copy to handover file
                    result_tuple = (str(self.best_path), str(self.best_path_distance), str(self.best_vehicle_num),
                                    str([]))
                    f = open(self.path_handover, 'w')
                    separator = '\n'
                    f.write(separator.join(result_tuple))
                    f.close()
                    file_name_results = 'result-a_'+str(self.ants_num)+'-t_'+str(total_given_time)+'.txt'
                    path_results = os.path.join(self.folder_name_result, file_name_results.split('.')[0] + '_' + str(self.time_slice) + '.txt')
                    shutil.copy(self.path_handover, path_results)

                    # 传入None作为结束标志
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
                found_path, found_path_distance, found_path_used_vehicle_num = path_info.get_path_info()
                while not path_found_queue.empty():
                    path, distance, vehicle_num = path_found_queue.get().get_path_info()

                    if distance < found_path_distance:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                    if vehicle_num < found_path_used_vehicle_num:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                # 如果找到的路径（which is feasible）的距离更短，则更新当前的最佳path的信息
                # If the found path (which is feasible) is a shorter distance away, update the current information
                # about the best path.
                if found_path_distance < self.best_path_distance:

                    # 搜索到更好的结果，更新start_time
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

                    # 如果需要绘制图形，则要找到的best path发送给绘图程序
                    # If you need to draw a drawing, the best path to find is sent to the plotter
                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                    # 通知acs_vehicle和acs_time两个线程，当前找到的best_path和best_path_distance
                    # Notify the acs_vehicle and acs_time threads that the best_path and best_path_distance are
                    # currently found.
                    global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_distance))
                    global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_distance))

                # 如果，这两个线程找到的路径用的车辆更少了，就停止这两个线程，开始下一轮迭代
                # If, the two threads find the path using fewer vehicles, stop the two threads and start the next
                # iteration
                # 向acs_time和acs_vehicle中发送停止信息
                # Sends stop messages to aces_time and aces_vehicle.

                if found_path_used_vehicle_num < best_vehicle_num:

                    # 搜索到更好的结果，更新start_time
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

                    # 停止acs_time 和 acs_vehicle 两个线程
                    # Stop the acs_time and acs_vehicle threads.
                    print('[macs]: send stop info to acs_time and acs_vehicle')
                    # 通知acs_vehicle和acs_time两个线程，当前找到的best_path和best_path_distance
                    # Notify the acs_vehicle and acs_time threads that the best_path and best_path_distance are
                    # currently found.
                    stop_event.set()

    @staticmethod
    def print_and_write_in_file(file_to_write=None, message='default message'):
        if file_to_write is None:
            print(message)
        else:
            print(message)
            file_to_write.write(str(message)+'\n')
