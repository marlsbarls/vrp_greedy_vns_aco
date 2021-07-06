#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import copy
import time
import vns.src.config.vns_config as cfg
import math
from math import radians, cos, sin, asin, sqrt
import random as rd
import matplotlib.pyplot as plt
from pathlib import Path
rd.seed(0)

dir_name = os.path.dirname(os.path.realpath('__file__'))


##############################################################
# Basic Functions

def get_customer_size(current_tour):
    counter = 1
    for tour in current_tour:
        for order in tour:
            if order != 0:
                counter += 1
    return counter


def get_distance_matrix(cust_size, xcoor, ycoor):
    dist_matrix = np.zeros((cust_size + 1, cust_size + 1))
    for i in range(cust_size):
        for j in range(cust_size):
            dist_matrix[i][j] = np.sqrt(
                (xcoor[i] - xcoor[j]) ** 2 + (ycoor[i] - ycoor[j]) ** 2)
    return dist_matrix


def total_distance(tours, distance):
    total_distance = 0
    for tour in tours:
        tour_distance = 0
        for i in range(len(tour) - 1):
            tour_distance += distance[tour[i]][tour[i + 1]]
        total_distance += tour_distance
    return total_distance


def time_checker(tour, travel_time, service_time, ready_time, due_time):
    time = 0
    counter = 0
    for i in range(1, len(tour)):
        time = max(time, ready_time[tour[i - 1]]) + \
            service_time[tour[i - 1]] + travel_time[tour[i - 1]][tour[i]]
        if time <= due_time[tour[i]]:
            counter += 1
        else:
            break
    if counter == len(tour) - 1:
        return True
    else:
        return False


def begin_time(tour, travel_time, service_time, ready_time):
    begin_service_time = [0]
    time = 0
    for i in range(1, len(tour)):
        time = max(time, ready_time[tour[i - 1]]) + \
            service_time[tour[i - 1]] + travel_time[tour[i - 1]][tour[i]]
        begin = max(time, ready_time[tour[i]])
        begin_service_time.append(begin)

    return begin_service_time


def total_time(tours, travel_time, service_time, ready_time):
    total_time = 0
    for tour in tours:
        tour_time = begin_time(tour, travel_time, service_time, ready_time)[-1]
        total_time += tour_time
    return total_time


def check_sequence(sequence, tour):
    i = tour.index(sequence[0])
    if tour[i+1] == sequence[1]:
        return True
    else:
        return False


def simulated_annealing_improvement(temp, improvement, sim_anneal_accept):
    q = np.random.uniform(0, 1)
    p_annahme = math.exp(round(improvement/temp, 5))
    if(q <= p_annahme):
        sim_anneal_accept.append(1)
        return True
    elif(q > p_annahme):
        sim_anneal_accept.append(0)
        return False


def update_temperature(temp):
    return 0.9*temp


def visibility_index(sub_tour, visibility):
    vis_set = set(visibility)
    matching_indices = [i for i, item in enumerate(
        sub_tour) if item in vis_set]
    if(len(matching_indices) > 0 and matching_indices[0] != 0):
        # There are orders in the current route that are already done
        index = matching_indices[0]
    else:
        # No order has been done yet, optimization may change route for all customers
        index = 1

    return index


'''Exporting functions'''


def create_planning_df(tours, current_order_df, distance_matrix, service_time, ready_time, planning_df=None, interval=0):
    if planning_df is None:
        planning_df = current_order_df[[
            'CUST_NO', 'SERVICETIME']].copy(deep=True)
        planning_df['VISITED'] = False
        planning_df['SCHEDULED_TIME'] = np.NaN
        planning_df['SCHEDULED_TOUR'] = np.NaN

    tour_id = 0
    for tour in tours:
        if(interval == 0):
            time = 0
        else:
            ready_time_temp = ready_time[int(tour[1])]
            time = math.ceil(ready_time_temp/interval) * interval
        for i in range(1, len(tour)-1):
            order_i = tour[i]
            order_i_1 = tour[i-1]

            if(order_i_1 != 0 and order_i != 0):
                if(ready_time[order_i_1] <= (time + service_time[order_i_1])):
                    time = time + service_time[order_i_1] + \
                        distance_matrix[order_i_1][order_i]
                else:
                    time = ready_time[order_i_1] + \
                        distance_matrix[order_i_1][order_i]
            else:
                if(ready_time[order_i_1] <= time):
                    time = time + distance_matrix[order_i_1][int(order_i)]

            planning_df.at[order_i, 'SCHEDULED_TIME'] = time
            planning_df.at[order_i, 'SCHEDULED_TOUR'] = tour_id

        tour_id += 1

    return planning_df


def two_opt_move(tour, distance, travel_time, service_time, ready_time, due_time, temperture, visibility):
    best_move = {
        "node1": -1,
        "node2": -1,
        "gain": 0,
        "tour": []
    }

    if len(tour)-visibility >= 5:
        for i in range(visibility, len(tour) - 3):
            for j in range(i + 2, len(tour) - 1):
                new_tour = tour[0:i + 1] + \
                    tour[i + 1:j + 1][::-1] + tour[j + 1:]
                if time_checker(new_tour, travel_time, service_time, ready_time, due_time):
                    imp = total_distance([tour], distance) - \
                        total_distance([new_tour], distance)
                    if (imp > best_move["gain"]):
                        best_move = {
                            "node1": i,
                            "node2": j,
                            "gain": round(imp, 4),
                            "tour": new_tour
                        }

    return best_move


def two_opt_search(sub_tour, distance, travel_time, service_time, ready_time, due_time, temperture, visibility):
    Best_Imp = float('-inf')
    Tour = []
    Improved_Route = []

    for i in range(len(sub_tour)):
        best_move = two_opt_move(
            sub_tour[i], distance, travel_time, service_time, ready_time, due_time, temperture, visibility_index(sub_tour[i], visibility))

        if best_move["node1"] != -1:
            Best_Imp += best_move["gain"]
            Tour.append(i)
            Improved_Route.append(best_move["tour"])

    return Tour, Improved_Route, Best_Imp


#  3-opt:

def exchange(tour, opt_case, a, c, e, distance):
    """
    Reconnects the tour given three edges to swap.

    """
    b, d, f = a + 1, c + 1, e + 1

    base = total_distance([tour], distance)

    if opt_case == 1:
        # 2-opt (a, e) [d, c] (b, f)
        sol = tour[:a + 1] + tour[e:d - 1:-1] + tour[c:b - 1:-1] + tour[f:]
    elif opt_case == 2:
        # 2-opt [a, b] (c, e) (d, f)
        sol = tour[:a + 1] + tour[b:c + 1] + tour[e:d - 1:-1] + tour[f:]
    elif opt_case == 3:
        # 2-opt (a, c) (b, d) [e, f]
        sol = tour[:a + 1] + tour[c:b - 1:-1] + tour[d:e + 1] + tour[f:]
    elif opt_case == 4:
        # 3-opt (a, d) (e, c) (b, f)
        sol = tour[:a + 1] + tour[d:e + 1] + tour[c:b - 1:-1] + tour[f:]
    elif opt_case == 5:
        # 3-opt (a, d) (e, b) (c, f)
        sol = tour[:a + 1] + tour[d:e + 1] + tour[b:c + 1] + tour[f:]
    elif opt_case == 6:
        # 3-opt (a, e) (d, b) (c, f)
        sol = tour[:a + 1] + tour[e:d - 1:-1] + tour[b:c + 1] + tour[f:]
    elif opt_case == 7:
        # 3-opt (a, c) (b, e) (d, f)
        sol = tour[:a + 1] + tour[c:b - 1:-1] + tour[e:d - 1:-1] + tour[f:]

    gain = total_distance([sol], distance)
    return sol, (base - gain)


def three_opt_search_best_improvement(tour, distance, travel_time, service_time, ready_time, due_time, temperature, visibility=0):
    opt_case = [4, 5, 6, 7]
    best_move = {
        "a": -1,
        "c": -1,
        "e": -1,
        "opt_case": -1,
        "gain": 0,
        "tour": []
    }

    n = len(tour)
    if(len(tour)-visibility >= 6):
        for a in range(visibility, n-5):
            for c in range(a+2, n-3):
                for e in range(c+2, n-1):
                    for case in opt_case:
                        new_tour, gain_expected = exchange(
                            tour, case, a, c, e, distance)
                        if(gain_expected > best_move["gain"]):
                            if(time_checker(new_tour, travel_time, service_time, ready_time, due_time)):
                                best_move = {
                                    "a": a,
                                    "c": c,
                                    "e": e,
                                    "opt_case": case,
                                    "gain": gain_expected,
                                    "tour": new_tour
                                }
    return best_move


def three_opt_search(sub_tour, distance, travel_time, service_time, ready_time, due_time, temperture, visibility):
    Best_Imp = float('-inf')
    Tour = []
    Improved_Route = []

    for i in range(len(sub_tour)):
        best_move = three_opt_search_best_improvement(
            sub_tour[i], distance, travel_time, service_time, ready_time, due_time, temperture, visibility_index(sub_tour[i], visibility))

        if best_move["opt_case"] != -1:
            Best_Imp += best_move["gain"]
            Tour.append(i)
            Improved_Route.append(best_move["tour"])

    return Tour, Improved_Route, Best_Imp


def two_optstar(tour1, tour2, distance, travel_time, service_time, ready_time, due_time, demand, capacity, visibility_t1, visibility_t2):
    best_imp = float('-inf')
    position1 = -1
    position2 = -1

    base = total_distance([tour1], distance) + \
        total_distance([tour2], distance)

    for i in range(visibility_t1, len(tour1) - 1):
        for j in range(visibility_t2, len(tour2) - 1):
            new_tour1 = tour1[:i + 1] + tour2[j + 1:]
            new_tour2 = tour2[:j + 1] + tour1[i + 1:]
            tour1_new_demand = sum(demand[new_tour1])
            tour2_new_demand = sum(demand[new_tour2])

            if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):

                time_check_2opts1 = time_checker(
                    new_tour1, travel_time, service_time, ready_time, due_time)
                time_check_2opts2 = time_checker(
                    new_tour2, travel_time, service_time, ready_time, due_time)

                if time_check_2opts1 and time_check_2opts2:
                    twoopts_cost = base - \
                        (total_distance([new_tour1], distance) +
                         total_distance([new_tour2], distance))

                    if twoopts_cost > best_imp:
                        best_imp = twoopts_cost
                        position1 = i
                        position2 = j

    return position1, position2, best_imp


def two_optstar_search(sub_tour, distance, travel_time, service_time, ready_time, due_time, demand, capacity, visibility):
    Best_Imp = float("-inf")
    Tour = []
    Improved_Route = []

    for t1 in range(len(sub_tour) - 1):
        for t2 in range(t1 + 1, len(sub_tour)):
            [position1, position2, imp] = two_optstar(sub_tour[t1], sub_tour[t2], distance, travel_time, service_time,
                                                      ready_time, due_time, demand, capacity, visibility_index(sub_tour[t1], visibility), visibility_index(sub_tour[t2], visibility))

            if imp > Best_Imp:
                Tour = [t1, t2]
                New_tour1 = sub_tour[t1][:position1 +
                                         1] + sub_tour[t2][position2 + 1:]
                New_tour2 = sub_tour[t2][:position2 +
                                         1] + sub_tour[t1][position1 + 1:]
                Improved_Route = [New_tour1, New_tour2]

                Best_Imp = imp

    return Tour, Improved_Route, Best_Imp


# # 3. Metaheuristic: VNS

# TODO: change access of service_time


def local_search(Sub_tour, distance, travel_time, service_time, ready_time, due_time, demand, temperature, visibility, Local_Tour, capacity, performance_counter, sim_anneal_accept, improvement_per_iteration):
    [Tour_two_opt, Improved_Route_two_opt, Improvement_two_opt] = two_opt_search(copy.deepcopy(
        Sub_tour), distance, travel_time, service_time, ready_time, due_time, temperature, visibility)

    [Tour_two_opt_star, Improved_Route_two_opt_star, Improvement_two_opt_star] = two_optstar_search(copy.deepcopy(
        Sub_tour), distance, travel_time, service_time, ready_time, due_time, demand, capacity, visibility)

    [Tour_three_opt, Improved_Route_three_opt, Improvement_three_opt] = three_opt_search(copy.deepcopy(
        Sub_tour), distance, travel_time, service_time, ready_time, due_time, temperature, visibility)

    if(Improvement_two_opt_star >= Improvement_two_opt and Improved_Route_two_opt_star >= Improved_Route_three_opt):
        Tour = Tour_two_opt_star
        Improved_Route = Improved_Route_two_opt_star
        performance_counter['two-opt-star'] += 1
    elif(Improvement_two_opt >= Improvement_two_opt_star and Improved_Route_two_opt >= Improved_Route_three_opt):
        Tour = Tour_two_opt
        Improved_Route = Improved_Route_two_opt
        performance_counter['two-opt'] += 1
    elif(Improvement_three_opt >= Improvement_two_opt and Improved_Route_three_opt >= Improved_Route_two_opt_star):
        Tour = Tour_three_opt
        Improved_Route = Improved_Route_three_opt
        performance_counter['three-opt'] += 1
    else:
        Tour = Tour_two_opt
        Improved_Route = Improved_Route_two_opt

    TMPSub_tour = copy.deepcopy(Sub_tour)
    for t in range(len(Tour)):
        TMPSub_tour[Tour[t]] = Improved_Route[t]

    CurrImprovement = total_distance(Local_Tour,  distance) - \
        total_distance(TMPSub_tour, distance)

    # Simulated annealing
    if CurrImprovement > 0 or simulated_annealing_improvement(temperature, CurrImprovement, sim_anneal_accept):
        improvement_per_iteration.append(CurrImprovement)
        Sub_tour = TMPSub_tour
    else:
        improvement_per_iteration.append(0)
        Sub_tour = Local_Tour

    return Sub_tour


def sort_tours(Sub_tour):
    sorted_tours = sorted(Sub_tour, key=len)
    divider = len(sorted_tours[int(len(sorted_tours)/2)])
    short_tours = []
    long_tours = []
    i = 0
    for tour in Sub_tour:
        if len(tour) < divider:
            short_tours.append(tour)
        else:
            long_tours.append(tour)
    return short_tours, long_tours


def remove_empty_tours(Sub_tour):
    Sub_tour = [Sub_tour[i] for i in range(len(Sub_tour)) if
                len(Sub_tour[i]) > 2]  # Remove empty tours
    return Sub_tour


def shaking(Input_tour, travel_time, service_time, ready_time, due_time, demand, neighbourhood, capacity, visibility):

    Sub_tour = copy.deepcopy(Input_tour)
    shaking_start = time.time()

    for i in range(0, neighbourhood):
        n = len(Sub_tour) - 1
        prob = rd.uniform(0, 1)
        if prob <= cfg.shaking["INSERT"]["PROBABILITY"] and n > 0:  # Insert
            prob_lengeth = rd.uniform(0, 1)
            bool_len = False
            if (prob_lengeth >= cfg.shaking["SORT_LEN"]["PROBABILITY"]):
                bool_len = True
                short_tours, long_tours = sort_tours(Sub_tour)

            while True:
                while True:

                    if(bool_len and len(short_tours) > 0 and len(long_tours) > 0):
                        Tour_short = rd.randint(0, len(short_tours)-1)
                        Tour_long = rd.randint(0, len(long_tours)-1)
                        Tour1 = [i for i in range(
                            len(Sub_tour)) if short_tours[Tour_short] == Sub_tour[i]][0]
                        Tour2 = [i for i in range(
                            len(Sub_tour)) if long_tours[Tour_long] == Sub_tour[i]][0]
                        if len(Sub_tour[Tour1]) >= 3 and len(Sub_tour[Tour2]) >= 4 and Tour1 != Tour2:
                            break
                    else:
                        # Tour1 = rd.randint(0, n - 1)
                        # Tour2 = rd.randint(Tour1 + 1, n)
                        Tour1 = rd.randint(0, n)
                        Tour2 = rd.randint(0, n)
                        if len(Sub_tour[Tour1]) >= 3 and len(Sub_tour[Tour2]) >= 4 and Tour1 != Tour2:
                            break

                if(visibility_index(Sub_tour[Tour1], visibility) <= len(Sub_tour[Tour1]) - 2 and visibility_index(Sub_tour[Tour2], visibility) <= len(Sub_tour[Tour2]) - 3):
                    Node11 = rd.randint(
                        visibility_index(Sub_tour[Tour1], visibility), len(Sub_tour[Tour1]) - 2)
                    Node12 = rd.randint(Node11, len(Sub_tour[Tour1]) - 1)
                    Node21 = rd.randint(
                        visibility_index(Sub_tour[Tour2], visibility), len(Sub_tour[Tour2]) - 3)
                    Node22 = Node21 + 1

                    New_tour1 = Sub_tour[Tour1][:Node11] + \
                        Sub_tour[Tour1][Node12:]
                    New_tour2 = Sub_tour[Tour2][:Node21+1] + \
                        Sub_tour[Tour1][Node11:Node12] + \
                        Sub_tour[Tour2][Node22:]

                    time_check1 = time_checker(
                        New_tour1, travel_time, service_time, ready_time, due_time)
                    time_check2 = time_checker(
                        New_tour2, travel_time, service_time, ready_time, due_time)
                    new_tour1_demand = sum(demand[New_tour1])
                    new_tour2_demand = sum(demand[New_tour2])

                    shaking_end = time.time()

                    if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                        Sub_tour[Tour1] = New_tour1
                        Sub_tour[Tour2] = New_tour2
                        break
                    elif shaking_end - shaking_start > 2:
                        break

        elif n > 0:
            prob2 = rd.uniform(0, 1)
            if prob2 <= cfg.shaking["CROSS"]["PROBABILITY"]:  # CROSS
                while True:
                    while True:
                        Tour1 = rd.randint(0, n)
                        Tour2 = rd.randint(0, n)
                        if len(Sub_tour[Tour1]) >= 4 and len(Sub_tour[Tour2]) >= 4:
                            break

                    if(Tour1 == Tour2 and visibility_index(Sub_tour[Tour1], visibility) <= len(Sub_tour[Tour1]) - 3):
                        Node11 = rd.randint(visibility_index(
                            Sub_tour[Tour1], visibility), len(Sub_tour[Tour1]) - 3)
                        Node12 = rd.randint(Node11, len(Sub_tour[Tour1]) - 2)
                        Node2 = rd.randint(Node12, len(Sub_tour[Tour2]) - 2)

                        New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour1][Node12+1:Node2 +
                                                                               1] + Sub_tour[Tour1][Node11:Node12+1] + Sub_tour[Tour1][Node2+1:]

                        time_check1 = time_checker(
                            New_tour1, travel_time, service_time, ready_time, due_time)

                        new_tour1_demand = sum(demand[New_tour1])

                        shaking_end = time.time()

                        if time_check1 and new_tour1_demand <= capacity:
                            Sub_tour[Tour1] = New_tour1
                            break
                        elif shaking_end - shaking_start > 2:
                            break
                    elif(visibility_index(Sub_tour[Tour1], visibility) <= len(Sub_tour[Tour1])-3 and visibility_index(Sub_tour[Tour2], visibility) <= len(Sub_tour[Tour2]) - 3):
                        Node11 = rd.randint(visibility_index(
                            Sub_tour[Tour1], visibility), len(Sub_tour[Tour1]) - 3)
                        Node12 = rd.randint(Node11, len(Sub_tour[Tour1]) - 2)
                        Node21 = rd.randint(visibility_index(
                            Sub_tour[Tour2], visibility), len(Sub_tour[Tour2]) - 3)
                        Node22 = rd.randint(Node21, len(Sub_tour[Tour2]) - 2)

                        New_tour1 = Sub_tour[Tour1][:Node11] + \
                            Sub_tour[Tour2][Node21:Node22 + 1] + \
                            Sub_tour[Tour1][Node12 + 1:]
                        New_tour2 = Sub_tour[Tour2][:Node21] + \
                            Sub_tour[Tour1][Node11:Node12 + 1] + \
                            Sub_tour[Tour2][Node22 + 1:]

                        time_check1 = time_checker(
                            New_tour1, travel_time, service_time, ready_time, due_time)
                        time_check2 = time_checker(
                            New_tour2, travel_time, service_time, ready_time, due_time)
                        new_tour1_demand = sum(demand[New_tour1])
                        new_tour2_demand = sum(demand[New_tour2])

                        shaking_end = time.time()

                        if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                            Sub_tour[Tour1] = New_tour1
                            Sub_tour[Tour2] = New_tour2
                            break
                        elif shaking_end - shaking_start > 2:
                            break
            # ICROSS
            elif (prob2 <= cfg.shaking["CROSS"]["PROBABILITY"]+cfg.shaking["ICROSS"]["PROBABILITY"]):

                while True:
                    while True:
                        Tour1 = rd.randint(0, n)
                        Tour2 = rd.randint(0, n)
                        if len(Sub_tour[Tour1]) >= 4 and len(Sub_tour[Tour2]) >= 4 and Tour1 != Tour2:
                            break

                    if(visibility_index(Sub_tour[Tour1], visibility) <= len(Sub_tour[Tour1]) - 3 and visibility_index(Sub_tour[Tour2], visibility) <= len(Sub_tour[Tour2]) - 3):
                        Node11 = rd.randint(
                            visibility_index(Sub_tour[Tour1], visibility), len(Sub_tour[Tour1]) - 3)
                        Node12 = rd.randint(Node11, len(Sub_tour[Tour1]) - 2)
                        Node21 = rd.randint(
                            visibility_index(Sub_tour[Tour2], visibility), len(Sub_tour[Tour2]) - 3)
                        Node22 = rd.randint(Node21, len(Sub_tour[Tour2]) - 2)

                        New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1][::-1] + Sub_tour[Tour1][
                            Node12 + 1:]
                        New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1][::-1] + Sub_tour[Tour2][
                            Node22 + 1:]

                        time_check1 = time_checker(
                            New_tour1, travel_time, service_time, ready_time, due_time)
                        time_check2 = time_checker(
                            New_tour2, travel_time, service_time, ready_time, due_time)
                        new_tour1_demand = sum(demand[New_tour1])
                        new_tour2_demand = sum(demand[New_tour2])

                        shaking_end = time.time()

                        if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                            Sub_tour[Tour1] = New_tour1
                            Sub_tour[Tour2] = New_tour2
                            break
                        elif shaking_end - shaking_start > 2:
                            break
                    else:
                        break

        Sub_tour = remove_empty_tours(Sub_tour)

    return Sub_tour


def run_vns(file, ini_tour, all_order_df, capacity, visibility, is_exp, **exp_params):
    '''---------------------------------------------------------------
    Experiment section: Set parameters, if current run is an experiment
    ------------------------------------------------------------------'''

    # TODO: Expand for all parameters

    if is_exp:
        cfg.shaking["INSERT"]["PROBABILITY"] = exp_params['insert_prob'] if 'insert_prob' in exp_params else cfg.shaking["INSERT"]["PROBABILITY"]
        cfg.shaking["INSERT"]["SORT_LEN"] = exp_params['sort_prob'] if 'sort_prob' in exp_params else cfg.shaking["SORT_LEN"]["PROBABILITY"]
        cfg.vns["InitialTemperature"] = exp_params['temp'] if 'temp' in exp_params else cfg.vns["InitialTemperature"]

    '''---------------------------------------------------------------
    Prepare relevant internal data:
        - build features from passed dataframe
        - calculate travel time matrix
        - initialize simulated annealing
        - initialize algorithm analysis
    ---------------------------------------------------------------'''

    # prepare data
    xcoor = all_order_df['XCOORD'].to_numpy()
    ycoor = all_order_df['YCOORD'].to_numpy()
    demand = all_order_df['DEMAND'].to_numpy()
    readytime = all_order_df['READYTIME'].to_numpy()
    # duetime = all_order_df['DUETIME'].to_numpy()
    duetime = all_order_df['DUETIME'].to_numpy()
    servicetime = all_order_df['SERVICETIME'].to_numpy()
    # cust_size = get_customer_size(ini_tour)
    cust_size = len(all_order_df)
    print("Cust size: ", cust_size)
    capacity = capacity
    # Calculate Travel Time Matrix
    distance_matrix = get_distance_matrix(cust_size, xcoor, ycoor)

    # set initial temperture for simulated annealing
    sa_temp = cfg.vns['InitialTemperature']

    # set stores for analysis
    DIST = []
    NO_VEHICLE = []
    RUN_TIME = []
    analysis_convergence_total = []
    analysis_convergence_shaking = []
    analysis_convergence_local = []
    analysis_restarts = []
    analysis_last_update = 0
    analysis_iterations_total = 0
    analysis_simulated_annealing_acceptance = []
    analysis_improvement_per_iteration = []
    initial_solution = copy.deepcopy(ini_tour)
    initial_distance = total_distance(ini_tour, distance_matrix)
    performance_counter = {
        'two-opt': 0,
        'three-opt': 0,
        'two-opt-star': 0
    }

    '''---------------------------------------------------------------
                         Actual Algortihm
    ---------------------------------------------------------------'''
    if(not is_exp):
        target_folder = os.path.join(
            dir_name, "data", "results_optimization", file)
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        outputfile = open(os.path.join(
            target_folder, 'output.txt'), 'w')
        outputfile.write(f'File: {file} Customer_Size:{cust_size} \n')
        outputfile.write(
            f'Iteration:0 Distance initial tour:{total_distance(ini_tour, distance_matrix)} Number of routes initial tour {len(ini_tour)}  \n')

    print('Start', file, cust_size)
    print('0', total_distance(ini_tour, distance_matrix), len(ini_tour))

    # Initial solution
    # Format: List<List<Integer>>, where Integer stands for Customer Number
    Sub_tour_VNS = copy.deepcopy(ini_tour)
    # MAINCODE
    for counter in range(cfg.vns["MaxRestarts"]):

        Sub_tour_local_search = copy.deepcopy(Sub_tour_VNS)

        # Define the set of the neighborhood structure for local research
        Shaking_Neighbor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        time_start = time.time()

        NO_IMP = 0
        k = 0
        iterations = 0
        STOP = False
        while not STOP:
            iterations += 1
            # Shaking process: in the 𝑘th neighborhood structure of current solution 𝑥 randomly generates a 𝑥s solution and the evaluationfunction of 𝑥 is 𝑓(𝑥)
            Sub_tour_shaking = shaking(Sub_tour_local_search, distance_matrix, servicetime, readytime, duetime, demand,
                                       Shaking_Neighbor[k], capacity, visibility)
            # Local search process: for 𝑥𝑠 to adopt somelocal search algorithm to solve local optimalsolution 𝑥𝑙, evaluation function of 𝑥𝑙 is 𝑓(𝑥𝑙)

            global_cost = total_distance(
                Sub_tour_VNS, distance_matrix)
            shake_cost = total_distance(
                Sub_tour_shaking, distance_matrix)
            old_cost = total_distance(Sub_tour_local_search, distance_matrix)

            Sub_tour_local_search = local_search(copy.deepcopy(Sub_tour_shaking), distance_matrix, distance_matrix,
                                                 servicetime, readytime, duetime, demand, sa_temp, visibility, Sub_tour_local_search, capacity, performance_counter, analysis_simulated_annealing_acceptance, analysis_improvement_per_iteration)

            new_cost = total_distance(Sub_tour_local_search,
                                      distance_matrix)

            print('Global cost: ', global_cost)
            print('Local cost: ', old_cost)
            print('Shaking cost: ', shake_cost)
            print('New cost: ', new_cost)
            print("Temp: ", sa_temp)

            # 𝑓(𝑥𝑙)<𝑓(𝑥)
            if (new_cost < global_cost):
                print('Global solution updated')
                Sub_tour_VNS = copy.deepcopy(Sub_tour_local_search)
                analysis_last_update = analysis_iterations_total + iterations
                k = 1
                NO_IMP = 0
            # 𝑓(𝑥𝑙)<𝑓(𝑥)
            else:
                NO_IMP += 1

                if iterations > cfg.vns['MaxIterations'] or NO_IMP > cfg.vns['MaxIterations_NoImp']:
                    STOP = True
                    analysis_restarts.append(iterations if len(
                        analysis_restarts) == 0 else analysis_restarts[-1] + iterations)
                else:
                    if k >= len(Shaking_Neighbor) - 1:
                        k = 0
                        STOP = False
                    else:
                        k += 1
                        STOP = False
            Sub_tour_VNS = [Sub_tour_VNS[i] for i in range(len(Sub_tour_VNS)) if
                            len(Sub_tour_VNS[i]) > 2]  # Remove empty tours
            analysis_convergence_total.append(total_distance(
                Sub_tour_VNS, distance_matrix))
            analysis_convergence_local.append(new_cost)
            analysis_convergence_shaking.append(shake_cost)
            if iterations % 5 == 0:
                sa_temp = update_temperature(sa_temp)

        time_end = time.time()

        dist = total_distance(Sub_tour_VNS, distance_matrix)
        no_veh = len(Sub_tour_VNS)
        time_exe = time_end - time_start
        analysis_iterations_total += iterations
        print(counter + 1, dist, no_veh, time_exe)

        if(not is_exp):
            outputfile.write(
                f'Iteration: {counter + 1}, cost: {dist}, number of vehicles: {no_veh}, runtime VNS: {time_exe} \n')
            outputfile.write(f'{Sub_tour_VNS} \n')

        DIST.append(dist)
        NO_VEHICLE.append(no_veh)
        RUN_TIME.append(time_exe)
        print(Sub_tour_VNS)

    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.plot(analysis_convergence_total, color='b', label='Global solution')
    plt.plot(analysis_convergence_shaking, color='g', label='Shaking solution')
    plt.plot(analysis_convergence_local, color='m',
             label='Local solution', alpha=0.5)
    plt.vlines(x=analysis_restarts[:-1], ymin=min(analysis_convergence_total), ymax=max(
        analysis_convergence_shaking), colors='k', label='Restart', linestyles="dashed")
    plt.title('Convergence')
    plt.ylabel('Costs')
    plt.xlabel('Iterations')
    plt.legend()

    if(is_exp):
        plt_path = f"exp_id_{exp_params['exp_id']}_convergence_total"
        plt.savefig("%s/experiments/solomon/%s/%s/convergence/%s.png" %
                    # (dir_name, file, exp_params['test_name'], plt_path))
                    (dir_name, file, exp_params['test_name'], plt_path))
    else:
        plt_path = f"convergence_total"
        plt.savefig("%s/experiments/solomon/%s/%s/result.csv" %
                    (dir_name, file, exp_params["test_name"]))

    plt.close()

    # Initiale Kosten, Initiale Lösung, Beste Kosten, Beste Lösung, Iteration of Last_Improvement, #Total_Iterations, Laufzeit,
    return_best_solution = Sub_tour_VNS
    return_best_distance = total_distance(
        return_best_solution, distance_matrix)
    return_last_improvement = analysis_last_update
    return_total_iterations = analysis_iterations_total
    return_runtime = sum(RUN_TIME)

    if(not is_exp):
        outputfile.write(
            f'\nsmallest numb of veh: {min(NO_VEHICLE)} average numb vehicle:{np.mean(NO_VEHICLE)} std. deviation numb vehicle: {np.std(NO_VEHICLE)} \n')
        outputfile.write(
            f'smallest cost: {min(DIST)}, average cost: {np.mean(DIST)}, std. deviateion cost: {np.std(DIST)}, average run time: {np.mean(RUN_TIME)} \n')
        outputfile.write(f"==================== \n")
        outputfile.close()

    print('\n', min(NO_VEHICLE), np.mean(NO_VEHICLE), np.std(NO_VEHICLE))
    print('Initial:', total_distance(ini_tour, distance_matrix), len(ini_tour))
    print(min(DIST), np.mean(DIST), np.std(DIST), np.mean(RUN_TIME))
    print("====================")

    if(is_exp):
        return return_best_solution, return_best_distance, return_last_improvement, initial_distance, return_total_iterations, return_runtime, performance_counter

    return Sub_tour_VNS, total_distance(Sub_tour_VNS, distance_matrix), return_last_improvement
