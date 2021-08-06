#!/usr/bin/env python
# coding: utf-8

# # 0. Data Processing
import random as rd
import math
import os
import numpy as np
import copy
import time
import matplotlib
import matplotlib.pyplot as plt
import vns.src.config.vns_config as cfg
import vns.src.config.preprocessing_config as prep_cfg
from math import radians, cos, sin, asin, sqrt, exp
from vns.src.helpers.DistanceMatrix import DistanceMatrix
from pathlib import Path
from collections import namedtuple

matplotlib.rcParams["savefig.dpi"] = 600

rd.seed(0)

plt.ioff()
dir_name = os.path.dirname(os.path.realpath('__file__'))


# Basic Function

# this method was provided by Nina Schwarm, (c) FU Berlin 2021
def haversine(latitude_target, longitude_target, latitude_origin, longitude_origin):
    r = 6372.8
    d_latitude = radians(latitude_origin - latitude_target)
    d_longitude = radians(longitude_origin - longitude_target)
    latitude_target = radians(latitude_target)
    latitude_origin = radians(latitude_origin)

    a = sin(d_latitude / 2) ** 2 + cos(latitude_target) * \
        cos(latitude_origin) * sin(d_longitude / 2) ** 2
    c = 2 * asin(sqrt(a))

    haversine_dist = r * c
    return haversine_dist


def get_travel_time_matrix(cust_size, xcoor, ycoor, xcoor_end, ycoor_end):
    travel_time_matrix = np.zeros((cust_size + 1, cust_size + 1))
    for i in range(cust_size + 1):
        for j in range(cust_size + 1):
            if i != j: 
                travel_time_matrix[i][j] = haversine(
                    ycoor[j], xcoor[j], ycoor_end[i], xcoor_end[i]) * cfg.minutes_per_kilometer
    return travel_time_matrix


def get_customer_size(current_tour):
    counter = 0
    for tour in current_tour:
        for order in tour:
            if order != 0:
                counter += 1
    return counter


# def total_cost(tours, travel_time, service_time, order_ids):
#     total_cost = 0
#     for tour in tours:
#         tour_time = 0
#         for i in range(len(tour) - 1):
#             traffic_phase = "off_peak" if tour_time < prep_cfg.traffic_times["phase_transition"][
#                 "from_shift_start"] else "phase_transition" if tour_time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"
#             tour_time += service_time[order_ids[tour[i]]+":" +
#                                       traffic_phase] if order_ids[tour[i]] != "order_0" else 0
#             tour_time += travel_time[tour[i]][tour[i + 1]]
#         tour_cost = (tour_time*cfg.cost_per_minute) + \
#             cfg.cost_per_driver
#         total_cost += tour_cost
#     return total_cost


# def total_cost(tours, travel_time, service_time, ready_time, order_ids):
#     total_cost = 0
#     for tour in tours:
#         tour_time = 0
#         for i in range(len(tour) - 1):
#             traffic_phase = "off_peak" if tour_time < prep_cfg.traffic_times["phase_transition"][
#                 "from_shift_start"] else "phase_transition" if tour_time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"
#             tour_time += max(service_time[order_ids[tour[i]]+":" +
#                                           traffic_phase] if order_ids[tour[i]] != "order_0" else 0 + travel_time[tour[i]][tour[i + 1]], ready_time[tour[i+1]]-tour_time)
#             # tour_time += travel_time[tour[i]][tour[i + 1]]
#         tour_cost = (tour_time*cfg.cost_per_minute) + \
#             cfg.cost_per_driver
#         total_cost += tour_cost
#     return total_cost

def total_cost(tours, travel_time_mat, service_time, ready_time, order_ids):
    total_cost = 0
    for tour in tours:
        current_time = 0
        travel_time = 0
        for i in range(len(tour) - 1):
            traffic_phase = "off_peak" if current_time < prep_cfg.traffic_times["phase_transition"][
            "from_shift_start"] else "phase_transition" if current_time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"
            if i == 0:
                travel_time += travel_time_mat[tour[i]][tour[i + 1]]
                current_time += max(travel_time_mat[tour[i]][tour[i + 1]], ready_time[tour[i+1]])
            else:
                travel_time += max(service_time[order_ids[tour[i]]+":" +
                                        traffic_phase] + travel_time_mat[tour[i]][tour[i + 1]], 
                                        ready_time[tour[i+1]]-current_time)
                current_time += max(service_time[order_ids[tour[i]]+":" +
                                        traffic_phase] + travel_time_mat[tour[i]][tour[i + 1]], 
                                        ready_time[tour[i+1]]-current_time)
        tour_cost = (travel_time*cfg.cost_per_minute) + \
            cfg.cost_per_driver
        total_cost += tour_cost
    return total_cost

def cal_total_time(tours, travel_time, service_time, ready_time, order_ids):
    total_time = 0
    for tour in tours:
        tour_time = 0
        for i in range(len(tour) - 1):
            traffic_phase = "off_peak" if tour_time < prep_cfg.traffic_times["phase_transition"][
                "from_shift_start"] else "phase_transition" if tour_time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"
            tour_time += max(service_time[order_ids[tour[i]]+":" +
                                          traffic_phase] if order_ids[tour[i]] != "order_0" else 0 + travel_time[tour[i]][tour[i + 1]], ready_time[tour[i+1]]-tour_time)
            #tour_time += travel_time[tour[i]][tour[i + 1]]
        total_time += tour_time
    return total_time


def total_distance(tours, travel_time_matrix):
    total_distance = 0
    for tour in tours:
        tour_distance = 0
        for i in range(len(tour) - 1):
            tour_distance += (travel_time_matrix[tour[i]]
                              [tour[i + 1]]/cfg.minutes_per_kilometer)
        total_distance += tour_distance
    return total_distance


def idle_time(tours, travel_time_matrix, service_time, ready_time, order_ids, interval=0):
    total_idle_time = 0
    for tour in tours:
        tour_idle_time = 0
        if(interval == 0):
            time = ready_time[tour[1]]
        else:
            ready_time = ready_time[tour[1]]
            time = math.ceil(ready_time/interval) * interval
        for i in range(len(tour) - 1):
            traffic_phase = "off_peak" if time < prep_cfg.traffic_times["phase_transition"][
                "from_shift_start"] else "phase_transition" if time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"
            time = time + (service_time[order_ids[tour[i]]+":" + traffic_phase] if order_ids[tour[i]
                                                                                             ] != "order_0" else 0) + travel_time_matrix[tour[i]][tour[i+1]]
            if(time < ready_time[tour[i+1]]):
                tour_idle_time += ready_time[tour[i+1]] - time
                time = ready_time[tour[i+1]]
        total_idle_time += tour_idle_time
    return total_idle_time


def time_checker(tour, travel_time, service_time, ready_time, due_time, order_id):
    time = 0
    counter = 0
    for i in range(1, len(tour)):
        traffic_phase = "off_peak" if time < prep_cfg.traffic_times["phase_transition"][
            "from_shift_start"] else "phase_transition" if time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"

        if(order_id[tour[i-1]] != 'order_0'):
            if(ready_time[tour[i-1]] <= (time + service_time[order_id[tour[i-1]]+":"+traffic_phase])):
                time = time + \
                    service_time[order_id[tour[i-1]]+":"+traffic_phase] + \
                    travel_time[tour[i-1]][tour[i]]
            else:
                time = ready_time[tour[i-1]] + \
                    travel_time[tour[i-1]][tour[i]]
        else:
            if(ready_time[tour[i-1]] <= time):
                time = time + \
                    travel_time[tour[i-1]][tour[i]]
        if (time <= due_time[tour[i]]):
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

# Sim Anneal Accept is only for analysing simulated annealing acceptance rate


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


# Exportig Functions


# def create_planning_df(tours, current_order_df, travel_time_matrix, service_time_matrix, ready_time, due_time, planning_df=None, interval=0):
#     if planning_df is None:
#         planning_df = current_order_df[[
#             'CUST_NO', 'SERVICETIME', 'order_id', 'end_poi_id']].copy(deep=True)
#         planning_df['VISITED'] = False
#         planning_df['SCHEDULED_TIME'] = np.NaN
#         planning_df['SCHEDULED_TOUR'] = np.NaN

#     tour_id = 0
#     for tour in tours:
#         if(interval == 0):
#             time = current_order_df.loc[current_order_df['CUST_NO']
#                                         == tour[1], 'READYTIME'].values[0]
#         else:
#             ready_time = current_order_df.loc[current_order_df['CUST_NO']
#                                               == tour[1], 'READYTIME'].values[0]
#             time = math.ceil(ready_time/interval) * interval
#         for i in range(1, len(tour)-1):
#             order_i = current_order_df.loc[current_order_df['CUST_NO']
#                                            == tour[i], 'order_id'].values[0]
#             order_i_1 = current_order_df.loc[current_order_df['CUST_NO']
#                                              == tour[i-1], 'order_id'].values[0]
#             traffic_phase = "off_peak" if time < prep_cfg.traffic_times["phase_transition"][
#                 "from_shift_start"] else "phase_transition" if time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"

#             if(order_i_1 != 'order_0' and order_i != 'order_0'):
#                 if(current_order_df.loc[current_order_df['CUST_NO'] == tour[i-1], 'READYTIME'].values[0] <= (time + service_time_matrix[order_i_1+":"+traffic_phase])):
#                     time = time + \
#                         service_time_matrix[order_i_1+":"+traffic_phase] + \
#                         travel_time_matrix[order_i_1+":"+order_i]
#                 else:
#                     time = current_order_df.loc[current_order_df['CUST_NO'] == tour[i-1],
#                                                 'READYTIME'].values[0] + travel_time_matrix[order_i_1+":"+order_i]
#             else:
#                 if(current_order_df.loc[current_order_df['CUST_NO'] == tour[i-1], 'READYTIME'].values[0] <= time):
#                     time = time + travel_time_matrix[order_i_1+":"+order_i]

#             planning_df.at[tour[i], 'SCHEDULED_TIME'] = time
#             planning_df.at[tour[i], 'SCHEDULED_TOUR'] = tour_id

#         tour_id += 1

#     return planning_df


def create_planning_df(tours, current_order_df, travel_time_matrix, service_time_matrix, ready_time, due_time, planning_df=None, interval=0):
    if planning_df is None:
        planning_df = current_order_df[[
            'CUST_NO', 'SERVICETIME', 'order_id', 'end_poi_id']].copy(deep=True)
        planning_df['VISITED'] = False
        planning_df['SCHEDULED_TIME'] = np.NaN
        planning_df['SCHEDULED_TOUR'] = np.NaN

    tour_id = 0
    for tour in tours:
        # Set initial time:
        if(interval == 0):
            # Static solution: Tour begins either at 0 or at ready time of first order of tour
            time = current_order_df.loc[current_order_df['CUST_NO']
                                        == tour[1], 'READYTIME'].values[0]
        else:
            # Dynamic solution: Tour begins either at 0 or at next planning interval
            ready_time = current_order_df.loc[current_order_df['CUST_NO']
                                              == tour[1], 'READYTIME'].values[0]
            time = math.ceil(ready_time/interval) * interval
        for i in range(1, len(tour)-1):
            order_i = current_order_df.loc[current_order_df['CUST_NO']
                                           == tour[i], 'order_id'].values[0]
            order_i_1 = current_order_df.loc[current_order_df['CUST_NO']
                                             == tour[i-1], 'order_id'].values[0]
            traffic_phase = "off_peak" if time < prep_cfg.traffic_times["phase_transition"][
                "from_shift_start"] else "phase_transition" if time < prep_cfg.traffic_times["rush_hour"]["from_shift_start"] else "rush_hour"

            # Wenn 1. Order = 0:
            if(order_i_1 == 'order_0'):
                ready_time = current_order_df.loc[current_order_df['CUST_NO']
                                                  == tour[i], 'READYTIME'].values[0]
                travel_time = travel_time_matrix["order_0:" + order_i]
                # wenn Statische Lösung:
                if(interval == 0):
                    # wenn der Auftrag bereits da ist und verplant werden kann, dann wird der Auftrag so früh wie möglich geplant.
                    # Der frühstmögliche Zeitpunkt ist somit der Zeitpunkt, zu dem die Strecke zurückgelegt wurde

                    if(time == 0 or ready_time < (travel_time)):
                        time = travel_time_matrix["order_0:" + order_i]
                    # Wenn der Auftrag nach der ready_time 0 eintrifft, muss er später verplant werden
                    # Reicht die Zeit vor der Ready-time aus, um den Weg zurückzulegen, so ist die frühstmögliche Zeit die Ready_Time
                    else:
                        time = ready_time
                    # Reicht die Zeit bis zur Readytime nicht aus, um die Strecke zurückzulegen, so ist die scheduled time = travel time

                # wenn dynamische Lösung:
                else:
                    # wenn der Auftrag bereits da ist und verplant werden kann, dann wird der Auftrag so früh wie möglich geplant.
                    # Der frühstmögliche Zeitpunkt ist somit der Zeitpunkt, zu dem die Strecke zurückgelegt wurde
                    if(time == 0):
                        time = travel_time_matrix["order_0:" + order_i]
                    # Wenn der Auftrag nach der ready_time 0 eintrifft, muss er später verplant werden
                    else:
                        # bei der dynamischen Lösung planen wir in Intervallen, dementsprechend kann der Auftrag frühestens nach dem Planungsinterval erfolgen
                        # Auch hier gilt, der frühstmögliche Zeitpunkt ist der Zeitpunkt der Ankunft
                        time = time + travel_time
                service_time = service_time_matrix[order_i +
                                                   ":" + traffic_phase]
                prev_end_time = time + service_time
            elif(order_i != 'order_0'):
                # Wenn der zu Planende Knoten nicht der erste Knoten der Tour ist, so ist er vom vorigen abhängig
                ready_time = current_order_df.loc[current_order_df['CUST_NO']
                                                  == tour[i], 'READYTIME'].values[0]

                travel_time = travel_time_matrix[order_i_1 + ":" + order_i]

                # Statische Lösung:
                if(interval == 0):
                    # Scenario a) Der Auftrag kann nahtlos nach dem vorigen ausgeführt werden. Bedingung: Readytime < als Zeit
                    # dann ist die Scheduled time die time + travel
                    if(ready_time < prev_end_time):
                        time = prev_end_time + travel_time

                    # Scenario b) Der Auftrag ist noch nicht verfügbar und kann dementsprechend erst zu seiner Readytime ausgeführt werden
                    else:
                        # dann ist die scheduled time die ready_time, wenn die travel time dafür ausreicht
                        if(ready_time > prev_end_time + travel_time):
                            time = ready_time
                        # sonst ist die scheduled time die end_time + travel time, wenn die Zeit nicht dafür ausreicht
                        else:
                            time = prev_end_time + travel_time
                # dynamische Lösung:
                    # Readytimes müssten alle verfügbar sein -> kann einfach verplant werden
                else:
                    time = prev_end_time + travel_time

            service_time = service_time_matrix[order_i +
                                               ":" + traffic_phase]
            prev_end_time = time + service_time

            planning_df.at[tour[i], 'SCHEDULED_TIME'] = time
            planning_df.at[tour[i], 'SCHEDULED_TOUR'] = tour_id

        tour_id += 1

    return planning_df

# # Local search operators

# 2-opt (operating one one route)


def two_opt_move(tour, travel_time, service_time, ready_time, due_time, temperture, order_ids, visibility):
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
                if time_checker(new_tour, travel_time, service_time, ready_time, due_time, order_ids):
                    imp = total_cost([tour], travel_time, service_time, ready_time, order_ids) - \
                        total_cost([new_tour], travel_time,
                                   service_time, ready_time, order_ids)
                    if (imp > best_move["gain"]):
                        best_move = {
                            "node1": i,
                            "node2": j,
                            "gain": round(imp, 4),
                            "tour": new_tour
                        }
    return best_move


def two_opt_search(sub_tour, travel_time, service_time, ready_time, due_time, temperture, order_ids,  visibility):
    Best_Imp = float('-inf')
    Tour = []
    Improved_Route = []

    for i in range(len(sub_tour)):
        best_move = two_opt_move(
            sub_tour[i], travel_time, service_time, ready_time, due_time, temperture, order_ids, visibility_index(sub_tour[i], visibility))

        if best_move["node1"] != -1:
            Best_Imp += best_move["gain"]
            Tour.append(i)
            Improved_Route.append(best_move["tour"])
    return Tour, Improved_Route, Best_Imp


#  3-opt (operating on 1 route)

def exchange(tour, opt_case, a, c, e, travel_time, service_time, ready_time, order_ids):
    """
    Reconnects the tour given three edges to swap.

    """
    b, d, f = a + 1, c + 1, e + 1

    base = total_cost([tour], travel_time, service_time, ready_time, order_ids)

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

    gain = total_cost([sol], travel_time, service_time, ready_time, order_ids)
    return sol, (base - gain)


def three_opt_search_best_improvement(tour, travel_time, service_time, ready_time, due_time, temperature, order_ids, visibility):
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
                            tour, case, a, c, e, travel_time, service_time, ready_time, order_ids)
                        if(gain_expected > best_move["gain"]):
                            if(time_checker(new_tour, travel_time, service_time, ready_time, due_time, order_ids)):
                                best_move = {
                                    "a": a,
                                    "c": c,
                                    "e": e,
                                    "opt_case": case,
                                    "gain": gain_expected,
                                    "tour": new_tour
                                }

    return best_move

# 2-opt* (Operating on 2 routes)


def three_opt_search(sub_tour, travel_time, service_time, ready_time, due_time, temperture, order_ids, visibility):
    Best_Imp = float('-inf')
    Tour = []
    Improved_Route = []

    for i in range(len(sub_tour)):
        best_move = three_opt_search_best_improvement(
            sub_tour[i], travel_time, service_time, ready_time, due_time, temperture, order_ids, visibility_index(sub_tour[i], visibility))

        if best_move["opt_case"] != -1:
            Best_Imp += best_move["gain"]
            Tour.append(i)
            Improved_Route.append(best_move["tour"])

    return Tour, Improved_Route, Best_Imp


def two_optstar(tour1, tour2, travel_time, service_time, ready_time, due_time, demand, capacity, order_ids, visibility_t1, visibility_t2):
    best_imp = float('-inf')
    position1 = -1
    position2 = -1

    base = total_cost([tour1], travel_time, service_time, ready_time, order_ids) + \
        total_cost([tour2], travel_time, service_time, ready_time, order_ids)

    for i in range(visibility_t1, len(tour1) - 1):
        for j in range(visibility_t2, len(tour2) - 1):
            new_tour1 = tour1[:i + 1] + tour2[j + 1:]
            new_tour2 = tour2[:j + 1] + tour1[i + 1:]
            tour1_new_demand = sum(demand[new_tour1])
            tour2_new_demand = sum(demand[new_tour2])

            if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):

                time_check_2opts1 = time_checker(
                    new_tour1, travel_time, service_time, ready_time, due_time, order_ids)
                time_check_2opts2 = time_checker(
                    new_tour2, travel_time, service_time, ready_time, due_time, order_ids)

                if time_check_2opts1 and time_check_2opts2:
                    twoopts_cost = base - \
                        (total_cost([new_tour1], travel_time, service_time, ready_time, order_ids) +
                         total_cost([new_tour2], travel_time, service_time, ready_time, order_ids))

                    if twoopts_cost > best_imp:
                        best_imp = twoopts_cost
                        position1 = i
                        position2 = j

    return position1, position2, best_imp


def two_optstar_search(sub_tour, travel_time, service_time, ready_time, due_time, demand, capacity, order_ids, visibility):
    Best_Imp = float('-inf')
    Tour = []
    Improved_Route = []

    for t1 in range(len(sub_tour) - 1):
        for t2 in range(t1 + 1, len(sub_tour)):
            [position1, position2, imp] = two_optstar(sub_tour[t1], sub_tour[t2], travel_time, service_time,
                                                      ready_time, due_time, demand, capacity, order_ids, visibility_index(sub_tour[t1], visibility), visibility_index(sub_tour[t2], visibility))

            if imp > Best_Imp:
                Tour = [t1, t2]
                New_tour1 = sub_tour[t1][:position1 +
                                         1] + sub_tour[t2][position2 + 1:]
                New_tour2 = sub_tour[t2][:position2 +
                                         1] + sub_tour[t1][position1 + 1:]
                Improved_Route = [New_tour1, New_tour2]

                Best_Imp = imp

    return Tour, Improved_Route, Best_Imp

# Main function of local search
# Input: Shaking solution
# Function: Calculates best improvement with each operator, accepts best solution with probability of simulated annealing
# Output: New local solution


def local_search(Sub_tour, travel_time, service_time, ready_time, due_time, demand, temperature, order_ids, visibility, Local_Tour, performance_counter, sim_anneal_accept, improvement_per_iteration):

    [Tour_two_opt, Improved_Route_two_opt, Improvement_two_opt] = two_opt_search(copy.deepcopy(
        Sub_tour), travel_time, service_time, ready_time, due_time, temperature, order_ids, visibility)

    [Tour_two_opt_star, Improved_Route_two_opt_star, Improvement_two_opt_star] = two_optstar_search(copy.deepcopy(
        Sub_tour), travel_time, service_time, ready_time, due_time, demand, cfg.capacity, order_ids, visibility)

    [Tour_three_opt, Improved_Route_three_opt, Improvement_three_opt] = three_opt_search(copy.deepcopy(
        Sub_tour), travel_time, service_time, ready_time, due_time, temperature, order_ids, visibility)

    if(Improvement_two_opt_star >= Improvement_two_opt and Improved_Route_two_opt_star >= Improved_Route_three_opt):
        Tour = Tour_two_opt_star
        Improved_Route = Improved_Route_two_opt_star
        performance_counter['two-opt-star'] += 1
    elif(Improvement_two_opt >= Improvement_two_opt_star and Improved_Route_two_opt >= Improved_Route_three_opt):
        Tour = Tour_two_opt
        Improved_Route = Improved_Route_two_opt
        performance_counter['two-opt'] += 1
    if(Improvement_three_opt >= Improvement_two_opt and Improved_Route_three_opt >= Improved_Route_two_opt_star):
        Tour = Tour_three_opt
        Improved_Route = Improved_Route_three_opt
        performance_counter['three-opt'] += 1
    else:
        Tour = Tour_two_opt
        Improved_Route = Improved_Route_two_opt

    TMPSub_tour = copy.deepcopy(Sub_tour)
    for t in range(len(Tour)):
        TMPSub_tour[Tour[t]] = Improved_Route[t]

    CurrImprovement = total_cost(Local_Tour,  travel_time, service_time, ready_time, order_ids) - \
        total_cost(TMPSub_tour, travel_time,
                   service_time, ready_time, order_ids)

    # Simulated annealing
    if CurrImprovement > 0 or simulated_annealing_improvement(temperature, CurrImprovement, sim_anneal_accept):
        improvement_per_iteration.append(CurrImprovement)
        Sub_tour = TMPSub_tour
    else:
        improvement_per_iteration.append(0)
        Sub_tour = Local_Tour

    return Sub_tour


# Shaking

def sort_tours(Sub_tour):
    sorted_tours = sorted(Sub_tour, key=len)
    divider = len(sorted_tours[int(len(sorted_tours)/2)])
    short_tours = []
    long_tours = []
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


def shaking(Input_tour, travel_time, service_time, ready_time, due_time, demand, neighbourhood, capacity, order_ids, visibility):

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
                        New_tour1, travel_time, service_time, ready_time, due_time, order_ids)
                    time_check2 = time_checker(
                        New_tour2, travel_time, service_time, ready_time, due_time, order_ids)
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
                            New_tour1, travel_time, service_time, ready_time, due_time, order_ids)

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
                            New_tour1, travel_time, service_time, ready_time, due_time, order_ids)
                        time_check2 = time_checker(
                            New_tour2, travel_time, service_time, ready_time, due_time, order_ids)
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
                            New_tour1, travel_time, service_time, ready_time, due_time, order_ids)
                        time_check2 = time_checker(
                            New_tour2, travel_time, service_time, ready_time, due_time, order_ids)
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


# Main Function to run VNS
def run_vns(file, ini_tour, all_order_df, visibility, planning_df, interval, is_exp=False, **exp_params):
    '''---------------------------------------------------------------
    Experiment section: Set parameters, if current run is an experiment
    ------------------------------------------------------------------'''

    if is_exp:
        cfg.shaking["INSERT"]["PROBABILITY"] = exp_params['insert_prob'] if 'insert_prob' in exp_params else cfg.shaking["INSERT"]["PROBABILITY"]
        cfg.shaking["INSERT"]["SORT_LEN"] = exp_params['sort_prob'] if 'sort_prob' in exp_params else cfg.shaking["SORT_LEN"]["PROBABILITY"]
        cfg.vns["InitialTemperature"] = exp_params['initial_temperature'] if 'initial_temperature' in exp_params else cfg.vns["InitialTemperature"]
        cfg.cost_per_hour = exp_params['cost_per_hour'] if 'cost_per_hour' in exp_params else cfg.cost_per_hour
        cfg.cost_per_driver = exp_params['cost_per_driver'] if 'cost_per_driver' in exp_params else cfg.cost_per_driver
        cfg.vns["MaxRestarts"] = exp_params['max_restarts'] if 'max_restarts' in exp_params else cfg.vns["MaxRestarts"]
        cfg.vns["MaxIterations_NoImp"] = exp_params[
            'max_iterations_without_improvement'] if 'max_iterations_without_improvement' in exp_params else cfg.vns["MaxIterations_NoImp"]

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
    duetime = all_order_df['DUETIME'].to_numpy()
    servicetime = DistanceMatrix.load_file(os.path.join(
        dir_name, 'input_data', 'surve_mobility', file + '_service_times'))
    order_ids = all_order_df['order_id'].values.tolist()
    xcoor_end = all_order_df['XCOORD_END'].to_numpy()
    ycoor_end = all_order_df['YCOORD_END'].to_numpy()
    cust_size = get_customer_size(ini_tour)
    capacity = 480

    # Calculate Travel Time Matrix
    travel_time_matrix = get_travel_time_matrix(
        cust_size, xcoor, ycoor, xcoor_end, ycoor_end)

    # set initial temperture for simulated annealing
    sa_temp = cfg.vns['InitialTemperature']

    # set store variables for analysis
    COST = []
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
    initial_cost = total_cost(ini_tour, travel_time_matrix,
                              servicetime, readytime, order_ids)
    performance_counter = {
        'two-opt': 0,
        'three-opt': 0,
        'two-opt-star': 0
    }

    '''---------------------------------------------------------------
                         Actual Algortihm
    ---------------------------------------------------------------'''
    # Output creation
    if(not is_exp):
        target_folder = os.path.join(
            dir_name, "results", "vns", "surve_mobility", file)
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        outputfile = open(os.path.join(
            target_folder, 'output.txt'), 'w')
        outputfile.write(f'File: {file} Customer_Size:{cust_size} \n')
        outputfile.write(
            f'Iteration:0 Distance initial tour:{total_cost(ini_tour, travel_time_matrix, servicetime, readytime, order_ids)} Number of routes initial tour {len(ini_tour)}  \n')

    print('Start', file, cust_size)
    print('0', initial_cost, len(ini_tour))
    # Initial solution
    # Format: List<List<Integer>>, where Integer stands for Customer Number
    Sub_tour_VNS = copy.deepcopy(ini_tour)

    # MAINCODE
    for counter in range(cfg.vns['MaxRestarts']):
        # convergence.append([])

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
            Sub_tour_shaking = shaking(Sub_tour_local_search, travel_time_matrix, servicetime, readytime, duetime, demand,
                                       Shaking_Neighbor[k], capacity, order_ids, visibility)
            # Local search process: for 𝑥𝑠 to adopt somelocal search algorithm to solve local optimalsolution 𝑥𝑙, evaluation function of 𝑥𝑙 is 𝑓(𝑥𝑙)

            global_cost = total_cost(
                Sub_tour_VNS, travel_time_matrix, servicetime, readytime, order_ids)
            shake_cost = total_cost(
                Sub_tour_shaking, travel_time_matrix, servicetime, readytime, order_ids)
            old_cost = total_cost(Sub_tour_local_search,
                                  travel_time_matrix, servicetime, readytime, order_ids)

            Sub_tour_local_search = local_search(copy.deepcopy(Sub_tour_shaking), travel_time_matrix,
                                                 servicetime, readytime, duetime, demand, sa_temp, order_ids, visibility, Sub_tour_local_search, performance_counter, analysis_simulated_annealing_acceptance, analysis_improvement_per_iteration)

            new_cost = total_cost(Sub_tour_local_search,
                                  travel_time_matrix, servicetime, readytime, order_ids)

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
            # convergence[counter].append(total_cost(
            #     Sub_tour_VNS, travel_time_matrix, servicetime, order_ids))
            analysis_convergence_total.append(total_cost(
                Sub_tour_VNS, travel_time_matrix, servicetime, readytime, order_ids))
            analysis_convergence_local.append(new_cost)
            analysis_convergence_shaking.append(shake_cost)
            if iterations % 5 == 0:
                sa_temp = update_temperature(sa_temp)

        time_end = time.time()

        cost = total_cost(Sub_tour_VNS, travel_time_matrix,
                          servicetime, readytime, order_ids)
        no_veh = len(Sub_tour_VNS)
        time_exe = time_end - time_start
        analysis_iterations_total += iterations
        if(not is_exp):
            outputfile.write(
                f'Iteration: {counter + 1}, cost: {cost}, number of vehicles: {no_veh}, runtime VNS: {time_exe} \n')
            outputfile.write(f'{Sub_tour_VNS} \n')

        print(counter + 1, cost, no_veh, time_exe)

        COST.append(cost)
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
        target_folder = os.path.join(
            dir_name, "results", "vns", "surve_mobility", "experiments", file, exp_params["test_name"], "convergence")
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        plt_path = f"exp_id_{exp_params['exp_id']}_convergence"
        plt.savefig("{}/results/vns/surve_mobility/experiments/{}/{}/convergence/{}.png".format(
            dir_name, file, exp_params["test_name"], plt_path),format='png')
    else:
        target_folder = os.path.join(
            dir_name, "results", "vns", "surve_mobility", file)
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        plt_path = f"convergence_total"
        plt.savefig("%s/results/vns/surve_mobility/%s/visualization/%s.png" %
                    (dir_name, file, plt_path))

    plt.close()

    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.plot([x*100 for x in analysis_simulated_annealing_acceptance], color='b',
             label='Simulated annealing accepts worse solution')
    plt.plot(analysis_improvement_per_iteration,
             color='g', label='Improvement per iteration')
    plt.vlines(x=analysis_restarts[:-1], ymin=min(analysis_convergence_total), ymax=max(
        analysis_convergence_shaking), colors='k', label='Restart', linestyles="dashed")
    plt.title('Simulated annealing')
    plt.ylabel('Costs')
    plt.xlabel('Iterations')
    plt.legend()

    if(is_exp):
        target_folder = os.path.join(
            dir_name, "results", "vns", "surve_mobility", "experiments", file, exp_params["test_name"], "convergence")
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        plt_path = f"exp_id_{exp_params['exp_id']}_sim_annealing"
        plt.savefig("{}/results/vns/surve_mobility/experiments/{}/{}/convergence/{}.png".format(
            dir_name, file, exp_params["test_name"], plt_path),format='png')
        
    
    else:
        target_folder = os.path.join(
            dir_name, "results", "vns", "surve_mobility", file)
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        plt_path = f"sim_annealing_total"
        plt.savefig("%s/results/vns/surve_mobility/%s/visualization/%s.png" %
                    (dir_name, file, plt_path))

    plt.close()

    # Initiale Kosten, Initiale Lösung, Beste Kosten, Beste Lösung, Iteration of Last_Improvement, #Total_Iterations, Laufzeit,
    return_best_cost = total_cost(
        Sub_tour_VNS, travel_time_matrix, servicetime, readytime, order_ids)
    return_best_solution = Sub_tour_VNS
    return_last_improvement = analysis_last_update
    return_total_iterations = analysis_iterations_total
    return_idle_time = idle_time(
        return_best_solution, travel_time_matrix, servicetime, readytime, order_ids)
    return_runtime = sum(RUN_TIME)
    return_tour_length = total_distance(Sub_tour_VNS, travel_time_matrix)

    if(not is_exp):
        outputfile.write(
            f'\nsmallest numb of veh: {min(NO_VEHICLE)} average numb vehicle:{np.mean(NO_VEHICLE)} std. deviation numb vehicle: {np.std(NO_VEHICLE)} \n')
        outputfile.write(
            f'smallest cost: {min(COST)}, average cost: {np.mean(COST)}, std. deviateion cost: {np.std(COST)}, average run time: {np.mean(RUN_TIME)} \n')
        outputfile.write(f"==================== \n")
        outputfile.close()

    print(min(COST), np.mean(COST), np.std(COST), np.mean(RUN_TIME))
    print("====================")

    print("Idle time: ", return_idle_time)

    prepared_travel_time_matrix = DistanceMatrix.load_file(os.path.join(
        dir_name, "input_data", "surve_mobility", file + "_travel_times"))
    planning_df = create_planning_df(
        Sub_tour_VNS, all_order_df, prepared_travel_time_matrix, servicetime, readytime, duetime, planning_df, interval)

    if(is_exp):
        return return_best_solution, return_best_cost, return_idle_time, planning_df, initial_cost, initial_solution, return_tour_length, return_last_improvement, return_total_iterations, return_runtime, performance_counter, analysis_convergence_total

    return all_order_df, planning_df, Sub_tour_VNS, initial_cost
