import numpy as np
import copy
from vns.src.algorithm_solomon.improved_vns_solomon import time_checker, check_sequence, total_distance, create_planning_df


class SavingsAlgorithm:
    def __init__(self, all_orders_df, dist_matrix, capacity, service_time, ready_time=None, due_time=None, dynamic=False):
        self.all_orders_df = all_orders_df
        # if dynamic:
        #     self.all_orders_df["CUST_NO."].astype('int32')
        self.dist_matrix = dist_matrix
        self.servicetime = service_time
        self.readytime = ready_time
        self.duetime = due_time
        self.dynamic = dynamic
        self.capacity = capacity

    @staticmethod
    def _get_current_orders(tours, planning_df, time, dynamic=False):
        # Returns list with currently executed order for each tour
        current_orders = []
        tour_ids = []
        i = 0
        for sub_tour in tours:
            temp_df = planning_df.loc[planning_df['VISITED'] == True]
            temp_df = temp_df.loc[temp_df["SCHEDULED_TOUR"] == i]
            if dynamic:
                if not temp_df.empty:
                    current_order = int(temp_df.max()['CUST_NO'])
                else:
                    current_order = temp_df.max()['CUST_NO']
            else:
                current_order = temp_df.max()['CUST_NO']
            if not np.isnan(current_order) and current_order in sub_tour:
                # If not last order, fix next order
                if(sub_tour[sub_tour.index(current_order) + 1] != 0):
                    current_orders.append(
                        sub_tour[sub_tour.index(current_order) + 1])
                    tour_ids.append(sub_tour.index(current_order) + 1)
                else:  # If last order, fix last order
                    current_orders.append(
                        sub_tour[sub_tour.index(current_order)])
                    tour_ids.append(sub_tour.index(current_order))
            # If driver is on his way to first order, fix at least first order
            # Give short orders (with only one tour) the ability to be moved to another tour
            elif len(sub_tour) > 3:
                current_orders.append(sub_tour[1])
                tour_ids.append((1))

            i = i+1
        return current_orders

    @staticmethod
    def _convert_index_to_cust_no(tours):
        tour_with_cust_no = copy.deepcopy(tours)
        i = 0
        for tour in tours:
            j = 0
            for order in tour:
                tour_with_cust_no[i][j] += 1
                j += 1
            i += 1

        return tour_with_cust_no

    # @staticmethod
    def _convert_cust_no_to_index(self, tours):
        tour_with_index = copy.deepcopy(tours)
        i = 0
        for tour in tours:
            j = 0
            for order in tour:
                tour_with_index[i][j] -= 1
                j += 1
            i += 1

        return tour_with_index

    def initialization(self, time=0):
        if(self.dynamic == 'dynamic'):
            current_order_df = self.all_orders_df[(
                self.all_orders_df.AVAILABLETIME <= time)]
        else:
            current_order_df = self.all_orders_df

        demand = self.all_orders_df['DEMAND'].to_numpy()
        cust_size = len(current_order_df)-1
        current_customers = current_order_df['CUST_NO'].tolist()

        Sub_tour = []
        # for i in current_customers:
        #     if i != 0:
        #         sub = [0, i, 0]
        #         Sub_tour.append(sub)
        # for index, row in current_order_df.iterrows():
        #     # cust_no = row['CUST_NO']
        #     sub = [0, int(row['CUST_NO']), 0]
        #     Sub_tour.append(sub)

        for i in range(cust_size + 1):
            sub = [0, i, 0]
            Sub_tour.append(sub)

        sub_size = len(Sub_tour) - 1
        saving_list = []

        for i in range(1, sub_size + 1):
            for j in range(1, sub_size + 1):
                if i != j:
                    saving = round(
                        self.dist_matrix[i, 0] + self.dist_matrix[0, j] - self.dist_matrix[i, j], 2)
                    saving_list.append([i, j, saving])
                    #     self.dist_matrix[i 0] + self.dist_matrix[0, (j-1)] - self.dist_matrix[(i-1), (j-1)], 2)
                    # saving_list.append([i, j, saving])

        np.asarray(saving_list)
        saving_list_sorted = sorted(
            saving_list, key=lambda x: x[2], reverse=True)

        # while len(saving_list_sorted) >= 1500:
        while len(saving_list_sorted) != 0:
            ind = saving_list_sorted[0][:2]
            a = [i for i in range(len(Sub_tour)) if ind[0] in Sub_tour[i]]
            b = [i for i in range(len(Sub_tour)) if ind[1] in Sub_tour[i]]

            if a != b:
                new_sub11 = Sub_tour[a[0]][::-1][:-1] + Sub_tour[b[0]][1:]
                new_sub12 = Sub_tour[a[0]][::-1][:-1] + \
                    Sub_tour[b[0]][::-1][1:]
                new_sub13 = Sub_tour[a[0]][:-1] + Sub_tour[b[0]][1:]
                new_sub14 = Sub_tour[a[0]][:-1] + Sub_tour[b[0]][::-1][1:]
                # print(new_sub11,new_sub12)

                if check_sequence(ind, new_sub11):
                    new_sub1 = new_sub11
                elif check_sequence(ind, new_sub12):
                    new_sub1 = new_sub12
                elif check_sequence(ind, new_sub13):
                    new_sub1 = new_sub13
                else:
                    new_sub1 = new_sub14

                new_sub2 = new_sub1[::-1]

                time_check1 = time_checker(
                    new_sub1, self.dist_matrix, self.servicetime, self.readytime, self.duetime)
                time_check2 = time_checker(
                    new_sub2, self.dist_matrix, self.servicetime, self.readytime, self.duetime)

                if time_check1:
                    new_sub = new_sub1
                elif time_check2:
                    new_sub = new_sub2

                merge_demand = sum(demand[Sub_tour[a[0]]]) + \
                    sum(demand[Sub_tour[b[0]]])

                del saving_list_sorted[0]
                k = [i for i in range(len(saving_list_sorted)) if
                     saving_list_sorted[i][0] == ind[1] or saving_list_sorted[i][1] == ind[0]]
                del saving_list_sorted[k[0]]

                # print(saving_list_sorted)

                if (merge_demand <= self.capacity and (time_check1 or time_check2)):
                    for ele in sorted([a[0], b[0]], reverse=True):
                        del Sub_tour[ele]

                    unlink_nodes = new_sub[2:-2]

                    for node in unlink_nodes:
                        index = [i for i in range(len(saving_list_sorted)) if
                                 saving_list_sorted[i][0] == node or saving_list_sorted[i][1] == node]
                        for j in sorted(index, reverse=True):
                            del saving_list_sorted[j]

                    Sub_tour.append(new_sub)
            else:
                del saving_list_sorted[0]
                k = [i for i in range(len(saving_list_sorted)) if
                     saving_list_sorted[i][0] == ind[1] or saving_list_sorted[i][1] == ind[0]]
                del saving_list_sorted[k[0]]

        Ini_tour = Sub_tour[1:]
        # Ini_tour = self._convert_index_to_cust_no(Sub_tour)
        return Ini_tour

    def insert_new(self, current_tour, time, interval):
        # current_tour = self._convert_cust_no_to_index(current_tour)

        planning_df = create_planning_df(
            current_tour, self.all_orders_df, self.dist_matrix, self.servicetime, self.readytime, self.duetime)

        # Think this only matters in vns
        planning_df.loc[planning_df['SCHEDULED_TIME']
                        <= time, 'VISITED'] = True

        # Alternative I: New orders can be added to the end of existing routes (savings between new orders and end points of all other ones)
        not_scheduled_list = planning_df.loc[planning_df['SCHEDULED_TIME'].isnull(),
                                             'CUST_NO'][1:].tolist()
        current_order_df = self.all_orders_df[(
            self.all_orders_df.AVAILABLETIME <= time) & self.all_orders_df['CUST_NO'].isin(not_scheduled_list)]

        demand = self.all_orders_df['DEMAND'].to_numpy()
        new_orders = current_order_df['CUST_NO'].tolist()
        end_orders = []
        for tour in current_tour:
            end_orders.append(tour[-2])

        possible_saving_ind = new_orders + end_orders
        new_sub_tours = []
        for index, row in current_order_df.iterrows():
            cust_no = index
            if(cust_no == 0):
                continue
            else:
                sub = [0, int(cust_no), 0]
                new_sub_tours.append(sub)

        saving_list = []

        for i in possible_saving_ind:
            for j in possible_saving_ind:
                if ((i != j) and (i != "0") and (j != "0")):
                    saving = round(self.dist_matrix[i][0] +
                                   self.dist_matrix[0][j] - self.dist_matrix[i][j])
                    saving_list.append([i, j, saving])

        np.asarray(saving_list)
        saving_list_sorted = sorted(
            saving_list, key=lambda x: x[2], reverse=True)

        Sub_tour = current_tour + new_sub_tours
        while len(saving_list_sorted) != 0:
            ind = saving_list_sorted[0][:2]
            a = [i for i in range(len(Sub_tour)) if ind[0] in Sub_tour[i]]
            b = [i for i in range(len(Sub_tour)) if ind[1] in Sub_tour[i]]

            if ((a != b) and (ind[0] == Sub_tour[a[0]][-2]) and (ind[1] == Sub_tour[b[0]][1])):
                new_sub = Sub_tour[a[0]][:-1] + Sub_tour[b[0]][1:]

                time_check = time_checker(
                    new_sub, self.dist_matrix, self.servicetime, self.readytime, self.duetime)

                merge_demand = sum(demand[Sub_tour[a[0]]]) + \
                    sum(demand[Sub_tour[b[0]]])

                del saving_list_sorted[0]

                if (merge_demand <= self.capacity and (time_check)):
                    for ele in sorted([a[0], b[0]], reverse=True):
                        del Sub_tour[ele]

                    unlink_nodes = new_sub[2:-2]

                    for node in unlink_nodes:
                        index = [i for i in range(len(saving_list_sorted)) if
                                 saving_list_sorted[i][0] == node or saving_list_sorted[i][1] == node]
                        for j in sorted(index, reverse=True):
                            del saving_list_sorted[j]

                    k = [i for i in range(len(saving_list_sorted)) if
                         saving_list_sorted[i][0] == ind[1] or saving_list_sorted[i][1] == ind[0]]
                    if (len(k) != 0):
                        del saving_list_sorted[k[0]]

                    Sub_tour.append(new_sub)

            else:
                del saving_list_sorted[0]

        # Warning: Tour contains indices and not customer numbers
        current_tour = Sub_tour

        return current_tour
