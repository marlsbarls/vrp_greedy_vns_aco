import numpy as np
from vns.src.algorithm.improved_vns import create_planning_df, total_cost, get_travel_time_matrix, total_distance, idle_time
import vns.src.config.vns_config as cfg
import vns.src.config.preprocessing_config as config


class SavingsAlgorithm:
    def __init__(self, all_orders_df, travel_time_matrix, service_time_matrix, consider_time):
        self.all_orders_df = all_orders_df
        self.travel_time_matrix = travel_time_matrix
        self.service_time_matrix = service_time_matrix

        # consider temporal and spaical distance
        # False is original
        self.consider_time = consider_time 
        # self.consider_time = False 
            
        
    # original
    @staticmethod
    def time_checker(tour, current_order_df, travel_time_matrix, service_time_matrix, ready_time, due_time, time=0):
        counter = 0
        for i in range(1, len(tour)):
            order_i = current_order_df.loc[current_order_df['CUST_NO']
                                           == tour[i], 'order_id'].values[0]
            order_i_1 = current_order_df.loc[current_order_df['CUST_NO']
                                             == tour[i-1], 'order_id'].values[0]
            # time = max(time, ready_time[tour[i - 1]]) + service_time[tour[i - 1]] + travel_time[tour[i - 1]][tour[i]]
            if (time < config.traffic_times["rush_hour"]["from_shift_start"]):
                traffic_phase = "rush_hour"
            else:
                traffic_phase = "phase_transition"
            if(order_i_1 != 'order_0'):
                if(current_order_df.loc[current_order_df['CUST_NO'] == tour[i-1], 'READYTIME'].values[0] <= (time + service_time_matrix[order_i_1+":"+traffic_phase])):
                    time = time + \
                        service_time_matrix[order_i_1+":"+traffic_phase] + \
                        travel_time_matrix[order_i_1+":"+order_i]
                else:
                    time = current_order_df.loc[current_order_df['CUST_NO'] == tour[i-1],
                                                'READYTIME'].values[0] + travel_time_matrix[order_i_1+":"+order_i]
            else:
                if(current_order_df.loc[current_order_df['CUST_NO'] == tour[i-1], 'READYTIME'].values[0] <= time):
                    time = time + travel_time_matrix[order_i_1+":"+order_i]
            if (time <= current_order_df.loc[current_order_df['CUST_NO'] == tour[i], 'DUETIME'].values[0]):
                counter += 1
            else:
                break
        if counter == len(tour) - 1:
            return True
        else:
            return False

    @staticmethod
    def _get_current_orders(tours, planning_df, time):
        # Returns list with currently executed order for each tour
        current_orders = []
        tour_ids = []
        i = 0
        for sub_tour in tours:
            temp_df = planning_df.loc[planning_df['VISITED'] == True]
            temp_df = temp_df.loc[temp_df["SCHEDULED_TOUR"] == i]
            current_order = temp_df.max()['CUST_NO']
            if not np.isnan(current_order):
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

    def initialization(self, time=0):

        current_order_df = self.all_orders_df[(
            self.all_orders_df.AVAILABLETIME <= time)]
        demand = current_order_df['DEMAND'].to_numpy()
        readytime = current_order_df['READYTIME'].to_numpy()
        duetime = current_order_df['DUETIME'].to_numpy()
        order_ids = current_order_df['order_id'].values.tolist()
        travel_time = get_travel_time_matrix(len(self.all_orders_df)-1, self.all_orders_df['XCOORD'], self.all_orders_df['YCOORD'], self.all_orders_df['XCOORD_END'], self.all_orders_df['YCOORD_END'])
        all_order_ids = self.all_orders_df['order_id'].values.tolist()

        # Route Construction
        # Savings Initialization - Roundtrip from depot to every customer
        Sub_tour = []
        for index, row in current_order_df.iterrows():
            cust_no = row['CUST_NO']
            if(cust_no == 0):
                continue
            else:
                sub = [0, cust_no, 0]
                Sub_tour.append(sub)

        saving_list = []

        for order_i in order_ids:
            for order_j in order_ids:
                if ((order_i != order_j) and (order_i != "order_0") and (order_j != "order_0")):
                    i = current_order_df.loc[current_order_df['order_id']
                                             == order_i, 'CUST_NO'].values[0]
                    j = current_order_df.loc[current_order_df['order_id']
                                             == order_j, 'CUST_NO'].values[0]
                    if not self.consider_time:
                        saving = round(self.travel_time_matrix[order_i+":"+"order_0"] +
                                   self.travel_time_matrix["order_0"+":"+order_j] - self.travel_time_matrix[order_i+":"+order_j])
                    elif self.consider_time:
                        # punishment if ready_time j > ready_time i
                        space_distance = round(self.travel_time_matrix[order_i+":"+"order_0"] +
                                   self.travel_time_matrix["order_0"+":"+order_j] - self.travel_time_matrix[order_i+":"+order_j])
                        ready_time_i = current_order_df.loc[current_order_df['CUST_NO']
                                             == i, 'READYTIME'].values[0]
                        ready_time_j = current_order_df.loc[current_order_df['CUST_NO']
                                             == j, 'READYTIME'].values[0]
                        if ready_time_j <= ready_time_i:
                            temp_distance = 0
                        else:
                            # temp_distance = ready_time_j - ready_time_i 
                            temp_distance = ready_time_i - ready_time_j  

                        saving = space_distance + temp_distance

                    saving_list.append([i, j, saving])

        np.asarray(saving_list)
        saving_list_sorted = sorted(
            saving_list, key=lambda x: x[2], reverse=True)

        while len(saving_list_sorted) != 0:
            # ind is i and j
            ind = saving_list_sorted[0][:2]
            # Check which subtour i and j are in
            a = [i for i in range(len(Sub_tour)) if ind[0] in Sub_tour[i]]
            b = [i for i in range(len(Sub_tour)) if ind[1] in Sub_tour[i]]

            # i and j must be part of different tours and i must be at end of tour and j at the beginning of tour
            if ((a != b) and (ind[0] == Sub_tour[a[0]][-2]) and (ind[1] == Sub_tour[b[0]][1])):
                new_sub = Sub_tour[a[0]][:-1] + Sub_tour[b[0]][1:]

                time_check = self.time_checker(
                    new_sub, current_order_df, self.travel_time_matrix, self.service_time_matrix, readytime, duetime)

                # time_check = self.time_checker(new_sub, travel_time, self.service_time_matrix, readytime, duetime, all_order_ids)

                merge_demand = sum(demand[Sub_tour[a[0]]]) + \
                    sum(demand[Sub_tour[b[0]]])

                del saving_list_sorted[0]

                if (merge_demand <= cfg.capacity and (time_check)):
                    for ele in sorted([a[0], b[0]], reverse=True):
                        del Sub_tour[ele]

                    unlink_nodes = new_sub[2:-2]

                    for node in unlink_nodes:
                        index = [i for i in range(len(saving_list_sorted)) if
                                 saving_list_sorted[i][0] == node or saving_list_sorted[i][1] == node]
                        for j in sorted(index, reverse=True):
                            del saving_list_sorted[j]

                    # test
                    k = [i for i in range(len(saving_list_sorted)) if
                         saving_list_sorted[i][0] == ind[1] or saving_list_sorted[i][1] == ind[0]]
                    if (len(k) != 0):
                        del saving_list_sorted[k[0]]

                    Sub_tour.append(new_sub)

            else:
                del saving_list_sorted[0]

        Ini_tour = Sub_tour
        planning_df = create_planning_df(
            Ini_tour, self.all_orders_df, self.travel_time_matrix, self.service_time_matrix, readytime, duetime)

        

        cost = total_cost(Ini_tour, travel_time, self.service_time_matrix, self.all_orders_df['READYTIME'], all_order_ids)

        # Ini_tour_4 = Ini_tour[4]
        # time_check = self.time_checker_new(Ini_tour_4, travel_time, self.service_time_matrix, readytime, duetime, all_order_ids)

        return current_order_df, planning_df, Ini_tour

    def insert_new(self, current_tour, planning_df, time, interval):

        # Think this only matters in vns
        planning_df.loc[planning_df['SCHEDULED_TIME']
                        <= time, 'VISITED'] = True

        # Alternative I: New orders can be added to the end of existing routes (savings between new orders and end points of all other ones)
        not_scheduled_list = planning_df.loc[planning_df['SCHEDULED_TIME'].isnull(
        ), 'CUST_NO']
        current_order_df = self.all_orders_df[(
            self.all_orders_df.AVAILABLETIME <= time) & self.all_orders_df['CUST_NO'].isin(not_scheduled_list)]

        # Sollte hier vielleicht current_order_df stehen?
        demand = self.all_orders_df['DEMAND'].to_numpy()
        readytime = current_order_df['READYTIME'].to_numpy()
        duetime = current_order_df['DUETIME'].to_numpy()

        travel_time = get_travel_time_matrix(len(self.all_orders_df)-1, self.all_orders_df['XCOORD'], self.all_orders_df['YCOORD'], self.all_orders_df['XCOORD_END'], self.all_orders_df['YCOORD_END'])
        all_order_ids = self.all_orders_df['order_id'].values.tolist()
        all_readytime = self.all_orders_df['READYTIME'].to_numpy()
        all_duetime = self.all_orders_df['DUETIME'].to_numpy()

        new_orders = current_order_df['CUST_NO'].values.tolist()
        end_orders = []
        for tour in current_tour:
            end_orders.append(tour[-2])

        possible_saving_ind = new_orders + end_orders
        # Route Construction
        # Savings Initialization - Roundtrip from depot to every customer
        new_sub_tours = []
        for index, row in current_order_df.iterrows():
            cust_no = row['CUST_NO']
            if(cust_no == 0):
                continue
            else:
                sub = [0, cust_no, 0]
                new_sub_tours.append(sub)

        saving_list = []
        order_ids = []
        for order in possible_saving_ind:
            order_ids.append(
                self.all_orders_df.loc[self.all_orders_df['CUST_NO'] == order, 'order_id'].values[0])

        for order_i in order_ids:
            for order_j in order_ids:
                if ((order_i != order_j) and (order_i != "order_0") and (order_j != "order_0")):
                    saving = round(self.travel_time_matrix[order_i+":"+"order_0"] +
                                   self.travel_time_matrix["order_0"+":"+order_j] - self.travel_time_matrix[order_i+":"+order_j])
                    i = self.all_orders_df.loc[self.all_orders_df['order_id']
                                               == order_i, 'CUST_NO'].values[0]
                    j = self.all_orders_df.loc[self.all_orders_df['order_id']
                                               == order_j, 'CUST_NO'].values[0]
                    if not self.consider_time:
                        saving = round(self.travel_time_matrix[order_i+":"+"order_0"] +
                                   self.travel_time_matrix["order_0"+":"+order_j] - self.travel_time_matrix[order_i+":"+order_j])
                    elif self.consider_time:
                        # punishment if ready_time j > ready_time i
                        space_distance = round(self.travel_time_matrix[order_i+":"+"order_0"] +
                                   self.travel_time_matrix["order_0"+":"+order_j] - self.travel_time_matrix[order_i+":"+order_j])
                        ready_time_i = self.all_orders_df .loc[self.all_orders_df ['CUST_NO']
                                             == i, 'READYTIME'].values[0]
                        ready_time_j = self.all_orders_df .loc[self.all_orders_df ['CUST_NO']
                                             == j, 'READYTIME'].values[0]
                        if ready_time_j <= ready_time_i:
                            temp_distance = 0
                        else:
                            # temp_distance = ready_time_j - ready_time_i 
                            temp_distance = ready_time_i - ready_time_j  

                        saving = space_distance + temp_distance

                    saving_list.append([i, j, saving])

        np.asarray(saving_list)
        saving_list_sorted = sorted(
            saving_list, key=lambda x: x[2], reverse=True)

        Sub_tour = current_tour + new_sub_tours
        while len(saving_list_sorted) != 0:
            # ind is i and j
            ind = saving_list_sorted[0][:2]
            # Check which subtour i and j are in
            a = [i for i in range(len(Sub_tour)) if ind[0] in Sub_tour[i]]
            b = [i for i in range(len(Sub_tour)) if ind[1] in Sub_tour[i]]

            # i and j must be part of different tours and i must be at end of tour and j at the beginning of tour
            if ((a != b) and (ind[0] == Sub_tour[a[0]][-2]) and (ind[1] == Sub_tour[b[0]][1])):
                new_sub = Sub_tour[a[0]][:-1] + Sub_tour[b[0]][1:]

                time_check = self.time_checker(
                    new_sub, self.all_orders_df, self.travel_time_matrix, self.service_time_matrix, readytime, duetime)

                # time_check = self.time_checker(new_sub, travel_time, self.service_time_matrix, all_readytime, all_duetime, all_order_ids)

                merge_demand = sum(demand[Sub_tour[a[0]]]) + \
                    sum(demand[Sub_tour[b[0]]])

                del saving_list_sorted[0]

                if (merge_demand <= cfg.capacity and (time_check)):
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

        current_tour = Sub_tour

        planning_df = create_planning_df(
            current_tour, self.all_orders_df, self.travel_time_matrix, self.service_time_matrix, readytime, duetime, planning_df, interval)  # time einfügen, für alle orders, die noch nicht scheduled sind, aber einer Tour zugeordnet sind -> time als Grundlage nutzen und dann drauf addieren

        current_orders = self._get_current_orders(
            current_tour, planning_df, time)

        travel_time = get_travel_time_matrix(len(self.all_orders_df)-1, self.all_orders_df['XCOORD'], self.all_orders_df['YCOORD'], self.all_orders_df['XCOORD_END'], self.all_orders_df['YCOORD_END'])

        all_order_ids = self.all_orders_df['order_id'].values.tolist()
        
        cost = total_cost(current_tour, travel_time, self.service_time_matrix, self.all_orders_df['READYTIME'], all_order_ids)

        return current_order_df, planning_df, current_tour, current_orders
