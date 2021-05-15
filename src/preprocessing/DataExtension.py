'''
Original Code provided by Nina Schwarm, (c) FU Berlin 2021
Modified by Lucas Merker & Marlene Buhl
'''

import pandas as pd
import numpy as np
import os
from pathlib import Path


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

        # MOD: see above
        self.available_time = available_time
        self.x_end = x_end
        self.y_end = y_end


class DataExtension:
    def __init__(self, base_dir, file_name, dynamicity, lam, average_orders):
        self.path_original_de = os.path.join(
            base_dir, 'data', 'solomon', file_name + ".csv")
        self.path_target_de = os.path.join(
            base_dir, 'data', 'solomon_dynamic', file_name + "_" + str(int(dynamicity*100)) + "_percent_dynamicity.csv")
        self.dynamicity = dynamicity
        self.node_num, self.nodes, self.vehicle_num, self.vehicle_capacity = self.create_from_file(
            self.path_original_de)
        self.time_slices = 32
        self.interval_length = self.vehicle_capacity / self.time_slices
        self.file_name = file_name
        self.lam = lam
        self.node_num_factor = (self.node_num-1)/average_orders

    def exists(self):
        return Path(self.path_target_de).is_file()

    # Modified: Change Column names to match established instance schema
    def create_from_file(self, file_path):
        # 从文件中读取服务点、客户的位置 Read the location of service points and customers from a file.
        all_orders_df = pd.read_csv(file_path, header=0, names=[
                                    "CUST_NO", "XCOORD", "YCOORD", "DEMAND", "READYTIME", "DUETIME", "SERVICETIME", "CAPACITY", "VEHICLENO"])
        print(all_orders_df['CAPACITY'].iloc[0])
        vehicle_capacity = int(all_orders_df['CAPACITY'].iloc[0])
        vehicle_num = int(all_orders_df['CAPACITY'].iloc[0])
        # CUST_NO,XCOORD,YCOORD,DEMAND,READYTIME,DUETIME,SERVICETIME
        nodes = list(Node(int(row["CUST_NO"]) - 1, float(row["XCOORD"]), float(row["YCOORD"]), float(row["DEMAND"]), float(
            row["READYTIME"]), float(row["DUETIME"]), float(row["SERVICETIME"]), 0, 0, 0) for index, row in all_orders_df.iterrows())

        node_num = len(nodes)
        return node_num, nodes, vehicle_num, vehicle_capacity

    # Generation of random poisson distributed variables for 28 intervals
    def create_poisson_var(self):
        liste = []
        for i in range(1, 29):
            var = np.random.poisson(lam=self.lam*self.node_num_factor)
            liste.append(var)
        return liste

    # Creation of list containing available times
    def create_available_time(self):
        orders_per_interval = self.create_poisson_var()
        number_dynamic_nodes = sum(orders_per_interval)

        # List of due dates + indices as basis for available time
        due_date = []
        available_time = [0]
        availability_time_dynamic_nodes = []
        for i in range(len(self.nodes)):
            due_date.append(self.nodes[i].due_time)

        due_date[0] = 0
        due_date_idx = sorted(range(len(due_date)), key=lambda k: due_date[k])
        due_date.sort()

        for i in range(len(due_date)-number_dynamic_nodes-1):
            available_time.append(0)

        for i in range(len(due_date)-number_dynamic_nodes-1, len(self.nodes)-1):
            availability_time_dynamic_nodes.append(due_date[i])

        counter = -1
        available_time_new = []
        for i in range(self.time_slices-4):
            if orders_per_interval[i] == 0:
                continue
            elif orders_per_interval[i] != 0:
                j = 1
                while j <= orders_per_interval[i]:
                    counter += 1

                    # Nodes that do not have their due date in the next two time slices get current time slice as available time.
                    if availability_time_dynamic_nodes[counter] > (i+3) * self.interval_length:
                        available_time_new.append(
                            ((i+1)*(self.interval_length))-(self.interval_length-1))

                    # Nodes whose due date is after the 28th interval get the 28th interval as available time.
                    elif availability_time_dynamic_nodes[counter] > (self.time_slices-4) * self.interval_length:
                        available_time_new.append(
                            ((self.time_slices-4)*(self.interval_length))-(self.interval_length-1))

                    # Nodes whose due date lies in the next 2 time slices receive the penultimate time slice before
                    # their due time as available time if the current time slice it >1.
                    elif availability_time_dynamic_nodes[counter] <= (i+3) * self.interval_length:
                        time_slice_node = availability_time_dynamic_nodes[counter]//self.interval_length

                        if i > 1:
                            available_time_new.append(
                                ((time_slice_node - 2) * (self.interval_length)) - (self.interval_length - 1))

                        else:
                            available_time_new.append(0)

                    j += 1

        available_time += available_time_new

        available_time_sorted_idx = []
        for i in range(len(due_date_idx)):
            idx = due_date_idx.index(i)
            available_time_sorted_idx.append(available_time[idx])

        return available_time_sorted_idx

    # Create lists as basis for columns in DataFrame.
    def create_lists(self):
        id = []
        xcoord = []
        ycoord = []
        demand = []
        ready_time = []
        due_date = []
        service_time = []
        available_time = self.create_available_time()
        xcoord_end = []
        ycoord_end = []
        for i in range(len(self.nodes)):
            id.append(self.nodes[i].id)
            xcoord.append(self.nodes[i].x)
            ycoord.append(self.nodes[i].y)
            demand.append(self.nodes[i].demand)
            ready_time.append(self.nodes[i].ready_time)
            due_date.append(self.nodes[i].due_time)
            service_time.append(self.nodes[i].service_time)
            xcoord_end.append(self.nodes[i].x)
            ycoord_end.append(self.nodes[i].y)
        return id, xcoord, ycoord, demand, ready_time, due_date, service_time, available_time, xcoord_end, ycoord_end

    # Create DataFrame with additional columns
    def create_df(self):
        id, xcoord, ycoord, demand, ready_time, due_date, service_time, available_time, xcoord_end, ycoord_end = self.create_lists()
        df = pd.DataFrame(list(zip(id, xcoord, ycoord, demand, ready_time, due_date, service_time, available_time,
                                   xcoord_end, ycoord_end)), columns=['CUST_NO', 'XCOORD', 'YCOORD', 'DEMAND',
                                                                      'READYTIME', 'DUETIME', 'SERVICETIME',
                                                                      'AVAILABLETIME', 'XCOORD_END', 'YCOORD_END'])
        return df

    # Export .txt file.
    def export(self):
        df = self.create_df()
        df.to_csv(self.path_target_de)

    # Modified: Only run dataextension, if file doesn't already exists
    def rundataextension(self):
        if(not self.exists()):
            self.export()

        else:
            print('-----FILE ALREADY EXISTS, no action taken-----')
        print('-----DATA EXTENSION FINALIZED-----')
        return self.vehicle_capacity, self.path_target_de
