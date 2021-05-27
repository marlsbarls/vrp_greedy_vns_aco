import pandas as pd
import numpy as np
from aco.vrptw_base import Node


class DataExtension:
    def __init__(self, path_original_de, path_target_de, file_name, dynamicity, lam, average_orders):
        self.path_target_de = path_target_de
        self.dynamicity = dynamicity
        self.node_num, self.nodes, self.vehicle_num, _ = self.create_from_file(path_original_de)
        self.vehicle_capacity = int(self.nodes[0].due_time)
        self.time_slices = 32
        self.interval_length = self.vehicle_capacity / self.time_slices
        self.file_name = file_name
        self.lam = lam
        self.node_num_factor = (self.node_num-1)/average_orders

    def create_from_file(self, file_path):
        # 从文件中读取服务点、客户的位置 Read the location of service points and customers from a file.
        node_list = []

        with open(file_path, 'rt') as f:
            count = 1
            for line in f:
                if count == 5:
                    vehicle_num, vehicle_capacity = line.split()
                    vehicle_num = int(vehicle_num)
                    vehicle_capacity = int(vehicle_capacity)
                elif count >= 10:
                    node_list.append(line.split())
                count += 1

        nodes = list(
            Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]),
                 float(item[6]), 0, 0, 0) for item in node_list)

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
        due_date_idx = sorted(range(len(due_date)), key=lambda k:due_date[k])
        due_date.sort()

        for i in range(len(due_date)-number_dynamic_nodes-1):
            available_time.append(0)

        for i in range(len(due_date)-number_dynamic_nodes-1,len(self.nodes)-1):
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
                        available_time_new.append(((i+1)*(self.interval_length))-(self.interval_length-1))

                    # Nodes whose due date is after the 28th interval get the 28th interval as available time.
                    elif availability_time_dynamic_nodes[counter] > (self.time_slices-4) * self.interval_length:
                        available_time_new.append(((self.time_slices-4)*(self.interval_length))-(self.interval_length-1))

                    # Nodes whose due date lies in the next 2 time slices receive the penultimate time slice before
                    # their due time as available time if the current time slice it >1.
                    elif availability_time_dynamic_nodes[counter] <= (i+3) * self.interval_length:
                        time_slice_node = availability_time_dynamic_nodes[counter]//self.interval_length

                        if i > 1:
                            available_time_new.append(((time_slice_node - 2) * (self.interval_length)) - (self.interval_length - 1))

                        else:
                            available_time_new.append(0)

                    j +=1

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
                                   xcoord_end, ycoord_end)), columns=['CUST_NO.', 'XCOORD.', 'YCOORD.', 'DEMAND',
                                                                   'READY_TIME', 'DUE_DATE', 'SERVICE_TIME',
                                                                   'AVAILABLE_TIME', 'XCOORD_END', 'YCOORD_END'])
        return df

    # Create header of export file.
    def create_file(self):
        f = open(self.path_target_de, 'w+')
        title = self.file_name.partition(".")[0]
        vehicles = self.vehicle_num
        capacity = int(self.nodes[0].due_time)
        f.write('{}\n\nVEHICLE\nNUMBER\tCAPACITY\n{}\t{}\n\nCUSTOMER\n'.format(title, vehicles, capacity))
        f.close()

    # Export .txt file.
    def export(self):
        self.create_file()
        df = self.create_df()
        f = open(self.path_target_de, 'a+')
        f.write('{}'.format(df.to_string(columns=(
            'CUST_NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY_TIME', 'DUE_DATE', 'SERVICE_TIME', 'AVAILABLE_TIME',
            'XCOORD_END', 'YCOORD_END'), index=False, justify='center')))
        f.close()


    def rundataextension(self):
        self.export()
        print('-----DATA EXTENSION FINALIZED-----')
        return self.nodes[0].due_time
