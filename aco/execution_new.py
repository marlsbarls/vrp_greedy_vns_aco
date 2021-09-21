from aco.vrptw_base import VrptwGraph
from aco.multiple_ant_colony_system import MultipleAntColonySystem
from aco.update_process import UpdateProcess
import os
from pathlib import Path
import pandas as pd
import time
from datetime import date
from aco.analysis_available_time import AnalysisAvailableTime
from vns.src.helpers.DistanceMatrix import DistanceMatrix


class Execution():
    def __init__(self, test_files, dynamic, data_type):
        # MOD: Marlene
        self.test_files = test_files
        self.dynamic = dynamic
        self.data_type = data_type
        if self.data_type == 'surve_mobility':
            self.source = 'r'
        if self.data_type == 'solomon':
            self.source = 't'
        
        # Input data overall
        # self.source = 'r'  # 'r': realworld test instance, 't': standard test instance
        self.ants_num = 100
        self.alpha = 1  # new
        self.beta = 1  # 2
        self.q0 = 0.9  # 0.5
        self.total_given_time = 15 # original
        if self.dynamic == 'static': 
            self.total_given_time = 480
        # self.total_given_time = 15
        # self.show_figure = True
        self.show_figure = False
        self.folder_name_handover = 'aco/handover'
        self.file_name_handover = 'handover.txt'


        # Input data realworld instance Chargery
        self.shift_begin = '09:00:00'
        self.order_reception_end = '15:59:59'
        self.shift_end = '17:00:00'
        self.shift_length = 480  # 8 hours = 480
        self.capacity = 480
        self.interval_length = 15  # minutes
        if self.dynamic == 'static':
            self.interval_length = 480  # minutes
        # self.minutes_per_km = 2
        self.minutes_per_km = 1
        self.vehicles = 25
        
        # # quick test
        # self.total_given_time = 1
        # self.interval_length = 1

        # Mod: Marlene 
        # opt_time = True minimizes travel time instead of travel distance
        # original = False
        self.opt_time = True 
        # self.opt_time = False
        if self.source == 't':
            self.opt_time = False

        ###
        # self.folder_name_original_dp = 'aco/files_unprepared'
        # self.folder_name_target_dp = 'aco/files_prepared'
        # self.folder_name_original_pp = self.folder_name_target_dp
        # self.folder_name_additional_data = 'aco/additional_data'
        ###

        self.folder_name_map = 'input_data/data_preparation/berlin_maps'

        self.file_name_hub_location = 'hub_location.csv'
        self.file_name_charging_station = 'charging_station_locations.csv'
        self.file_name_car_wash = 'car_wash_locations.csv'
        self.file_name_gas_station = 'gas_station_locations.csv'
        self.file_name_task_type_duration = 'task_type_duration.csv'
        self.file_name_map = 'berlin.shp'

        # Input data standard test instance
        self.intervals_order_reception = 28
        self.total_intervals = 32
        self.days_in_analysis = 232
        self.average_orders = 66.93
        self.folder_name_analysis_available_time = 'aco/additional_data/analysis_available_time'

        self.parameters = {                                      
                    'number of ants': self.ants_num,
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'q0': self.q0,
                    'shift length': self.shift_length,
                    'capacity': self.capacity,
                    'interval length': self.interval_length,  
                    'minutes per km': self.minutes_per_km,
                    'vehicles': self.vehicles,
                    'optimize for time': self.opt_time
        }

    def intervals(self, shift_length, working_day_intervals):
        if shift_length % working_day_intervals != 0:
            return False
        else:
            return True

    def run_macs(self):
        # Input files
        if self.source == 'r':
            folder_name_target_pp = 'input_data/surve_mobility/orders'
            self.folder_name_testfile = folder_name_target_pp
        
        if self.source == 't':
            folder_name_target_pp = 'input_data/solomon/orders'
            self.folder_name_testfile = folder_name_target_pp
           
        self.result_df = pd.DataFrame(columns=[
        'parameters', 'cost per driver', 'cost per hour', 'final_cost', 'idle_time', 'tour_length', 'vehicle_number', 'runtime', 'final_tour'])
        #  Run MACS and Update Process alternately, Update Process is last executed prior to the last execution of MACS
        if self.intervals(self.shift_length, self.interval_length):
            intervals = int(self.shift_length//self.interval_length)

            for file_name in os.listdir(self.folder_name_testfile):
                if not 'orders' in file_name:
                    continue
                if not file_name.split('_')[0] in self.test_files:
                    continue

                print('---------------')
                print(file_name)
                print('---------------')

                dir_name = os.path.dirname(os.path.realpath('__file__'))
                current_file_name = file_name.split('.')[0][:-7]
                if self.source == 'r':
                    service_time_matrix = DistanceMatrix.load_file(os.path.join(
                    dir_name, 'input_data', 'surve_mobility', current_file_name + '_service_times'))
                
                    all_orders_df = pd.read_csv(self.folder_name_testfile+'/'+file_name)
                    order_ids = all_orders_df['order_id'].values.tolist()
                
                
                today = date.today()
                date_today = today.strftime("%b-%d-%Y")
                target_folder_results = os.path.join(
                dir_name, "results", "aco", self.data_type, file_name.partition('_')[0], self.dynamic, date_today)
                Path(target_folder_results).mkdir(parents=True, exist_ok=True)
                folder_name_result = target_folder_results

                path_testfile = os.path.join(self.folder_name_testfile, file_name)
                path_visualization = os.path.join(folder_name_result, 'visualization')
                Path(path_visualization).mkdir(parents=True, exist_ok=True)


                path_handover = os.path.join(self.folder_name_handover, self.file_name_handover)
                main_folder = os.path.dirname(os.path.realpath(__file__)).split('aco')[0]
                path_map = os.path.join(main_folder, self.folder_name_map, self.file_name_map)

                for i in range(0, intervals+1):
                    time_slice = i

                    if self.source == 'r':
                        graph = VrptwGraph(path_testfile, path_handover, time_slice, self.source, self.minutes_per_km, service_time_matrix, 
                                           order_ids, test_type=self.dynamic, opt_time=self.opt_time)
                        macs = MultipleAntColonySystem(graph, source=self.source, path_handover=path_handover, path_map=path_map, 
                                                       folder_name_result=folder_name_result, result_df=self.result_df, parameter=self.parameters, ants_num=self.ants_num, alpha=self.alpha, beta=self.beta, 
                                                       q0=self.q0, time_slice=time_slice, whether_or_not_to_show_figure=self.show_figure, 
                                                       service_time_matrix=service_time_matrix, order_ids=order_ids, opt_time=self.opt_time)
                    elif self.source == 't':
                        graph = VrptwGraph(path_testfile, path_handover, time_slice, self.source, self.minutes_per_km, 
                                           test_type=self.dynamic)
                        macs = MultipleAntColonySystem(graph, source=self.source, path_handover=path_handover, path_map=path_map,
                                                       folder_name_result=folder_name_result, result_df=self.result_df, parameter=self.parameters, ants_num=self.ants_num, alpha=self.alpha, 
                                                       beta=self.beta, q0=self.q0, time_slice=time_slice, whether_or_not_to_show_figure=self.show_figure)

                    macs.run_multiple_ant_colony_system(total_given_time=self.total_given_time)

                    if i == intervals:
                        break
                    # only execute one intervall when static
                    if self.dynamic == 'static':
                        break

                    else:
                        print('-----UPDATE PROCESS STARTED BEFORE TIME SLICE', time_slice+1, '-----')
                        if self.source == 'r':
                            up = UpdateProcess(graph, path_handover=path_handover, time_slice=time_slice,
                                               interval_length=self.interval_length, path_testfile=path_testfile, source=self.source,
                                               minutes_per_km=self.minutes_per_km, test_type=self.dynamic, 
                                               service_time_matrix=service_time_matrix, order_ids=order_ids, opt_time=self.opt_time)
                        elif self.source == 't':
                            up = UpdateProcess(graph, path_handover=path_handover, time_slice=time_slice,
                                               interval_length=self.interval_length, path_testfile=path_testfile, source=self.source,
                                               minutes_per_km=self.minutes_per_km, test_type=self.dynamic)
                        up.runupdateprocess()
                        time.sleep(5)

        else:
            print("ERROR: working_day_intervals must be divisor of shift_length")


# execution = Execution()
# execution.run_data_prep()
# execution.run_macs()

