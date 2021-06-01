from aco.vrptw_base import VrptwGraph
from aco.multiple_ant_colony_system import MultipleAntColonySystem
from aco.update_process import UpdateProcess
import os
import pandas as pd
from aco.data_preparation_r import DataPreparation
from aco.pre_processing import PreProcessing
import time
from aco.data_extension_t import DataExtension
from aco.analysis_available_time import AnalysisAvailableTime
from vns.src.helpers.DistanceMatrix import DistanceMatrix

class Execution():
    def __init__(self):
        
        # Input data overall
        self.source = 'r'  # 'r': realworld test instance, 't': standard test instance
        self.ants_num = 100
        self.alpha = 1  # new
        self.beta = 1  # 2
        self.q0 = 0.9  # 0.5
        self.total_given_time = 15
        self.show_figure = True
        self.folder_name_handover = 'aco/handover'
        self.file_name_handover = 'handover.txt'

        # Input data realworld instance Chargery
        self.shift_begin = '09:00:00'
        self.order_reception_end = '15:59:59'
        self.shift_end = '17:00:00'
        self.shift_length = 480  # 8 hours = 480
        self.capacity = 480
        self.interval_length = 15  # minutes
        self.minutes_per_km = 2
        self.vehicles = 25
        self.location_change = ['refuel', 'car_wash', 'truck_wash', 'stationary_charge']

        self.folder_name_original_dp = 'aco/files_unprepared'
        self.folder_name_target_dp = 'aco/files_prepared'
        self.folder_name_original_pp = self.folder_name_target_dp
        self.folder_name_additional_data = 'aco/additional_data'
        self.folder_name_map = 'aco/additional_data'

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

    def intervals(self, shift_length, working_day_intervals):
        if shift_length % working_day_intervals != 0:
            return False
        else:
            return True


    def run_data_prep(self):
        if self.source == 't':

            # Run Analysis of Available Time
            print('-----ANALYSIS AVAILABLE TIME STARTED-----')
            aat = AnalysisAvailableTime(self.folder_name_analysis_available_time, self.intervals_order_reception, self.days_in_analysis)
            dynamicity, lam = aat.run_analysis()
            folder_name_target_pp = 'aco/testfiles-solomon-100'
            self.folder_name_testfile = folder_name_target_pp

            #  Run Data Extension, only run test files with identical shift duration at a time
            for file_name in os.listdir('aco/files_unprepared'):
                print('-----DATA EXTENSION STARTED, FILE', file_name, '-----')
                path_original_de = os.path.join(self.folder_name_original_dp, file_name)
                path_target_de = os.path.join(folder_name_target_pp, file_name.split('.')[0] + '-prepared.txt')
                de = DataExtension(path_original_de, path_target_de, file_name, dynamicity, lam, self.average_orders)
                self.shift_length = int(de.rundataextension())
                self.interval_length = self.shift_length/self.total_intervals

        elif self.source == 'r':
            # folder_name_target_pp = 'aco/testfiles-chargery'
            folder_name_target_pp = 'data/input_data_surve/orders'
            self.folder_name_testfile = folder_name_target_pp

            # #  Run Data Preparation
            # for file_name in os.listdir('aco/files_unprepared'):
            #     print('-----DATA PREPARATION STARTED, FILE', file_name, '-----')
            #     path_original_dp = os.path.join(self.folder_name_original_dp, file_name)
            #     path_target_dp = os.path.join(self.folder_name_target_dp, file_name.split('.')[0] + '-prepared.csv')
            #     path_hub_location = os.path.join(self.folder_name_additional_data, self.file_name_hub_location)
            #     dp = DataPreparation(path_original_dp, path_target_dp, self.shift_begin, self.order_reception_end, self.shift_end,
            #                         path_hub_location)
            #     dp.rundatapreparation()

            # #  Run Pre Processing
            # for file_name in os.listdir(self.folder_name_original_pp):
            #     print('-----PRE PROCESSING STARTED, FILE', file_name, '-----')
            #     path_original_pp = os.path.join(self.folder_name_original_pp, file_name)
            #     path_target_pp = os.path.join(folder_name_target_pp, file_name.split('.')[0] + '.txt')

            #     path_charging_station = os.path.join(self.folder_name_additional_data, self.file_name_charging_station)
            #     path_car_wash = os.path.join(self.folder_name_additional_data, self.file_name_car_wash)
            #     path_gas_station = os.path.join(self.folder_name_additional_data, self.file_name_gas_station)
            #     path_task_type_duration = os.path.join(self.folder_name_additional_data, self.file_name_task_type_duration)
            #     path_task_list = path_task_type_duration

            #     dp = PreProcessing(path_original_pp, path_target_pp, file_name, path_task_list, path_charging_station,
            #                     path_car_wash, path_gas_station, self.location_change, self.minutes_per_km,
            #                     path_task_type_duration, self.vehicles, self.capacity)
            #     dp.runpreprocessing()

        else:
            folder_name_target_pp = None 
            self.folder_name_testfile = None
            print('Source is required to be \'r\' (real world test instance) or \'t\' (standard test instance), folder_name_testfile is', self.folder_name_testfile, 'otherwise.')

    def run_macs(self):
        #  Run MACS and Update Process alternately, Update Process is last executed prior to the last execution of MACS
        if self.intervals(self.shift_length, self.interval_length):
            intervals = int(self.shift_length//self.interval_length)

            for file_name in os.listdir(self.folder_name_testfile):
                if not 'orders' in file_name:
                    continue

                dir_name = os.path.dirname(os.path.realpath('__file__'))
                current_file_name = file_name.split('.')[0][:-7]
                service_time_matrix = DistanceMatrix.load_file(os.path.join(
                dir_name, 'vns', 'data', 'results_preprocessing', current_file_name + '_service_times'))
                
                all_orders_df = pd.read_csv(self.folder_name_testfile+'/'+file_name)
                order_ids = all_orders_df['order_id'].values.tolist()
                
                folder_name_result = 'aco/results'
                if self.source == 'r':
                    folder_name_result += '/' + file_name[:10]
                elif self.source == 't':
                    file_name_folder = file_name.partition('-')[0]
                    folder_name_result += '/' + file_name_folder
                path_testfile = os.path.join(self.folder_name_testfile, file_name)
                folder_name_visualization = 'aco/visualization'
                path_visualization = os.path.join(folder_name_result, folder_name_visualization)
                try:
                    os.mkdir(folder_name_result)
                    try:
                        os.mkdir(path_visualization)
                    except OSError:
                        print("Creation of the directory %s failed, check if already in place!" % path_visualization)
                    else:
                        print("Successfully created the directory %s " % path_visualization)
                except OSError:
                    print("Creation of the directory %s failed, check if already in place!" % folder_name_result)
                else:
                    print("Successfully created the directory %s " % folder_name_result)

                path_handover = os.path.join(self.folder_name_handover, self.file_name_handover)
                path_map = os.path.join(self.folder_name_map, self.file_name_map)

                for i in range(0, intervals+1):
                    time_slice = i
                    graph = VrptwGraph(path_testfile, path_handover, time_slice, self.source, self.minutes_per_km, service_time_matrix, order_ids)
                    macs = MultipleAntColonySystem(graph, source=self.source, path_handover=path_handover, path_map=path_map, folder_name_result=folder_name_result, ants_num=self.ants_num, alpha=self.alpha, beta=self.beta, q0=self.q0,
                                                time_slice=time_slice, whether_or_not_to_show_figure=self.show_figure, service_time_matrix=service_time_matrix, order_ids=order_ids)
                    macs.run_multiple_ant_colony_system(total_given_time=self.total_given_time)

                    if i == intervals:
                        break

                    else:
                        print('-----UPDATE PROCESS STARTED BEFORE TIME SLICE', time_slice+1, '-----')
                        up = UpdateProcess(graph, path_handover=path_handover, time_slice=time_slice,
                                        interval_length=self.interval_length, path_testfile=path_testfile, source=self.source,
                                        minutes_per_km=self.minutes_per_km, service_time_matrix=service_time_matrix, order_ids=order_ids)
                        up.runupdateprocess()
                        time.sleep(5)

        else:
            print("ERROR: working_day_intervals must be divisor of shift_length")


execution = Execution()
execution.run_data_prep()
execution.run_macs()

