from aco.vrptw_base import VrptwGraph
from aco.multiple_ant_colony_system import MultipleAntColonySystem
from aco.update_process import UpdateProcess
import os
from aco.data_preparation_r import DataPreparation
from aco.pre_processing import PreProcessing
import time
from aco.data_extension_t import DataExtension
from aco.analysis_available_time import AnalysisAvailableTime


# Input data overall
source = 'r'  # 'r': realworld test instance, 't': standard test instance
ants_num = 100
alpha = 1  # new
beta = 1  # 2
q0 = 0.9  # 0.5
total_given_time = 15
show_figure = True
folder_name_handover = './handover'
file_name_handover = 'handover.txt'

# Input data realworld instance Chargery
shift_begin = '09:00:00'
order_reception_end = '15:59:59'
shift_end = '17:00:00'
shift_length = 480  # 8 hours = 480
capacity = 480
interval_length = 15  # minutes
minutes_per_km = 2
vehicles = 25
location_change = ['refuel', 'car_wash', 'truck_wash', 'stationary_charge']

folder_name_original_dp = './files_unprepared'
folder_name_target_dp = './files_prepared'
folder_name_original_pp = folder_name_target_dp
folder_name_additional_data = './additional_data'
folder_name_map = './additional_data'

file_name_hub_location = 'hub_location.csv'
file_name_charging_station = 'charging_station_locations.csv'
file_name_car_wash = 'car_wash_locations.csv'
file_name_gas_station = 'gas_station_locations.csv'
file_name_task_type_duration = 'task_type_duration.csv'
file_name_map = 'berlin.shp'


# Input data standard test instance
intervals_order_reception = 28
total_intervals = 32
days_in_analysis = 232
average_orders = 66.93
folder_name_analysis_available_time = './additional_data/analysis_available_time'



def intervals(shift_length, working_day_intervals):
    if shift_length % working_day_intervals != 0:
        return False
    else:
        return True

if source == 't':

    # Run Analysis of Available Time
    print('-----ANALYSIS AVAILABLE TIME STARTED-----')
    aat = AnalysisAvailableTime(folder_name_analysis_available_time, intervals_order_reception, days_in_analysis)
    dynamicity, lam = aat.run_analysis()
    folder_name_target_pp = './testfiles-solomon-100'
    folder_name_testfile = folder_name_target_pp

    #  Run Data Extension, only run test files with identical shift duration at a time
    for file_name in os.listdir('./files_unprepared'):
        print('-----DATA EXTENSION STARTED, FILE', file_name, '-----')
        path_original_de = os.path.join(folder_name_original_dp, file_name)
        path_target_de = os.path.join(folder_name_target_pp, file_name.split('.')[0] + '-prepared.txt')
        de = DataExtension(path_original_de, path_target_de, file_name, dynamicity, lam, average_orders)
        shift_length = int(de.rundataextension())
        interval_length = shift_length/total_intervals

elif source == 'r':
    # folder_name_target_pp = './testfiles-chargery-original'
    # TODO: fix relative routing (if needed)
    # folder_name_target_pp = './testfiles-chargery-new'
    folder_name_target_pp = 'aco/testfiles-chargery-new'
    folder_name_testfile = folder_name_target_pp

    #  Run Data Preparation
    # TODO: fix relative routing (if needed)
    for file_name in os.listdir('aco/files_unprepared'):
        print('-----DATA PREPARATION STARTED, FILE', file_name, '-----')
        path_original_dp = os.path.join(folder_name_original_dp, file_name)
        path_target_dp = os.path.join(folder_name_target_dp, file_name.split('.')[0] + '-prepared.csv')
        path_hub_location = os.path.join(folder_name_additional_data, file_name_hub_location)
        dp = DataPreparation(path_original_dp, path_target_dp, shift_begin, order_reception_end, shift_end,
                                path_hub_location)
        dp.rundatapreparation()

    # #  Run Pre Processing
    # for file_name in os.listdir(folder_name_original_pp):
    #     print('-----PRE PROCESSING STARTED, FILE', file_name, '-----')
    #     path_original_pp = os.path.join(folder_name_original_pp, file_name)
    #     path_target_pp = os.path.join(folder_name_target_pp, file_name.split('.')[0] + '.txt')

    #     path_charging_station = os.path.join(folder_name_additional_data, file_name_charging_station)
    #     path_car_wash = os.path.join(folder_name_additional_data, file_name_car_wash)
    #     path_gas_station = os.path.join(folder_name_additional_data, file_name_gas_station)
    #     path_task_type_duration = os.path.join(folder_name_additional_data, file_name_task_type_duration)
    #     path_task_list = path_task_type_duration

    #     dp = PreProcessing(path_original_pp, path_target_pp, file_name, path_task_list, path_charging_station,
    #                        path_car_wash, path_gas_station, location_change, minutes_per_km,
    #                        path_task_type_duration, vehicles, capacity)
    #     dp.runpreprocessing()

else:
    folder_name_target_pp = None
    folder_name_testfile = None
    print('Source is required to be \'r\' (real world test instance) or \'t\' (standard test instance), folder_name_testfile is', folder_name_testfile, 'otherwise.')


#  Run MACS and Update Process alternately, Update Process is last executed prior to the last execution of MACS
if intervals(shift_length, interval_length):
    intervals = int(shift_length//interval_length)

    for file_name in os.listdir(folder_name_testfile):
        folder_name_result = './results'
        if source == 'r':
            folder_name_result += '/' + file_name[:10]
        elif source == 't':
            file_name_folder = file_name.partition('-')[0]
            folder_name_result += '/' + file_name_folder
        path_testfile = os.path.join(folder_name_testfile, file_name)
        folder_name_visualization = './visualization'
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

        path_handover = os.path.join(folder_name_handover, file_name_handover)
        path_map = os.path.join(folder_name_map, file_name_map)

        for i in range(0, intervals+1):
            time_slice = i
            graph = VrptwGraph(path_testfile, path_handover, time_slice, source, minutes_per_km)
            macs = MultipleAntColonySystem(graph, source=source, path_handover=path_handover, path_map=path_map, folder_name_result=folder_name_result, ants_num=ants_num, alpha=alpha, beta=beta, q0=q0,
                                            time_slice=time_slice, whether_or_not_to_show_figure=show_figure)
            macs.run_multiple_ant_colony_system(total_given_time=total_given_time)

            if i == intervals:
                break

            else:
                print('-----UPDATE PROCESS STARTED BEFORE TIME SLICE', time_slice+1, '-----')
                up = UpdateProcess(graph, path_handover=path_handover, time_slice=time_slice,
                                    interval_length=interval_length, path_testfile=path_testfile, source=source,
                                    minutes_per_km=minutes_per_km)
                up.runupdateprocess()
                time.sleep(5)

    print('')

else:
    print("ERROR: working_day_intervals must be divisor of shift_length")



# def run_data_prep():
#     if source == 't':

#         # Run Analysis of Available Time
#         print('-----ANALYSIS AVAILABLE TIME STARTED-----')
#         aat = AnalysisAvailableTime(folder_name_analysis_available_time, intervals_order_reception, days_in_analysis)
#         dynamicity, lam = aat.run_analysis()
#         folder_name_target_pp = './testfiles-solomon-100'
#         folder_name_testfile = folder_name_target_pp

#         #  Run Data Extension, only run test files with identical shift duration at a time
#         for file_name in os.listdir('./files_unprepared'):
#             print('-----DATA EXTENSION STARTED, FILE', file_name, '-----')
#             path_original_de = os.path.join(folder_name_original_dp, file_name)
#             path_target_de = os.path.join(folder_name_target_pp, file_name.split('.')[0] + '-prepared.txt')
#             de = DataExtension(path_original_de, path_target_de, file_name, dynamicity, lam, average_orders)
#             shift_length = int(de.rundataextension())
#             interval_length = shift_length/total_intervals

#     elif source == 'r':
#         # folder_name_target_pp = './testfiles-chargery-original'
#         # TODO: fix relative routing (if needed)
#         # folder_name_target_pp = './testfiles-chargery-new'
#         folder_name_target_pp = 'aco/VRPTW-ACO-python/testfiles-chargery-new'
#         folder_name_testfile = folder_name_target_pp

#         #  Run Data Preparation
#         # TODO: fix relative routing (if needed)
#         for file_name in os.listdir('aco/VRPTW-ACO-python/files_unprepared'):
#             print('-----DATA PREPARATION STARTED, FILE', file_name, '-----')
#             path_original_dp = os.path.join(folder_name_original_dp, file_name)
#             path_target_dp = os.path.join(folder_name_target_dp, file_name.split('.')[0] + '-prepared.csv')
#             path_hub_location = os.path.join(folder_name_additional_data, file_name_hub_location)
#             dp = DataPreparation(path_original_dp, path_target_dp, shift_begin, order_reception_end, shift_end,
#                                  path_hub_location)
#             dp.rundatapreparation()

#         # #  Run Pre Processing
#         # for file_name in os.listdir(folder_name_original_pp):
#         #     print('-----PRE PROCESSING STARTED, FILE', file_name, '-----')
#         #     path_original_pp = os.path.join(folder_name_original_pp, file_name)
#         #     path_target_pp = os.path.join(folder_name_target_pp, file_name.split('.')[0] + '.txt')

#         #     path_charging_station = os.path.join(folder_name_additional_data, file_name_charging_station)
#         #     path_car_wash = os.path.join(folder_name_additional_data, file_name_car_wash)
#         #     path_gas_station = os.path.join(folder_name_additional_data, file_name_gas_station)
#         #     path_task_type_duration = os.path.join(folder_name_additional_data, file_name_task_type_duration)
#         #     path_task_list = path_task_type_duration

#         #     dp = PreProcessing(path_original_pp, path_target_pp, file_name, path_task_list, path_charging_station,
#         #                        path_car_wash, path_gas_station, location_change, minutes_per_km,
#         #                        path_task_type_duration, vehicles, capacity)
#         #     dp.runpreprocessing()

#     else:
#         folder_name_target_pp = None
#         folder_name_testfile = None
#         print('Source is required to be \'r\' (real world test instance) or \'t\' (standard test instance), folder_name_testfile is', folder_name_testfile, 'otherwise.')

# def intervals(shift_length, working_day_intervals):
#     if shift_length % working_day_intervals != 0:
#         return False
#     else:
#         return True


# def run_macs():
#     #  Run MACS and Update Process alternately, Update Process is last executed prior to the last execution of MACS
#     if intervals(shift_length, interval_length):
#         intervals = int(shift_length//interval_length)

#         for file_name in os.listdir(folder_name_testfile):
#             folder_name_result = './results'
#             if source == 'r':
#                 folder_name_result += '/' + file_name[:10]
#             elif source == 't':
#                 file_name_folder = file_name.partition('-')[0]
#                 folder_name_result += '/' + file_name_folder
#             path_testfile = os.path.join(folder_name_testfile, file_name)
#             folder_name_visualization = './visualization'
#             path_visualization = os.path.join(folder_name_result, folder_name_visualization)
#             try:
#                 os.mkdir(folder_name_result)
#                 try:
#                     os.mkdir(path_visualization)
#                 except OSError:
#                     print("Creation of the directory %s failed, check if already in place!" % path_visualization)
#                 else:
#                     print("Successfully created the directory %s " % path_visualization)
#             except OSError:
#                 print("Creation of the directory %s failed, check if already in place!" % folder_name_result)
#             else:
#                 print("Successfully created the directory %s " % folder_name_result)

#             path_handover = os.path.join(folder_name_handover, file_name_handover)
#             path_map = os.path.join(folder_name_map, file_name_map)

#             for i in range(0, intervals+1):
#                 time_slice = i
#                 graph = VrptwGraph(path_testfile, path_handover, time_slice, source, minutes_per_km)
#                 macs = MultipleAntColonySystem(graph, source=source, path_handover=path_handover, path_map=path_map, folder_name_result=folder_name_result, ants_num=ants_num, alpha=alpha, beta=beta, q0=q0,
#                                                time_slice=time_slice, whether_or_not_to_show_figure=show_figure)
#                 macs.run_multiple_ant_colony_system(total_given_time=total_given_time)

#                 if i == intervals:
#                     break

#                 else:
#                     print('-----UPDATE PROCESS STARTED BEFORE TIME SLICE', time_slice+1, '-----')
#                     up = UpdateProcess(graph, path_handover=path_handover, time_slice=time_slice,
#                                        interval_length=interval_length, path_testfile=path_testfile, source=source,
#                                        minutes_per_km=minutes_per_km)
#                     up.runupdateprocess()
#                     time.sleep(5)

#         print('')

#     else:
#         print("ERROR: working_day_intervals must be divisor of shift_length")



# def main():
#     # Input data overall
#     source = 'r'  # 'r': realworld test instance, 't': standard test instance
#     ants_num = 100
#     alpha = 1  # new
#     beta = 1  # 2
#     q0 = 0.9  # 0.5
#     total_given_time = 15
#     show_figure = True
#     folder_name_handover = './handover'
#     file_name_handover = 'handover.txt'

#     # Input data realworld instance Chargery
#     shift_begin = '09:00:00'
#     order_reception_end = '15:59:59'
#     shift_end = '17:00:00'
#     shift_length = 480  # 8 hours = 480
#     capacity = 480
#     interval_length = 15  # minutes
#     minutes_per_km = 2
#     vehicles = 25
#     location_change = ['refuel', 'car_wash', 'truck_wash', 'stationary_charge']

#     folder_name_original_dp = './files_unprepared'
#     folder_name_target_dp = './files_prepared'
#     folder_name_original_pp = folder_name_target_dp
#     folder_name_additional_data = './additional_data'
#     folder_name_map = './additional_data'

#     file_name_hub_location = 'hub_location.csv'
#     file_name_charging_station = 'charging_station_locations.csv'
#     file_name_car_wash = 'car_wash_locations.csv'
#     file_name_gas_station = 'gas_station_locations.csv'
#     file_name_task_type_duration = 'task_type_duration.csv'
#     file_name_map = 'berlin.shp'


#     # Input data standard test instance
#     intervals_order_reception = 28
#     total_intervals = 32
#     days_in_analysis = 232
#     average_orders = 66.93
#     folder_name_analysis_available_time = './additional_data/analysis_available_time'

#     run_data_prep()
#     run_macs()


# if __name__ == '__main__':
#     main()