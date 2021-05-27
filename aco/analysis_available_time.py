import os
import pandas as pd
import numpy as np


class AnalysisAvailableTime:
    def __init__(self, folder_prepared, intervals_order_reception, total_days):
        self.folder_prepared = folder_prepared
        self.intervals_order_reception = intervals_order_reception
        self.total_days = total_days
        self.all_available_times = []
        self.all_available_times_non_zero = []
        self.order_id_list = []

    def run_analysis(self):
        for file_name in os.listdir(self.folder_prepared):
            path_unprepared = os.path.join(self.folder_prepared, file_name)
            df = pd.read_csv(path_unprepared,
                             sep=',',
                             header=0,
                             names=['order_id',
                                    'task_id',
                                    'task_type',
                                    'XCOORD.',  # lat
                                    'YCOORD.',  # lon
                                    'AVAILABLE_TIME',
                                    'DUE_DATE',
                                    'READY_TIME'])

            for i in range(len(df)):
                if df.iloc[i, 0] not in self.order_id_list:
                    self.all_available_times.append(df.iloc[i, 5])
                    self.order_id_list.append(df.iloc[i, 0])
                else:
                    continue

            for i in self.all_available_times:
                if i > 0:
                    self.all_available_times_non_zero.append(i)
                else:
                    continue

            mean_intervals = len(self.all_available_times_non_zero) / self.total_days / self.intervals_order_reception
            print("Order reception mean per interval:", mean_intervals, ', this value serves as Lambda for generating poisson distributed variables.')

            len_all_available_times = len(self.all_available_times)
            available_time_zero = self.all_available_times.count(0)
            available_time_non_zero = len_all_available_times - available_time_zero
            dynamicity = available_time_non_zero / len_all_available_times

            total = 0
            for i in range(1, 1000):
                liste = []
                for i in range(1, 29):
                    var = np.random.poisson(lam=mean_intervals)
                    liste.append(var)
                total += sum(liste)
            print('1000 assignments of random poisson distributed variables to 28 intervals give a mean of', total / 1000 / 66.93, 'dynamic orders per average day (66.93 orders).')
            print('This applies approximately to the identified dynamicity of all days of 2020 of', dynamicity,'.')
            print('-----ANALYSIS AVAILABLE TIME FINALIZED-----')
            return dynamicity, mean_intervals