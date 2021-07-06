'''
Original Code provided by Nina Schwarm, (c) FU Berlin 2021
Modified by Lucas Merker & Marlene Buhl
'''
import pandas as pd
import numpy as np
from datetime import datetime, date
import vns.src.config.preprocessing_config as cfg
import os
from pathlib import Path


class DataPreparation:
    def __init__(self, base_dir, original_file):
        self.base_dir = base_dir
        self.target_folder = os.path.join(
            self.base_dir, "data", "results_data_preparation")
        Path(self.target_folder).mkdir(parents=True, exist_ok=True)
        self.original_file_path = os.path.join(
            self.base_dir, "data", "raw_data", original_file + '.csv')
        self.target_file_path = os.path.join(
            self.target_folder, original_file + '.csv')
        self.shift_begin = pd.to_datetime(
            cfg.shift_begin, format='%H:%M:%S').time()
        self.order_reception_end = pd.to_datetime(
            cfg.order_reception_end, format='%H:%M:%S').time()
        self.shift_end = pd.to_datetime(
            cfg.shift_end, format='%H:%M:%S').time()
        hub = pd.read_csv(os.path.join(
            self.base_dir, "data", "locations", "hub_location.csv"))
        self.x_depot = hub.iloc[0, 1]
        self.y_depot = hub.iloc[0, 0]
        self.df = pd.read_csv(self.original_file_path,
                              sep=',',
                              header=0,
                              names=['date',
                                     'time',
                                     'cancel_time',
                                     'order_id',
                                     'task_id',
                                     'task_type',
                                     'XCOORD.',  # lon
                                     'YCOORD.'])  # lat

    @staticmethod
    def availabletime(df, shift_begin, order_reception_end):
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
        df['AVAILABLE TIME'] = np.nan
        df['AVAILABLE TIME'] = np.where(((df['time'] > order_reception_end) | (df['time'] < shift_begin)), 0,
                                        df['time'])
        time_col = df.columns.get_loc('time')
        availible_time_col = df.columns.get_loc('AVAILABLE TIME')
        for i in range(len(df)):
            if shift_begin < df.iloc[i, time_col] < order_reception_end:
                df.iloc[i, availible_time_col] = round((datetime.combine(date.today(), df.iloc[i, time_col]) -
                                                        datetime.combine(date.today(), shift_begin)).total_seconds() / 60)

    # Modified: Included calculation of due date, if cancallation date is given
    @staticmethod
    def duetime(df, shift_begin, shift_end):
        df['cancel_time'] = pd.to_datetime(
            df['cancel_time'], format='%H:%M:%S', errors='coerce').dt.time
        df['DUE DATE'] = np.nan
        cancel_col = df.columns.get_loc("cancel_time")
        col = df.columns.get_loc("DUE DATE")
        for i in range(len(df)):
            if df.iloc[i, col]:
                if not pd.isnull(df.iloc[i, cancel_col]):
                    df.iloc[i, col] = round((datetime.combine(date.today(), df.iloc[i, cancel_col]) -
                                             datetime.combine(date.today(), shift_begin)).total_seconds() / 60)
                else:
                    df.iloc[i, col] = round((datetime.combine(date.today(), shift_end) -
                                             datetime.combine(date.today(), shift_begin)).total_seconds() / 60)
        df['DUE DATE'] = df['DUE DATE'].astype(int)

    @ staticmethod
    def readytime(df):
        df['READY TIME'] = df['AVAILABLE TIME']

    @ staticmethod
    def drop_unused_columns(df):
        df.drop(['date', 'time', 'cancel_time'], axis=1, inplace=True)

    def exportcsv(self, df, target_file_path, shift_end, shift_begin):
        df1 = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
        df = df1.append(df, ignore_index=True)
        df.iloc[0, df.columns.get_loc('XCOORD.')] = self.x_depot
        df.iloc[0, df.columns.get_loc('YCOORD.')] = self.y_depot
        df.iloc[0, df.columns.get_loc('DUE DATE')] = round((datetime.combine(date.today(), shift_end) -
                                                            datetime.combine(date.today(), shift_begin)).total_seconds() / 60)
        df.to_csv(target_file_path, header=True)

    def rundatapreparation(self):
        self.availabletime(self.df, self.shift_begin, self.order_reception_end)
        self.duetime(self.df, self.shift_begin, self.shift_end)
        self.readytime(self.df)
        self.drop_unused_columns(self.df)
        self.exportcsv(self.df, self.target_file_path,
                       self.shift_end, self.shift_begin)
        print('-----DATA PREPARATION FINALIZED-----')
