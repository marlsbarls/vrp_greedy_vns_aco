import pandas as pd
import numpy as np
from datetime import datetime, date


class DataPreparation:
    def __init__(self, original_file_path, target_file_path, shift_begin, order_reception_end, shift_end,
                 hub_location_file):
        self.original_file_path = original_file_path
        self.target_file_path = target_file_path
        self.shift_begin = pd.to_datetime(shift_begin, format='%H:%M:%S').time()
        self.order_reception_end = pd.to_datetime(order_reception_end, format='%H:%M:%S').time()
        self.shift_end = pd.to_datetime(shift_end, format='%H:%M:%S').time()
        self.hub_location_file = hub_location_file

    # Add column AVAILABLE TIME, fill it according to time and shift/order reception times.
    def available_time(self, df):
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
        df['AVAILABLE TIME'] = np.nan
        df['AVAILABLE TIME'] = np.where(((df['time'] > self.order_reception_end) | (df['time'] < self.shift_begin)), 0,
                                        df['time'])

        for i in range(len(df)):
            if self.shift_begin <= df.iloc[i, 1] < self.order_reception_end:
                df.iloc[i, 7] = round((datetime.combine(date.today(), df.iloc[i, 1]) -
                                       datetime.combine(date.today(), self.shift_begin)).total_seconds() / 60)

        return df

    # Add column DUE DATE and fill with shift duration in minutes. To be adapted once due time can be exported from data
    # in standardized form.
    def due_time(self, df):
        df['DUE DATE'] = np.nan

        for i in range(len(df)):
            if df.iloc[i, 8]:
                df.iloc[i, 8] = round((datetime.combine(date.today(), self.shift_end) -
                                       datetime.combine(date.today(), self.shift_begin)).total_seconds() / 60)
        df['DUE DATE'] = df['DUE DATE'].astype(int)

        return df

    # Add column READY TIME and fill with values from AVAILABLE TIME. To be adapted once ready time can be exported from
    # data in standardized form.
    def ready_time(self, df):
        df['READY TIME'] = df['AVAILABLE TIME']

        return df

    # Drop columns date and time
    def new_df(self, df):
        df.drop(['date', 'time'], axis=1, inplace=True)

        return df

    # Get Depot Nodes
    def get_depot(self, hub_location_file):
        hub = pd.read_csv(hub_location_file)
        x_depot = hub.iloc[0, 0]
        y_depot = hub.iloc[0, 1]

        return x_depot, y_depot

    # Add Depot to Dataframe
    def add_depot(self, df):
        df_help = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
        df = df_help.append(df, ignore_index=True)
        x_depot, y_depot = self.get_depot(self.hub_location_file)

        for column in df:
            if column == 'XCOORD.':
                idx = df.columns.get_loc(column)
                df.iloc[0, idx] = x_depot

            if column == 'YCOORD.':
                idx = df.columns.get_loc(column)
                df.iloc[0, idx] = y_depot

            if column == 'DUE DATE':
                idx = df.columns.get_loc(column)
                df.iloc[0, idx] = round((datetime.combine(date.today(), self.shift_end) -
                                         datetime.combine(date.today(), self.shift_begin)).total_seconds() / 60)

        return df

    # Build Dataframe out of export file and manipulate
    def build_df(self):
        df = pd.read_csv(self.original_file_path,
                         sep=',',
                         header=0,
                         names=['date',
                                'time',
                                'order_id',
                                'task_id',
                                'task_type',
                                'XCOORD.',  # lat
                                'YCOORD.'])  # lon

        df = self.available_time(df)
        df = self.due_time(df)
        df = self.ready_time(df)
        df = self.new_df(df)
        df = self.add_depot(df)

        return df

    # Export manipulated Dataframe
    def export_csv(self):
        df = self.build_df()
        df.to_csv(self.target_file_path, header=True)

    # Run Data Preparation through starting export method
    def rundatapreparation(self):
        self.export_csv()
        print('-----DATA PREPARATION FINALIZED-----')
