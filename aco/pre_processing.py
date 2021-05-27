import pandas as pd
import math
from math import radians, cos, sin, asin, sqrt
import numpy as np


class PreProcessing:
    def __init__(self, original_file_path, target_file_path, target_file_name, file_name_task_list,
                 file_name_charging_station, file_name_car_wash, file_name_gas_station, location_change,
                 minutes_per_km, file_name_task_type_duration, vehicles, capacity):
        self.original_file_path = original_file_path
        self.target_file_path = target_file_path
        self.target_file_name = target_file_name
        self.file_name_task_list = file_name_task_list
        self.file_name_charging_station = file_name_charging_station
        self.file_name_car_wash = file_name_car_wash
        self.file_name_gas_station = file_name_gas_station
        self.location_change = location_change
        self.minutes_per_km = minutes_per_km
        self.file_name_task_type_duration = file_name_task_type_duration
        self.vehicles = vehicles
        self.capacity = capacity

    # Build list of tasks
    def get_task_list(self):
        task_list = []
        df_task_list = pd.read_csv(self.file_name_task_list, usecols=['task_type'], header=0)

        for i in range(len(df_task_list)):
            task_list.append(df_task_list.iloc[i, 0])

        return task_list

    # Add columns Latitude, Longitude and Duration for every task in Dataframe, add columns XCOORD_END and YCOORD_END.
    def addcolumns(self, df):
        task_list = self.get_task_list()

        for col in task_list:
            df[col] = 0

            if col in self.location_change:
                idx = df.columns.get_loc(col) + 1
                df.insert(idx, 'Latitude', 0, True)
                df.insert(idx+1, 'Longitude', 0, True)
                df.insert(idx+2, 'Duration', 0, True)

            else:
                continue

        idx = len(df.columns)
        df.insert(idx, 'YCOORD_END', 0, True)
        df.insert(idx, 'XCOORD_END', 0, True)

        return df

    # Fill task columns, 1 if task in order, 0 otherwise
    def fillcolumns(self, df):
        for i in range(len(df)):
            task = df.iloc[i, 2]

            for column in df:
                if task == column:
                    idx = df.columns.get_loc(column)
                    df.iloc[i, idx] = 1

                else:
                    continue

        return df

    # Merge rows of one order to a single row
    def mergerows(self, df):
        i = 0
        unique = df['order_id'].nunique()

        while i < unique - 1:
            j = i

            if j < unique-1:
                while j < unique and df.iloc[j, 0] == df.iloc[j+1, 0]:
                    task = df.iloc[j+1, 2]

                    for column in df:
                        if task == column:
                            idx = df.columns.get_loc(column)
                            df.iloc[j, idx] = 1
                            df.drop(df.index[j+1], inplace=True)

                        else:
                            continue

            else:
                continue

            i += 1

        if unique < len(df):
            leftrows = len(df) - unique
            j = unique-1

            for k in range(0, leftrows):
                if df.iloc[j, 0] == df.iloc[j+1, 0]:
                    task = df.iloc[j + 1, 2]

                    for column in df:

                        if task == column:
                            idx = df.columns.get_loc(column)
                            df.iloc[j, idx] = 1
                            df.drop(df.index[j + 1], inplace=True)

                        else:
                            continue

                else:
                    continue

        return df

    # Get haversine distance between two geo-coordinates
    @staticmethod
    def haversine(latitude_target, longitude_target, latitude_origin, longitude_origin):
        r = 6372.8
        d_latitude = radians(latitude_origin - latitude_target)
        d_longitude = radians(longitude_origin - longitude_target)
        latitude_target = radians(latitude_target)
        latitude_origin = radians(latitude_origin)

        a = sin(d_latitude / 2) ** 2 + cos(latitude_target) * cos(latitude_origin) * sin(d_longitude / 2) ** 2
        c = 2 * asin(sqrt(a))

        haversine_dist = r * c

        return haversine_dist

    # Identify closest target location out of Dataframe to given location
    def identify_close_poi(self, df_distance, latitude_origin, longitude_origin):
        best_distance = 9999
        best_row = None

        for row in range(len(df_distance)):
            latitude_target, longitude_target = df_distance.iloc[row, 0], df_distance.iloc[row, 1]
            distance = self.haversine(latitude_target, longitude_target, latitude_origin, longitude_origin)

            if distance < best_distance:
                best_distance = distance
                best_row = row

            else:
                continue

        return df_distance.iloc[best_row, 0], df_distance.iloc[best_row, 1], math.ceil(best_distance *
                                                                                       self.minutes_per_km)

    # If refuel task is included in order, add location of closest gas station and travel time from previous location
    # to Dataframe
    def assign_gas_station(self, df):
        df_cw = pd.read_csv(self.file_name_gas_station, usecols=('latitude', 'longitude'))
        df_cw['latitude'] = df_cw['latitude'].astype(float)
        df_cw['longitude'] = df_cw['longitude'].astype(float)
        idx_refuel = df.columns.get_loc('refuel')

        for i in range(len(df)):
            if df.iloc[i, idx_refuel] == 1:
                df.iloc[i, idx_refuel + 1], df.iloc[i, idx_refuel + 2], df.iloc[
                    i, idx_refuel + 3] = self.identify_close_poi(df_cw, df.iloc[i, 3], df.iloc[i, 4])

            else:
                continue

        return df

    # If car wash task or truck wash is included in order, add location of closest car wash location and travel time
    # from previous location to Dataframe
    def assign_car_wash(self, df):
        df_cw = pd.read_csv(self.file_name_car_wash, usecols=('latitude', 'longitude'))
        df_cw['latitude'] = df_cw['latitude'].astype(float)
        df_cw['longitude'] = df_cw['longitude'].astype(float)
        idx_car_wash = df.columns.get_loc('car_wash')
        idx_refuel = df.columns.get_loc('refuel')
        idx_truck_wash = df.columns.get_loc('truck_wash')

        for i in range(len(df)):
            if df.iloc[i, idx_car_wash] == 1:
                if df.iloc[i, idx_refuel] == 1:
                    df.iloc[i, idx_car_wash + 1], df.iloc[i, idx_car_wash + 2], df.iloc[
                        i, idx_car_wash + 3] = self.identify_close_poi(df_cw, df.iloc[i, idx_refuel + 1],
                                                                       df.iloc[i, idx_refuel + 2])
                else:
                    df.iloc[i, idx_car_wash + 1], df.iloc[i, idx_car_wash + 2], df.iloc[
                        i, idx_car_wash + 3] = self.identify_close_poi(df_cw, df.iloc[i, 3], df.iloc[i, 4])

            if df.iloc[i, idx_truck_wash] == 1:
                if df.iloc[i, idx_refuel] == 1:
                    df.iloc[i, idx_truck_wash + 1], df.iloc[i, idx_truck_wash + 2], df.iloc[
                        i, idx_truck_wash + 3] = self.identify_close_poi(df_cw, df.iloc[i, idx_refuel + 1],
                                                                         df.iloc[i, idx_refuel + 2])
                else:
                    df.iloc[i, idx_truck_wash + 1], df.iloc[i, idx_truck_wash + 2], df.iloc[
                        i, idx_truck_wash + 3] = self.identify_close_poi(df_cw, df.iloc[i, 3], df.iloc[i, 4])

        return df

    # If stationary charge task is included in order, add location of closest charging station and travel time
    # from previous location to Dataframe
    def assign_charging_station(self, df):
        df_cs = pd.read_csv(self.file_name_charging_station, usecols=('latitude', 'longitude'))
        df_cs['latitude'] = df_cs['latitude'].astype(float)
        df_cs['longitude'] = df_cs['longitude'].astype(float)
        idx_stationary_charge = df.columns.get_loc('stationary_charge')
        idx_refuel = df.columns.get_loc('refuel')
        idx_car_wash = df.columns.get_loc('car_wash')
        idx_truck_wash = df.columns.get_loc('truck_wash')

        for i in range(len(df)):
            if df.iloc[i, idx_stationary_charge] == 1:

                if df.iloc[i, idx_truck_wash] == 1:
                    df.iloc[i, idx_stationary_charge + 1], df.iloc[i, idx_stationary_charge + 2], df.iloc[
                        i, idx_stationary_charge + 3] = self.identify_close_poi(df_cs, df.iloc[i, idx_truck_wash + 1],
                                                                                df.iloc[i, idx_truck_wash + 2])

                elif df.iloc[i, idx_car_wash] == 1:
                    df.iloc[i, idx_stationary_charge + 1], df.iloc[i, idx_stationary_charge + 2], df.iloc[
                        i, idx_stationary_charge + 3] = self.identify_close_poi(df_cs, df.iloc[i, idx_car_wash + 1],
                                                                                df.iloc[i, idx_truck_wash + 2])

                elif df.iloc[i, idx_refuel] == 1:
                    df.iloc[i, idx_stationary_charge + 1], df.iloc[i, idx_stationary_charge + 2], df.iloc[
                        i, idx_stationary_charge + 3] = self.identify_close_poi(df_cs, df.iloc[i, idx_car_wash + 1],
                                                                                df.iloc[i, idx_refuel + 2])

                else:
                    df.iloc[i, idx_stationary_charge + 1], df.iloc[i, idx_stationary_charge + 2], df.iloc[
                        i, idx_stationary_charge + 3] = self.identify_close_poi(df_cs, df.iloc[i, 3], df.iloc[i, 4])

        return df

    # Calculate total service time and add to Dataframe
    def calculate_duration(self, df):
        task_list = self.get_task_list()
        durations = pd.read_csv(self.file_name_task_type_duration, header=0, index_col=0, squeeze=True).to_dict()
        total_duration = 0
        idx1 = df.columns.get_loc('DUE_DATE')
        df.insert(idx1+1, 'SERVICE_TIME', 0, True)

        for i in range(len(df)):
            for column in df[task_list]:
                idx2 = df.columns.get_loc(column)

                if df.iloc[i, idx2] == 1:
                    total_duration += durations[column]

                else:
                    continue

            for column in df[self.location_change]:
                idx3 = df.columns.get_loc(column)

                if df.iloc[3, idx3] == 1:
                    total_duration += df.iloc[i, idx3 + 3]

                else:
                    continue

            df.iloc[i, idx1 + 1] = total_duration
            total_duration = 0

        return df

    # Assign values of column SERVICE_TIME to column DEMAND
    def assign_demand(self, df):
        df['DEMAND'] = df['SERVICE_TIME']

        return df

    # Identify last location of order and assign to columns XCOORD_END and YCOORD_END
    def end_point(self, df):
        idx_stationary_charge = df.columns.get_loc('stationary_charge')
        idx_refuel = df.columns.get_loc('refuel')
        idx_car_wash = df.columns.get_loc('car_wash')
        idx_truck_wash = df.columns.get_loc('truck_wash')
        idx_startpoint_x = df.columns.get_loc('XCOORD.')
        idx_startpoint_y = df.columns.get_loc('YCOORD.')
        idx_endpoint_x = df.columns.get_loc('XCOORD_END')
        idx_endpoint_y = df.columns.get_loc('YCOORD_END')

        for i in range(len(df)):
            if df.iloc[i, idx_stationary_charge] == 1:
                df.iloc[i, idx_endpoint_x], df.iloc[i, idx_endpoint_y] = df.iloc[
                                                                             i, idx_stationary_charge + 1], df.iloc[
                                                                             i, idx_stationary_charge + 2]

            elif df.iloc[i, idx_refuel] == 1:
                df.iloc[i, idx_endpoint_x], df.iloc[i, idx_endpoint_y] = df.iloc[
                                                                             i, idx_refuel + 1], df.iloc[
                                                                             i, idx_refuel + 2]

            elif df.iloc[i, idx_truck_wash] == 1:
                df.iloc[i, idx_endpoint_x], df.iloc[i, idx_endpoint_y] = df.iloc[
                                                                             i, idx_truck_wash + 1], df.iloc[
                                                                             i, idx_truck_wash + 2]

            elif df.iloc[i, idx_car_wash] == 1:
                df.iloc[i, idx_endpoint_x], df.iloc[i, idx_endpoint_y] = df.iloc[
                                                                             i, idx_car_wash + 1], df.iloc[
                                                                             i, idx_car_wash + 2]

            else:
                df.iloc[i, idx_endpoint_x], df.iloc[i, idx_endpoint_y] = df.iloc[i, idx_startpoint_x], df.iloc[
                    i, idx_startpoint_y]

        return df

    # Round geo-coordinates for uniform format
    def round_coordinates(self, df):
        idx_x = df.columns.get_loc('XCOORD.')
        idx_y = df.columns.get_loc('YCOORD.')

        for i in range(len(df)):
            df.iloc[i, idx_x] = np.around(df.iloc[i, idx_x], 6)
            df.iloc[i, idx_y] = np.around(df.iloc[i, idx_y], 6)

        return df

    # Drop columns that are not required anymore
    def drop_columns(self, df):
        final_columns = ['XCOORD.', 'YCOORD.', 'AVAILABLE_TIME', 'DUE_DATE', 'SERVICE_TIME', 'READY_TIME', 'DEMAND',
                         'XCOORD_END', 'YCOORD_END']
        dup_idx = df.columns.duplicated()
        dup = df.columns[dup_idx].unique()
        rename_cols = []
        i = 1

        for col in df.columns:
            if col in dup:
                rename_cols.extend([col + '_' + str(i)])
                i += 1

            else:
                rename_cols.extend([col])

        df.columns = rename_cols

        for column in df:
            if column not in final_columns:
                del df[column]

            else:
                continue

        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
        df.reset_index(level=df.index.names, inplace=True)
        df.rename(columns={'index': 'CUST_NO.'}, inplace=True)

        return df

    # Create header for final file
    def create_header(self):
        f = open(self.target_file_path, 'w+')
        date = self.target_file_name[:10]
        vehicles = self.vehicles
        capacity = self.capacity
        f.write('{}\n\nVEHICLE\nNUMBER\tCAPACITY\n{}\t{}\n\nCUSTOMER\n'.format(date, vehicles, capacity))
        f.close()

    # Build Dataframe out of export file and manipulate
    def build_df(self):
        df = pd.read_csv(self.original_file_path,
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

        df = self.addcolumns(df)
        df = self.fillcolumns(df)
        df = self.mergerows(df)
        df = self.assign_gas_station(df)
        df = self.assign_car_wash(df)
        df = self.assign_charging_station(df)
        df = self.calculate_duration(df)
        df = self.assign_demand(df)
        df = self.end_point(df)
        df = self.round_coordinates(df)
        df = self.drop_columns(df)

        return df

    # Export manipulated Dataframe
    def export_txt(self):
        self.create_header()
        df = self.build_df()
        f = open(self.target_file_path, 'a+')
        f.write('{}'.format(df.to_string(columns=('CUST_NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY_TIME', 'DUE_DATE',
                                                  'SERVICE_TIME', 'AVAILABLE_TIME', 'XCOORD_END', 'YCOORD_END'),
                                         index=False, justify='center')))
        f.close()

    # Run Pre Processing through starting export method
    def runpreprocessing(self):
        self.export_txt()
        print('-----PRE PROCESSING FINALIZED-----')
