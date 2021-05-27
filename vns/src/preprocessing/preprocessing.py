import pandas as pd
import geopandas
import os
from src.helpers.RangeFinder import RangeFinder
import src.config.preprocessing_config as cfg
import random
import numpy as np
import json
from src.helpers.DistanceMatrix import DistanceMatrix
from math import radians, cos, sin, asin, sqrt
from src.traffic_data.TrafficInterface_Reduced import TrafficInterface_Reduced
from pathlib import Path


class Preprocessing:
    def __init__(self, base_dir, file_name, **kwargs):
        self.base_dir = base_dir
        self.file_name = file_name
        self.target_folder = kwargs["target_folder"] if "target_folder" in kwargs else os.path.join(
            self.base_dir, "data", "results_preprocessing")
        Path(self.target_folder).mkdir(parents=True, exist_ok=True)
        self.config_string = kwargs["charge_avail_config_string"] if "charge_avail_config_string" in kwargs else ""
        if("charge_avail_config" in kwargs):
            cfg.dynamic_tasks["stationary_charge"]["availability"] = kwargs["charge_avail_config"]
        self.source_file_path = os.path.join(
            self.base_dir, "data", "results_data_preparation", file_name + ".csv")
        self.service_times_target_file_path = os.path.join(
            self.target_folder, file_name + self.config_string + "_service_times.txt")
        self.travel_times_target_file_path = os.path.join(
            self.target_folder, file_name + self.config_string + "_travel_times.txt")
        self.order_df_target_file_path = os.path.join(
            self.target_folder, file_name + self.config_string + "_orders.csv")
        self.task_df = pd.read_csv(self.source_file_path)
        self.task_gdf = geopandas.GeoDataFrame(
            self.task_df, geometry=geopandas.points_from_xy(self.task_df['XCOORD.'], self.task_df['YCOORD.']))
        self.api_key = os.getenv("HERE_API_KEY")
        self.matrix_url = "https://matrix.router.hereapi.com/v8/matrix"
        # Analysis, which # nearest charging station is utilized how much
        self.charging_counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Helper class to reduce data volume
    # Instead of requesting every distance from every task to every possible POI, we limit the visibility for each task to 5 nearest POIs
    # measured by haversine distance between coordinates
    def extract_required_pois(self):
        ''' construct location data frames'''
        charge_df = pd.read_csv(os.path.join(
            self.base_dir, "data", "locations", cfg.dynamic_tasks["stationary_charge"]["loc_file"]))
        charge_gdf = geopandas.GeoDataFrame(charge_df, geometry=geopandas.points_from_xy(
            charge_df.longitude, charge_df.latitude))

        refuel_df = pd.read_csv(os.path.join(
            self.base_dir, "data", "locations", cfg.dynamic_tasks["refuel"]["loc_file"]))
        refuel_gdf = geopandas.GeoDataFrame(
            refuel_df, geometry=geopandas.points_from_xy(refuel_df.longitude, refuel_df.latitude))

        wash_df = pd.read_csv(os.path.join(
            self.base_dir, "data", "locations", cfg.dynamic_tasks["car_wash"]["loc_file"]))
        wash_gdf = geopandas.GeoDataFrame(
            wash_df, geometry=geopandas.points_from_xy(wash_df.longitude, wash_df.latitude))

        for i in range(cfg.poi_visibility_limit):
            idx = len(self.task_gdf.columns)
            self.task_gdf.insert(idx, str(i) + '_closest_id', 0, False)
            self.task_gdf.insert(idx + 1, str(i) +
                                 '_closest_geometry', 0, False)

        charging_finder = RangeFinder(charge_gdf)
        refuel_finder = RangeFinder(refuel_gdf)
        wash_finder = RangeFinder(wash_gdf)
        col_task_type = self.task_gdf.columns.get_loc("task_type")
        wash = ["exterior_cleaning", "car_wash", "truck_wash"]
        # For every task, get required pois
        for i in range(len(self.task_gdf)):
            task_type = self.task_gdf.iloc[i, col_task_type]
            # Get closest POIs, depending on task_type
            if(task_type == "stationary_charge"):
                dfs = charging_finder.nearest_neighbor(
                    self.task_gdf[self.task_gdf.index == i], None, False, cfg.poi_visibility_limit)
            elif(task_type == "refuel"):
                dfs = refuel_finder.nearest_neighbor(
                    self.task_gdf[self.task_gdf.index == i], None, False, cfg.poi_visibility_limit)
            elif(task_type == "exterior_cleaning" or task_type == "car_wash" or task_type == "truck_wash"):
                dfs = wash_finder.nearest_neighbor(
                    self.task_gdf[self.task_gdf.index == i], None, False, cfg.poi_visibility_limit)
            else:
                dfs = []

            if(len(dfs) > 0):
                for idx, df in enumerate(dfs):
                    col = self.task_gdf.columns.get_loc(
                        str(idx) + "_closest_id")
                    task_type = self.task_gdf.iloc[i,
                                                   self.task_gdf.columns.get_loc("task_type")]
                    self.task_gdf.iloc[i, col] = (
                        task_type + "_" if not task_type in wash else "wash_") + str(df.iloc[0, df.columns.get_loc("#")])
                    self.task_gdf.iloc[i, col+1] = df.iloc[0,
                                                           df.columns.get_loc("geometry")]

        return self.task_gdf

    @staticmethod
    def get_nearest_pois(row, matrix):
        path_list = []
        for i in range(5):
            path_keys = matrix.get_keys_matching_pattern(
                ["task_"+row["task_id"] + ":", row[str(i)+"_closest_id"]])
            path_list.append(
                (row[str(i)+"_closest_id"], np.mean([matrix.get_value(key) for key in path_keys])))
        path_list.sort(key=lambda x: x[1])
        return path_list
        # return path_list[0], path_list[1]

    @staticmethod
    def get_geometry_from_id(row, id):
        for i in range(0, cfg.poi_visibility_limit):
            if(row[str(i) + '_closest_id'] == id):
                return row[str(i) + '_closest_geometry']

    def create_dataframe(self, task_matrix):
        unique_order_ids = self.task_gdf.order_id.unique()
        service_times_kv_store = {}
        data = []
        for i in range(0, len(unique_order_ids)):
            temp = self.task_gdf[self.task_gdf["order_id"]
                                 == unique_order_ids[i]]

            service_time, static_service_time, end_point, task_id = 0, 0, None, None

            order_id = "order_" + temp.iloc[0]['order_id']
            if(order_id != "order_0"):
                for idx, row in temp.iterrows():
                    task_type = row["task_type"]
                    if(task_type in cfg.dynamic_tasks.keys()):

                        sorted_nearest_pois = self.get_nearest_pois(
                            row, task_matrix)

                        prob = random.uniform(0, 1)
                        for n in range(5):
                            if (prob <= cfg.dynamic_tasks[task_type]["availability"][n]):
                                end_point = (sorted_nearest_pois[n][0], self.get_geometry_from_id(
                                    row, sorted_nearest_pois[n][0]))
                                if(task_type == "stationary_charge"):
                                    self.charging_counter[n + 1] += 1
                                break

                        task_id = "task_" + row["task_id"]
                        static_service_time += cfg.dynamic_tasks[task_type]["static_demand"]

                    else:
                        static_service_time += cfg.static_tasks[task_type]

                for time in cfg.traffic_times.keys():
                    service_time = static_service_time + (round(task_matrix.get_value(
                        task_id + ":" + end_point[0]+":" + cfg.traffic_times[time]["name"])/60) + 1 if task_id else 0)
                    service_times_kv_store.update(
                        {order_id + ":" + cfg.traffic_times[time]["name"]: service_time})

            # Ausgangs-DF
            # index, order_id, task_id, task_type, XCOORD., YCOORD., AVAILIBLE TIME, DUE DATE, READY TIME,
            # geometry, 0_closest_id 	0_closest_geometry 	1_closest_id 	1_closest_geometry 	2_closest_id
            # 2_closest_geometry 	3_closest_id 	3_closest_geometry 	4_closest_id 	4_closest_geometry

            # ZIEL_DF:
            # CUST_NO, XCOORD, YCOORD, DEMAND, READYTIME, DUETIME, SERVICETIME, XCOORD_END, YCOORD_END, order_id, end_poi_id
            data.append([i, temp.iloc[0]['XCOORD.'], temp.iloc[0]['YCOORD.'], service_time, temp.iloc[0]['READY TIME'], temp.iloc[0]['DUE DATE'], service_time,
                         end_point[1].x if end_point else temp.iloc[0]['XCOORD.'], end_point[1].y if end_point else temp.iloc[0]['YCOORD.'], order_id, end_point[0] if end_point else order_id])
        order_frame = pd.DataFrame(data, columns=["CUST_NO", "XCOORD", "YCOORD", "DEMAND", "READYTIME",
                                                  "DUETIME", "SERVICETIME", "XCOORD_END", "YCOORD_END", "order_id", "end_poi_id"])
        return order_frame, service_times_kv_store

    # this method was provided by Nina Schwarm, (c) FU Berlin 2021
    @staticmethod
    def haversine(latitude_target, longitude_target, latitude_origin, longitude_origin):
        r = 6372.8
        d_latitude = radians(latitude_origin - latitude_target)
        d_longitude = radians(longitude_origin - longitude_target)
        latitude_target = radians(latitude_target)
        latitude_origin = radians(latitude_origin)

        a = sin(d_latitude / 2) ** 2 + cos(latitude_target) * \
            cos(latitude_origin) * sin(d_longitude / 2) ** 2
        c = 2 * asin(sqrt(a))

        haversine_dist = r * c
        return haversine_dist

    def get_final_matrix(self, order_df):
        travel_time_kv_store = {}
        for i_ori, origin in order_df.iterrows():
            for i_dest, destination in order_df.iterrows():
                if not origin["order_id"] == destination["order_id"]:
                    travel_time = self.haversine(destination["YCOORD"], destination["XCOORD"],
                                                 origin["YCOORD_END"], origin["XCOORD_END"]) * cfg.minutes_per_kilometer
                    travel_time_kv_store.update(
                        {origin["order_id"]+":"+destination["order_id"]: travel_time})
        return travel_time_kv_store

    @staticmethod
    def write_file(file_path, data):
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def exportcsv(file_path, df):
        df.to_csv(file_path, header=True, index=False)

    def runpreprocessing(self):
        self.extract_required_pois()

        # Check whether travel_times already exists, pull them from API otherwise
        ti = TrafficInterface_Reduced(
            self.base_dir, self.file_name, self.task_gdf)
        if((not ti.exists()) or cfg.retrieve_traffic_data):
            ti.retrieve_travel_time_matrix()

        task_matrix = DistanceMatrix(self.base_dir, self.file_name + "_tasks")
        order_df, service_times_kv_store = self.create_dataframe(task_matrix)
        travel_times_kv_store = self.get_final_matrix(order_df)

        self.write_file(self.service_times_target_file_path,
                        service_times_kv_store)
        self.write_file(self.travel_times_target_file_path,
                        travel_times_kv_store)
        self.exportcsv(self.order_df_target_file_path, order_df)
        print("Utilization of charging stations: ", self.charging_counter)
        print("--- PREPROCESSING FINALIZED ---")
        return order_df, service_times_kv_store, travel_times_kv_store, self.charging_counter
