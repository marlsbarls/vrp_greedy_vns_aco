
import vns.src.config.preprocessing_config as cfg
from collections import OrderedDict
from shapely.geometry import Point
import requests
import pandas as pd
import time
import json
import os
from pathlib import Path


class TrafficInterface_Reduced:
    def __init__(self, base_dir, file_name, task_gdf):
        # task_gdf is expected to be dataframe, resulting from Preprosessing.extract_required_pois
        self.date = file_name
        self.base_dir = base_dir
        self.target_folder = os.path.join(
            self.base_dir, "data", "travel_distances")
        Path(self.target_folder).mkdir(parents=True, exist_ok=True)
        self.target_file_path = os.path.join(
            self.target_folder, file_name + "_tasks.txt")
        self.task_gdf = task_gdf
        self.api_key = os.getenv("HERE_API_KEY")
        self.matrix_url = "https://matrix.router.hereapi.com/v8/matrix?async=false"

    def exists(self):
        # print(self.target_file_path)
        # print(os.path.isfile(self.target_file_path))
        return os.path.isfile(self.target_file_path)

    @staticmethod
    def get_unique_pois(row):
        unique_coordinates = {}
        for i in range(5):
            id = row[str(i) + "_closest_id"]
            point = row[str(i) + "_closest_geometry"]
            unique_coordinates.update({id: {"lat": point.y, "lng": point.x}})
        return unique_coordinates

    @staticmethod
    def get_index_of_matrix(origins, destinations, i, j):
        # i, j are expected to be ids
        # origins & destinations are expected to match format Dict<id: Dict<lat: int, lng: int>>
        return len(destinations.keys()) * list(origins).index(i) + list(destinations).index(j)

    def get_matrix_synchronus(self, origins, destinations, departure_time="any", transport_mode="car"):
        # origins & destinations are expected to match schema List<Dict<lat: float, lng: float>>
        # transport_mode (optional) is expected to equal "car" or "bicycle"
        # departure time (optional) is expected to be iso formatted
        # unique_coordinates.values()
        data = {
            "origins": origins,
            "destinations": destinations,
            "regionDefinition": {
                "type": "circle",
                "center": {"lat": 52.520008, "lng": 13.404954},
                "radius": 25000
            },
            "matrixAttributes": ["travelTimes", "distances"],
            "routingMode": "fast",
            "transportMode": transport_mode,
            "departureTime": departure_time
        }

        res = requests.post(self.matrix_url + "&apiKey=" + self.api_key,
                            json=data, headers={"Content-Type": "application/json"})
        # print(res.request.url, res.request.path_url, res.request.body)
        if(res.status_code == 200):
            response = res.json()
            # print(res.request.body)
            # print(res.json())
            response = res.json()
            return response['matrix'].get('travelTimes', None), response['matrix'].get('errorCodes', None)
        else:
            raise Exception("Error in submitting matrix: ",
                            res.status_code, res.json())

    @staticmethod
    def write_file(file_path, data):
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def load_file(path, file):
        with open(path + file + '.txt', 'r') as json_file:
            data = json.load(json_file)
            return data

    def get_dynamic_travel_times(self, task, pois, time):
        # expect time to match schema Dict<name: string, from_shift_start: int, formatted: time>
        '''tasks = {"origin_0": {"lat": 1, "lng": 2},
                 "origin_1": {"lat": 1, "lng": 2}}
        pois = {"destination_0": {"lat": 1, "lng": 2},
                "destination_1": {"lat": 1, "lng": 2}}
        dynamic_travel_times = [0, 1, 2, 4]'''
        # "2020-10-19T08:00:00+01:00"
        departure_time = self.date + "T" + time["formatted"] + "+01:00"
        dynamic_travel_times, error_codes = self.get_matrix_synchronus(
            list(task.values()), list(pois.values()), departure_time)
        '''self.write_file(os.path.join(self.base_dir, "data", "travel_distances",
                                     "backup", self.date + "_" + time["name"] + "_dynamic.txt"), dynamic_travel_times)'''
        kv_store = {}
        for origin in task.keys():
            for destination in pois.keys():
                kv_store.update({origin + ":" + destination + ":" +
                                 time['name']: dynamic_travel_times[self.get_index_of_matrix(task, pois, origin, destination)]})
        return kv_store

    def retrieve_travel_time_matrix(self):
        '''TODO: Needs to be filled accordingly'''
        travel_times = {}
        counter = 1
        for idx, row in self.task_gdf[self.task_gdf["0_closest_geometry"] != 0].iterrows():
            print("Task " + str(counter) + "/" +
                  str(len(self.task_gdf[self.task_gdf["0_closest_geometry"] != 0].index)))
            origin = {"task_" + row["task_id"]
                : {"lat": row["YCOORD."], "lng": row["XCOORD."]}}
            destinations = self.get_unique_pois(row)
            for time in cfg.traffic_times.keys():
                print(row["task_id"] + ":" + time)
                travel_times = {**travel_times, **self.get_dynamic_travel_times(
                    origin, destinations, cfg.traffic_times[time])}
                self.write_file(self.target_file_path, travel_times)
            counter += 1
        print("----TRAFFIC DATA RETRIEVED ----")
        # TODO: Uncomment real here apis
