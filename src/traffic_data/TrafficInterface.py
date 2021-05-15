'''
First working version of the Traffic Interface. During testing it was found, that the utilized data volume exceeds the maximum requests
of the free here API tier. A reduced version was derived, which is now in use by preprocessing step.

This file is a relic, which is supposed to show how a traffic interface could work in a perfect world with better financial conditions.
'''

from src.preprocessing.preprocessing2 import extract_required_pois
import src.config.preprocessing_config as cfg
from collections import OrderedDict
from shapely.geometry import Point
import requests
import pandas as pd
import time
import json
import os


class TrafficInterface:
    def __init__(self, base_dir, file_name, task_gdf):
        # task_gdf is expected to be dataframe, resulting from Preprosessing.extract_required_pois
        self.date = file_name
        self.base_dir = base_dir
        self.target_file_path = os.path.join(
            self.base_dir, "data", "travel_distances", file_name + "_tasks.txt")
        self.task_gdf = task_gdf
        self.api_key = os.getenv("HERE_API_KEY")
        self.matrix_url = "https://matrix.router.hereapi.com/v8/matrix"

    def get_unique_tasks(self):
        unique_coordinates = {}
        for ind, row in self.task_gdf[["task_id", "geometry"]].iterrows():
            index = row[0]
            point = row[1]
            unique_coordinates.update(
                {"task_" + str(index): {"lat": point.y, "lng": point.x}})
        return unique_coordinates

    def get_unique_pois(self):
        unique_coordinates = {}
        for i in range(5):
            loc = self.task_gdf[[
                str(i) + "_closest_id", str(i) + "_closest_geometry"]]
            for ind, row in loc.iterrows():
                index = row[0]
                point = row[1]
                if(index != 0):
                    if(not index in unique_coordinates.keys()):
                        unique_coordinates.update(
                            {index: {"lat": point.y, "lng": point.x}})
        return unique_coordinates

    @staticmethod
    def get_index_of_matrix(origins, destinations, i, j):
        # i, j are expected to be ids
        # origins & destinations are expected to match format Dict<id: Dict<lat: int, lng: int>>
        return len(destinations.keys()) * list(origins).index(i) + list(destinations).index(j)

    def post_matrix(self, origins, destinations, transport_mode="car", departure_time="any"):
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

        res = requests.post(self.matrix_url + "?apiKey=" + self.api_key,
                            json=data, headers={"Content-Type": "application/json"})
        #print(res.request.url, res.request.path_url, res.request.body)
        if(res.status_code == 202):
            response = res.json()
            # print(res.json())
            return response['matrixId'], response['statusUrl']
        else:
            raise Exception("Error in submitting matrix: ",
                            res.status_code, res.json())

    def get_matrix(self, matrix_id):
        time.sleep(2)
        counter = 1
        calculation_complete = False
        while not calculation_complete:
            print('Attempt:' + str(counter))
            res = requests.get(self.matrix_url + "/" + matrix_id + "?apiKey=" +
                               self.api_key, headers={"Content-Type": "application/json"})
            if(len(res.history) > 0):
                calculation_complete = True
                response = res.json()
                # return response['matrix'], response['matrix']['travelTimes'], response['matrix'].get('errorCodes', None)
                return response['matrix'].get('travelTimes', None), response['matrix'].get('errorCodes', None)
            else:
                time.sleep(10)
                counter += 1

    @staticmethod
    def write_file(file_path, data):
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def load_file(path, file):
        with open(path + file + '.txt', 'r') as json_file:
            data = json.load(json_file)
            return data

    def get_static_travel_times(self, pois, tasks):
        static_origins = pois
        static_origins.update({"task_0": tasks['task_0']})
        '''static_origins = {"origin_0": {"lat": 1, "lng": 2},
                          "origin_1": {"lat": 1, "lng": 2}}
        tasks = {"destination_0": {"lat": 1, "lng": 2},
                 "destination_1": {"lat": 1, "lng": 2}}
        static_travel_times = [0, 1, 2, 4]'''
        matrix_id, status_url = self.post_matrix(
            list(static_origins.values()), list(tasks.values()), "bicycle")
        print("Matrix-ID (statisch): ", matrix_id)
        static_travel_times, error_codes = self.get_matrix(matrix_id)
        self.write_file(os.path.join(self.base_dir, "data", "travel_distances",
                                     "backup", self.date + "_static.txt"), static_travel_times)
        self.write_file(os.path.join(self.base_dir, "data", "travel_distances",
                                     "backup", self.date + "_static_errors.txt"), error_codes)
        kv_store = {}
        for origin in static_origins.keys():
            for destination in tasks.keys():
                kv_store.update({origin + ":" + destination: static_travel_times[self.get_index_of_matrix(
                    static_origins, tasks, origin, destination)]})

        return kv_store

    def get_dynamic_travel_times(self, tasks, pois, time):
        # expect time to match schema Dict<name: string, from_shift_start: int, formatted: time>
        '''tasks = {"origin_0": {"lat": 1, "lng": 2},
                 "origin_1": {"lat": 1, "lng": 2}}
        pois = {"destination_0": {"lat": 1, "lng": 2},
                "destination_1": {"lat": 1, "lng": 2}}
        dynamic_travel_times = [0, 1, 2, 4]'''
        # "2020-10-19T08:00:00+01:00"
        departure_time = self.date + "T" + time["formatted"] + "+01:00"
        matrix_id, status_url = self.post_matrix(
            list(tasks.values()), list(pois.values()), "car", departure_time)
        print("Matrix-ID (dynamisch): ", departure_time, ": ", matrix_id)
        dynamic_travel_times, error_codes = self.get_matrix(matrix_id)
        self.write_file(os.path.join(self.base_dir, "data", "travel_distances",
                                     "backup", self.date + "_" + time["name"] + "_dynamic.txt"), dynamic_travel_times)
        self.write_file(os.path.join(self.base_dir, "data", "travel_distances",
                                     "backup", self.date + "_" + time + "_dynamic_errors.txt"), error_codes)
        kv_store = {}
        for origin in tasks.keys():
            for destination in pois.keys():
                kv_store.update({origin + ":" + destination + ":" +
                                 time['name']: dynamic_travel_times[self.get_index_of_matrix(tasks, pois, origin, destination)]})
        return kv_store

    def retrieve_travel_time_matrix(self):
        '''TODO: Needs to be filled accordingly'''
        tasks = self.get_unique_tasks()
        pois = self.get_unique_pois()

        # print(list(tasks.values()))
        # print(pois)

        travel_times = self.get_static_travel_times(pois, tasks)
        for time in cfg.traffic_times.keys():
            travel_times = {
                **travel_times, **self.get_dynamic_travel_times(tasks, pois, cfg.traffic_times[time])}

        self.write_file(self.target_file_path, travel_times)
        print("----TRAFFIC DATA RETRIEVED ----")
        # TODO: Uncomment real here apis
