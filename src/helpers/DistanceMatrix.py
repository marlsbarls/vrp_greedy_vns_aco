'''
helper class to maintain a key value store, representing a distance matrix
primarely used in preprocessing
'''

import json
import os


class DistanceMatrix:
    @staticmethod
    def load_file(file_path):
        with open(file_path + '.txt', 'r') as json_file:
            data = json.load(json_file)
            return data

    def __init__(self, base_dir, file_name):
        self.base_dir = base_dir
        self.matrix_file_path = os.path.join(
            self.base_dir, "data", "travel_distances", file_name)
        self.kv_store = self.load_file(self.matrix_file_path)

    def get_value(self, key):
        return self.kv_store[key]

    def get_keys_matching_pattern(self, pattern):
        keys = [k for k in self.kv_store.keys() if all(
            map(k.__contains__, pattern))]
        return keys

    @staticmethod
    def write_file(file_path, data):
        with open(file_path + '.txt', 'w') as outfile:
            json.dump(data, outfile)
