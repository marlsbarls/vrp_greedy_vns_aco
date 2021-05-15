shift_begin = '09:00:00'
order_reception_end = '16:00:00'
shift_end = '17:00:00'
poi_visibility_limit = 5
minutes_per_kilometer = 2
retrieve_traffic_data = False

dynamic_tasks = {
    "stationary_charge": {"loc_file": "charging_station_locations.csv", "static_demand": 5, "availability": [0.5, 0.75, 1, 1, 1]},
    "refuel": {"loc_file": "gas_station_locations.csv", "static_demand": 15, "availability": [1, 1, 1, 1, 1]},
    "car_wash": {"loc_file": "car_wash_locations.csv", "static_demand": 20, "availability": [1, 1, 1, 1, 1]},
    "exterior_cleaning": {"loc_file": "car_wash_locations.csv", "static_demand": 20, "availability": [1, 1, 1, 1, 1]}

}

static_tasks = {
    "front_interior_cleaning": 10,
    "interior_cleaning": 20,
    "unplug": 5
}

traffic_times = {
    "rush_hour": {"name": "rush_hour", "from_shift_start": 390, "formatted": "16:30:00"},
    "phase_transition": {"name": "phase_transition", "from_shift_start": 300, "formatted": "14:00:00"},
    "off_peak": {"name": "off_peak", "from_shift_start": 0, "formatted": "11:30:00"}
}


here_api = {
    "matrix_url": "https://matrix.router.hereapi.com/v8/matrix",
    "app_id": "I5uhnu7sk12yOgzPAP5t"
}
