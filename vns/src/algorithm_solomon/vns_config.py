cost_per_hour = 9.5
cost_per_minute = cost_per_hour / 60
cost_per_driver = 400
minutes_per_kilometer = 2
capacity = 480

shaking = {
    "ICROSS": {
        "PROBABILITY": 0.4
    },
    "INSERT": {
        "PROBABILITY": 0.9
    },
    "CROSS": {
        "PROBABILITY": 0.6
    },
    "SORT_LEN": {
        "PROBABILITY": 0.5
    }
}

vns = {
    "2-OPT": {
        "PROBABILITY": 0.5
    },
    "InitialTemperature": 10,
    "MaxRestarts": 10,
    "MaxIterations_NoImp": 50,
    "MaxIterations": 100
}
