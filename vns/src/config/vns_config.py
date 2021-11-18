cost_per_hour = 9.5
cost_per_minute = cost_per_hour / 60
cost_per_driver = 400
minutes_per_kilometer = 2
capacity = int(480)
num_vehicles = int(25)
greedy_time_weight = 0
# capacity = 80

shaking = {
    "ICROSS": {
        "PROBABILITY": 0.4
    },
    "INSERT": {
        "PROBABILITY": 0.97
    },
    "CROSS": {
        "PROBABILITY": 0.6
    },
    "SORT_LEN": {
        "PROBABILITY": 1
    }
}

vns = {
    "2-OPT": {
        "PROBABILITY": 0.5
    },
    # Original
    "InitialTemperature": 0,
    "MaxRestarts": 7,
    "MaxIterations_NoImp": 50,
    "MaxIterations": 100,
    "MaxRunTimeD": 15,
    "MaxRunTimeS": 480
    
    # "MaxRunTimeS": 0.1
    # "MaxRunTimeS": 15

    # "MaxRunTime": 0.1,
    # "InitialTemperature": 0,
    # "MaxRestarts": 1,
    # "MaxIterations_NoImp": 5,
    # "MaxIterations": 10

    # Static Experiement
    # "InitialTemperature": 0,
    # "MaxRestarts": 7*5,
    # "MaxIterations_NoImp": 50*5,
    # "MaxIterations": 100*5

}
