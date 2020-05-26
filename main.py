import pandas as pd
import time
from dynamic_programming import solve_knapsack_problem_dp, solve_knapsack_problem_dp_with_reds, solve_knapsack_problem_dp_with_memory
from greedy import greedy
from branch_and_bounds import simple_bnb, sorted_simple_bnb
import os
from IPython.display import display

def read_data_from_file(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    weight_knapsack = int(data[0].split(" ")[1])
    weight_items = list()
    value_items = list()
    for id, line in enumerate(data):
        if id>0:
            weight_items.append(int(data[id].split(" ")[1]))
            value_items.append(int(data[id].split(" ")[0]))
    if value_items[-1] == 0 or weight_items[-1] == 0:
        value_items.pop()
        weight_items.pop()

    data = pd.DataFrame()
    data["weight"] = [0] + weight_items
    data["value"] = [0] + value_items

    return weight_knapsack, data


def solve_task(function, data, weight_knapsack, task_name):
    bt = time.perf_counter()
    result, method_name = function(data, weight_knapsack)
    result_time = time.perf_counter()-bt
    # print(task_name, method_name, ": Result - ", result, " Time  - ", result_time)
    return method_name, result, result_time


def run_all_low(methods):
    # path = "data/low-dimensional/"
    # filenames = os.listdir(path)

    path = "data/large_scale"
    # filenames = ["knapPI_1_100_1000_1", "knapPI_2_100_1000_1", "knapPI_3_100_1000_1"]

    # filenames = ["knapPI_1_100_1000_1", "knapPI_2_100_1000_1", "knapPI_3_100_1000_1",
    #              "knapPI_1_200_1000_1", "knapPI_2_200_1000_1", "knapPI_3_200_1000_1",
    #              "knapPI_1_500_1000_1", "knapPI_2_500_1000_1", "knapPI_3_500_1000_1",
    #              "knapPI_1_1000_1000_1", "knapPI_2_1000_1000_1", "knapPI_3_1000_1000_1"]  #
    filenames = ["knapPI_1_500_1000_1", "knapPI_2_500_1000_1", "knapPI_3_500_1000_1",
                 ]  #"knapPI_1_1000_1000_1", "knapPI_2_1000_1000_1", "knapPI_3_1000_1000_1"

    result = {"Data":[], "Method":[], "Time":[],  "Value":[]}
    for file in filenames:
        weight_knapsack, data = read_data_from_file(os.path.join(path, file))
        for method in methods:
            tmp = solve_task(method, data, weight_knapsack, file)
            result["Data"].append(file)
            result["Method"].append(tmp[0])
            result["Value"].append(tmp[1])
            result["Time"].append(tmp[2])
    result = pd.DataFrame(result)
    display(result)


def run_one(methods):
    path = "data/low-dimensional/"
    file = 'f7_l-d_kp_7_50'

    # path = "data/large_scale"
    # file = 'knapPI_1_2000_1000_1'

    result = {"Data":[], "Method":[], "Time":[],  "Value":[]}

    weight_knapsack, data = read_data_from_file(os.path.join(path, file))
    for method in methods:
        tmp = solve_task(method, data, weight_knapsack, file)
        result["Data"].append(file)
        result["Method"].append(tmp[0])
        result["Value"].append(tmp[1])
        result["Time"].append(tmp[2])
    result = pd.DataFrame(result)
    display(result)


if __name__ == "__main__":
    methods = [sorted_simple_bnb,  greedy, solve_knapsack_problem_dp_with_memory, solve_knapsack_problem_dp_with_reds, solve_knapsack_problem_dp] # simple_bnb
    # run_one(methods)
    run_all_low(methods)
