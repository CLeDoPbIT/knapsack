import pandas as pd
import time
from dynamic_programming import solve_knapsack_problem_dp, solve_knapsack_problem_dp_with_reds, solve_knapsack_problem_dp_with_memory, solve_knapsack_problem_dp_with_memory_fast
from greedy import greedy
from branch_and_bounds import simple_bnb, sorted_simple_bnb
from generate_benchmarks import generate_uniform
from backtracking import backtracking, backtracking_bnb, backtracking_bnb_red, backtracking_red
import os
from IPython.display import display
from IPython.core.display import HTML
import matplotlib.pyplot as plt


pd.option_context('display.max_rows', None)
pd.set_option('display.max_columns', None)

if not os.path.exists("data/random_uniform/"):
    os.mkdir("data/random_uniform/")


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


def solve_task(function, data, weight_knapsack):
    bt = time.perf_counter()
    result, method_name, operations = function(data, weight_knapsack)
    result_time = time.perf_counter()-bt
    return method_name, result, result_time, operations


def run_all_low(methods):
    path = "data/low-dimensional/"
    filenames = os.listdir(path)

    # path = "data/large_scale"
    # filenames = [
    #              "knapPI_1_200_1000_1", "knapPI_2_200_1000_1", "knapPI_3_200_1000_1",
    #              "knapPI_1_500_1000_1", "knapPI_2_500_1000_1", "knapPI_3_500_1000_1",
    #              ]  #

    result = {"Data":[], "Method":[], "Time":[],  "Value":[], "Operations":[]}
    for file in filenames:
        weight_knapsack, data = read_data_from_file(os.path.join(path, file))
        for method in methods:
            tmp = solve_task(method, data, weight_knapsack)
            result["Data"].append(file)
            result["Method"].append(tmp[0])
            result["Value"].append(tmp[1])
            result["Time"].append(tmp[2])
            result["Operations"].append(tmp[3])

        result["Data"].append("---------")
        result["Method"].append("---------")
        result["Value"].append("---------")
        result["Time"].append("-------")
        result["Operations"].append("-------")

        print("Done", file)
    result = pd.DataFrame(result)
    result.to_csv("result.csv")
    display(result)


def run_one(methods):
    path = "data/low-dimensional/"
    file = 'f4_l-d_kp_4_11'

    # path = "data/large_scale"
    # file = 'knapPI_1_2000_1000_1'

    result = {"Data":[], "Method":[], "Time":[],  "Value":[], "Operations":[]}

    weight_knapsack, data = read_data_from_file(os.path.join(path, file))
    for method in methods:
        tmp = solve_task(method, data, weight_knapsack)
        result["Data"].append(file)
        result["Method"].append(tmp[0])
        result["Value"].append(tmp[1])
        result["Time"].append(tmp[2])
        result["Operations"].append(tmp[3])

    result = pd.DataFrame(result)
    display(result)


def run_random_tasks(methods):
    nb = [50, 100, 1000, 10000, 100000]
    result = {"Data": [], "Method": [], "Time": [], "Value": [], "Operations": []}

    # for number_items in nb:
    #     weight_knapsack = 10**number_items # number_items
    #     number_items = 100
    #     values_boundary = [1, weight_knapsack]
    #     weights_boundary = [1, weight_knapsack]
    #     number_random_data = 1
    for number_items in nb:
        weight_knapsack = number_items
        values_boundary = [1, number_items]
        weights_boundary = [1, number_items]
        number_items = 1000

        number_random_data = 1
        print("DATA", weight_knapsack, number_items, values_boundary, weights_boundary)
        for nrd in range(number_random_data):
            print("Done", nrd, "Total number", number_random_data)
            data = generate_uniform(number_items, values_boundary, weights_boundary)
            data.to_csv("data/random_uniform/"+str(nrd)+".csv")

            for method in methods:
                tmp = solve_task(method, data, weight_knapsack)
                result["Data"].append(number_items)
                result["Method"].append(tmp[0])
                result["Value"].append(tmp[1])
                result["Time"].append(tmp[2])
                result["Operations"].append(tmp[3])
    result = pd.DataFrame(result)

    sdp_time = result[result["Method"]=="Simple DP"]["Time"]
    dp_cc_time = result[result["Method"]=="backtracking"]["Time"]
    dp_lb_time = result[result["Method"]=="Sorted + Red DP"]["Time"]
    dp_cc_lb_time = result[result["Method"]=="backtracking red"]["Time"]
    bnb_hashed_time = result[result["Method"]=="backtracking bnb"]["Time"]


    # nb = [10**i for i in nb]
    plt.plot(nb, sdp_time, label="Simple DP")
    plt.plot(nb, dp_cc_time, label="DP CC")
    plt.plot(nb, dp_lb_time, label="DP LE")
    plt.plot(nb, dp_cc_lb_time, label="DP CC LE")
    plt.plot(nb, bnb_hashed_time, label="BnB Сaсhed")
    plt.legend()
    plt.xlabel("Capacity of knapsack")
    plt.ylabel("Time, sec.")
    plt.show()


    sdp_time = result[result["Method"]=="Simple DP"]["Operations"]
    dp_cc_time = result[result["Method"]=="backtracking"]["Operations"]
    dp_lb_time = result[result["Method"]=="Sorted + Red DP"]["Operations"]
    dp_cc_lb_time = result[result["Method"]=="backtracking red"]["Operations"]
    bnb_hashed_time = result[result["Method"]=="backtracking bnb"]["Operations"]

    plt.plot(nb, sdp_time, label="Simple DP")
    plt.plot(nb, dp_cc_time, label="DP CC")
    plt.plot(nb, dp_lb_time, label="DP LE")
    plt.plot(nb, dp_cc_lb_time, label="DP CC LE")
    plt.plot(nb, bnb_hashed_time, label="BnB Сaсhed")
    plt.legend()
    plt.xlabel("Capacity of knapsack")
    plt.ylabel("Operations")
    plt.show()

    display(result)
    result.to_csv("result_random.csv")



if __name__ == "__main__":
    # methods = [backtracking, backtracking_red, backtracking_bnb, greedy, solve_knapsack_problem_dp_with_reds, solve_knapsack_problem_dp] # solve_knapsack_problem_dp_with_memory_fast  sorted_simple_bnb simple_bnb, bnb_guys
    methods = [simple_bnb, backtracking_bnb]
    run_one(methods)
    # run_all_low(methods)
    # run_random_tasks(methods)
