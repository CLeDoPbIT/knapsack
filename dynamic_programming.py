import numpy as np
import pandas as pd


def solve_knapsack_problem_dp(data, weight_knapsack):

    weights_subjects = list(data["weight"])
    value = list(data["value"])

    matrix = np.zeros((len(weights_subjects)+1, weight_knapsack+1))
    for i in range(len(weights_subjects)+1):
        for w in range(weight_knapsack+1):
            if i == 0 or w == 0:
                matrix[i][w] = 0
            elif w-weights_subjects[i-1] >= 0:
                matrix[i][w] = max(value[i-1] + matrix[i-1][w-weights_subjects[i-1]], matrix[i-1][w])
            else:
                matrix[i][w] = matrix[i-1][w]
    for i in matrix:
        print(i)

    return matrix[-1][-1]


def solve_knapsack_problem_dp_evolved(data, weight_knapsack):
    data = data.sort_values(by=["weight"])
    # todo: need make tree for fast calculating needed cells from dp table


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

    data = pd.DataFrame()
    data["weight"] = weight_items
    data["value"] = value_items

    return weight_knapsack, data


if __name__ == "__main__":
    filename = 'data/low-dimensional/f3_l-d_kp_4_20'
    weight_knapsack, data = read_data_from_file(filename)
    print(solve_knapsack_problem_dp(data, weight_knapsack))

