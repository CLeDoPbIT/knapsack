import numpy as np


def solve_knapsack_problem(value, weights_subjects, weight_knapsack):
    matrix = np.zeros((len(weights_subjects), weight_knapsack+1))
    for i in range(1, len(weights_subjects)):
        for w in range(1, weight_knapsack+1):
            if w-weights_subjects[i] >= 0:
                matrix[i][w] = max(value[i] + matrix[i-1][w-weights_subjects[i]], matrix[i-1][w])
    print(matrix)
    return matrix[-1][-1]


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
    return weight_knapsack, weight_items, value_items


if __name__ == "__main__":
    # VALUE = [0, 8, 2, 4, 5]
    # WEIGHT_SUBJECTS = [0, 5, 1, 3, 2]
    # WEIGHT_KNAPSACK = 8

    filename = 'data/low-dimensional/f3_l-d_kp_4_20'
    weight_knapsack, weight_items, value_items = read_data_from_file(filename)
    print(solve_knapsack_problem(value_items, weight_items, weight_knapsack))
