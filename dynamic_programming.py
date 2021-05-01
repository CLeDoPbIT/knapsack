import numpy as np
import time


def solve_knapsack_problem_dp(data, weight_knapsack):

    weights_subjects = list(data["weight"])
    value = list(data["value"])

    matrix = np.zeros((len(weights_subjects), weight_knapsack+1))
    time_result_one_max = []
    counter_cells = 0
    for i in range(1, len(weights_subjects)):
        # print(i, weight_knapsack+1 )
        for w in range(1, weight_knapsack+1):
            counter_cells += 1
            if i == 0 or w == 0:
                matrix[i][w] = 0
            elif w-weights_subjects[i] >= 0:
                bt1 = time.perf_counter()
                # print(w)
                matrix[i][w] = max(value[i] + matrix[i-1][w-weights_subjects[i]], matrix[i-1][w])
                time_result_one_max.append(time.perf_counter() - bt1)

            else:
                matrix[i][w] = matrix[i-1][w]
    print("time_result_one_max DP", sum(time_result_one_max)/len(time_result_one_max), len(time_result_one_max), counter_cells, len(weights_subjects)*weight_knapsack)

    ans = matrix[-1][-1]
    del matrix
    return ans, "Simple DP", counter_cells


def solve_knapsack_problem_dp_with_reds(data, weight_knapsack):
    data = data.sort_values(by=["weight"], ascending=False)  # todo: now quick sort, change to another sort ???

    weights_subjects = list(data["weight"])
    weights_subjects.insert(0, weights_subjects[-1])
    weights_subjects.pop(-1)

    values_subjects = list(data["value"])
    values_subjects.insert(0, values_subjects[-1])
    values_subjects.pop(-1)

    matrix = np.zeros((len(weights_subjects), weight_knapsack + 1))

    # construct left bound
    list_calc = list()

    list_calc.append([len(weights_subjects) - 1, weight_knapsack])
    queue_calc_weights = np.zeros(len(weights_subjects))
    queue_calc_weights[-1] = weight_knapsack


    while len(list_calc) != 0:  # O(n)
        current_node = list_calc.pop()

        if current_node[0] != 0 and current_node[1] != 0:
            if current_node[1] - weights_subjects[current_node[0]] > 0:
                list_calc.append([current_node[0] - 1, current_node[1] - weights_subjects[current_node[0]]])
                queue_calc_weights[current_node[0] - 1] = current_node[1] - weights_subjects[current_node[0]]
            else:
                list_calc.append([current_node[0] - 1, current_node[1]])
                queue_calc_weights[current_node[0] - 1] = current_node[1]

        else:
            break

    queue_calc_weights = queue_calc_weights.astype(int)
    counter_cells = 0
    time_result_one_max = []

    for i in range(1, len(weights_subjects)):
        for w in range(queue_calc_weights[i], weight_knapsack+1):
            counter_cells += 1
            if i == 0 or w == 0:
                matrix[i][w] = 0
            elif w-weights_subjects[i] >= 0:
                bt1 = time.perf_counter()
                matrix[i][w] = max(values_subjects[i] + matrix[i-1][w-weights_subjects[i]], matrix[i-1][w])
                time_result_one_max.append(time.perf_counter() - bt1)

            else:
                matrix[i][w] = matrix[i-1][w]
    ans = matrix[-1][-1]
    print("time_result_one_max DPR", sum(time_result_one_max)/len(time_result_one_max), len(time_result_one_max), counter_cells)

    del matrix
    return ans, "Sorted + Red DP", counter_cells


def solve_knapsack_problem_dp_with_memory(data, weight_knapsack):

    data = data.sort_values(by=["weight"], ascending=False)  # todo: now quick sort, change to another sort ???

    weights_subjects = list(data["weight"])
    weights_subjects.insert(0, weights_subjects[-1])
    weights_subjects.pop(-1)

    values_subjects = list(data["value"])
    values_subjects.insert(0, values_subjects[-1])
    values_subjects.pop(-1)

    matrix = np.zeros((len(weights_subjects), weight_knapsack+1))


    list_calc = list()

    list_calc.append([len(weights_subjects) - 1, weight_knapsack])
    queue_calc_weights = np.zeros(len(weights_subjects))
    queue_calc_weights[-1] = weight_knapsack


    while len(list_calc) != 0:  # O(n)
        current_node = list_calc.pop()

        if current_node[0] != 0 and current_node[1] != 0:
            if current_node[1] - weights_subjects[current_node[0]] > 0:
                list_calc.append([current_node[0] - 1, current_node[1] - weights_subjects[current_node[0]]])
                queue_calc_weights[current_node[0]-1] = current_node[1] - weights_subjects[current_node[0]]
            else:
                list_calc.append([current_node[0] - 1, current_node[1]])
                queue_calc_weights[current_node[0]-1] = current_node[1]

        else:
            break

    queue_calc_weights = queue_calc_weights.astype(int)
    # minimum = min(queue_calc_weights)
    # queue_calc_weights = [minimum for i in queue_calc_weights if i==0]


    time_result_max = []
    time_result_green = []
    time_result_one_max = []
    last_row = []
    counter_cells = 0

    last_row.append(weights_subjects[1])
    matrix[1][weights_subjects[1]:] = values_subjects[1]

    for i in range(2, len(weights_subjects)): #todo: fix bug then huge weight of knapsack and there are no green in right of red cells. Need to add one cell before red.
        current_row = last_row.copy()

        current_row.append(weights_subjects[i])
        for weight in last_row:
            current_row.append(weight+weights_subjects[i]) # last_row[weight]+values_subjects[i]

        bt = time.perf_counter()
        current_row = np.array(current_row)
        current_row = sorted(list(set(current_row[np.logical_and(current_row >= queue_calc_weights[i-1], current_row<=weight_knapsack)]))) # current_row>= queue_calc_weights[i]
        # print(i, len(set(current_row)))
        time_result_green.append(time.perf_counter()-bt)
        for green_nodes in current_row:
            # print(green_nodes)
            counter_cells += 1
            bt1 = time.perf_counter()
            matrix[i][green_nodes] = max(values_subjects[i] + matrix[i - 1][green_nodes - weights_subjects[i]], matrix[i - 1][green_nodes])
            time_result_one_max.append(time.perf_counter()-bt1)
            # matrix[i][green_nodes:] = matrix[i][green_nodes]

        time_result_max.append(time.perf_counter()-bt)

        last_row = list(current_row).copy()

    # print("time_result_max!", sum(time_result_max))
    # print("time_result_green!", sum(time_result_green))
    print("time_result_one_max DPRG", sum(time_result_one_max)/len(time_result_one_max), len(time_result_one_max), counter_cells)

    return matrix[-1][-1], "Memory DP", counter_cells


def solve_knapsack_problem_dp_with_memory_fast(data, weight_knapsack):

    data = data.sort_values(by=["weight"], ascending=False)  # todo: now quick sort, change to another sort ???

    weights_subjects = list(data["weight"])
    weights_subjects.insert(0, weights_subjects[-1])
    weights_subjects.pop(-1)

    values_subjects = list(data["value"])
    values_subjects.insert(0, values_subjects[-1])
    values_subjects.pop(-1)

    matrix = np.zeros((len(weights_subjects), weight_knapsack+1))


    list_calc = list()

    list_calc.append([len(weights_subjects) - 1, weight_knapsack])
    queue_calc_weights = np.zeros(len(weights_subjects))
    queue_calc_weights[-1] = weight_knapsack

    while len(list_calc) != 0:  # O(n)
        current_node = list_calc.pop()

        if current_node[0] != 0 and current_node[1] != 0:
            if current_node[1] - weights_subjects[current_node[0]] > 0:
                list_calc.append([current_node[0] - 1, current_node[1] - weights_subjects[current_node[0]]])
                queue_calc_weights[current_node[0]-1] = current_node[1] - weights_subjects[current_node[0]]
            else:
                list_calc.append([current_node[0] - 1, current_node[1]])
                queue_calc_weights[current_node[0]-1] = current_node[1]

        else:
            break

    queue_calc_weights = queue_calc_weights.astype(int)

    time_result_max = []
    time_result_green = []
    time_result_one_max = []
    last_row = []
    counter_cells = 0

    last_row.append(weights_subjects[1])
    # matrix[1][weights_subjects[1]:] = values_subjects[1]

    result = {0:0}

    for i in range(1, len(weights_subjects)):
        current_weights = list(result.keys())
        tmp_result = result.copy()
        # current_result = dict()
        i_cw = 0
        while i_cw < len(current_weights):
            if current_weights[i_cw] + weights_subjects[i] <= weight_knapsack:  # queue_calc_weights[i] <=
                tmp_result[current_weights[i_cw] + weights_subjects[i]] = 0
                # current_result[current_weights[i_cw] + weights_subjects[i]] = 0
            i_cw += 1

        current_weights = sorted(list(tmp_result.keys())[1:])
        # current_weights = list(current_result.keys())

        for w in current_weights:
            counter_cells += 1
            if i == 0 or w == 0:
                matrix[i][w] = 0
            elif w-weights_subjects[i] >= 0:
                bt1 = time.perf_counter()
                matrix[i][w:] = max(values_subjects[i] + matrix[i-1][w-weights_subjects[i]], matrix[i-1][w])
                time_result_one_max.append(time.perf_counter() - bt1)
            else:
                matrix[i][w] = matrix[i-1][w]
        result = tmp_result

    # print("time_result_max!", sum(time_result_max))
    # print("time_result_green!", sum(time_result_green))
    print("time_result_one_max DPG", sum(time_result_one_max)/len(time_result_one_max), len(time_result_one_max), counter_cells)

    return matrix[-1][-1], "DPG", counter_cells



# def solve_knapsack_problem_dp_with_memory(data, weight_knapsack):
#
#     data = data.sort_values(by=["weight"], ascending=False)  # todo: now quick sort, change to another sort ???
#
#     weights_subjects = list(data["weight"])
#     weights_subjects.insert(0, weights_subjects[-1])
#     weights_subjects.pop(-1)
#
#     values_subjects = list(data["value"])
#     values_subjects.insert(0, values_subjects[-1])
#     values_subjects.pop(-1)
#
#     matrix = np.zeros((len(weights_subjects), weight_knapsack+1))
#
#
#     list_calc = list()
#
#     list_calc.append([len(weights_subjects) - 1, weight_knapsack])
#     queue_calc_weights = np.zeros(len(weights_subjects))
#     queue_calc_weights[-1] = weight_knapsack
#
#
#     while len(list_calc) != 0:  # O(n)
#         current_node = list_calc.pop()
#
#         if current_node[0] != 0 and current_node[1] != 0:
#             if current_node[1] - weights_subjects[current_node[0]] > 0:
#                 list_calc.append([current_node[0] - 1, current_node[1] - weights_subjects[current_node[0]]])
#                 queue_calc_weights[current_node[0]-1] = current_node[1] - weights_subjects[current_node[0]]
#             else:
#                 list_calc.append([current_node[0] - 1, current_node[1]])
#                 queue_calc_weights[current_node[0]-1] = current_node[1]
#
#         else:
#             break
#
#     queue_calc_weights = queue_calc_weights.astype(int)
#     # minimum = min(queue_calc_weights)
#     # queue_calc_weights = [minimum for i in queue_calc_weights if i==0]
#
#
#     time_result_max = []
#     time_result_green = []
#     time_result_one_max = []
#     last_row = set()
#
#     last_row.add(weights_subjects[1])
#     matrix[1][weights_subjects[1]:] = values_subjects[1]
#     for i in range(2, len(weights_subjects)):
#         current_row = last_row.copy()
#
#         current_row.add(weights_subjects[i])
#         for weight in last_row:
#             if queue_calc_weights[i-1] <= weight+weights_subjects[i] <=weight_knapsack:
#                 green_nodes = weight + weights_subjects[i]
#                 current_row.add(green_nodes)
#
#                 bt1 = time.perf_counter()
#                 matrix[i][green_nodes] = max(
#                     values_subjects[i] + matrix[i - 1][green_nodes - weights_subjects[i]],
#                     matrix[i - 1][green_nodes])
#                 time_result_one_max.append(time.perf_counter() - bt1)
#                 matrix[i][green_nodes:] = matrix[i][green_nodes]
#                 current_row.add(weight+weights_subjects[i]) # last_row[weight]+values_subjects[i]
#
#         # matrix[i][current_row[-1]:] = max(
#         #     values_subjects[i] + matrix[i - 1][current_row[-1] - weights_subjects[i]],
#         #     matrix[i - 1][current_row[-1]])
#
#         # bt = time.perf_counter()  # TODO: NEED TO INSERT MATRIX
#         # current_row = np.array(current_row)
#         # current_row = sorted(list(set(current_row[np.logical_and(current_row >= queue_calc_weights[i-1], current_row<=weight_knapsack)]))) # current_row>= queue_calc_weights[i]
#         # # print(i, len(set(current_row)))
#         # time_result_green.append(time.perf_counter()-bt)
#         # for green_nodes in range(1, len(current_row)):
#         #     # print(green_nodes)
#         #     bt1 = time.perf_counter()
#         #     matrix[i][current_row[green_nodes-1]] = max(values_subjects[i] + matrix[i - 1][current_row[green_nodes-1] - weights_subjects[i]], matrix[i - 1][current_row[green_nodes-1]])
#         #     time_result_one_max.append(time.perf_counter()-bt1)
#         #     matrix[i][current_row[green_nodes-1]:current_row[green_nodes]] = matrix[i][current_row[green_nodes-1]]
#         # matrix[i][current_row[-1]:] = max(
#         #     values_subjects[i] + matrix[i - 1][current_row[-1] - weights_subjects[i]],
#         #     matrix[i - 1][current_row[-1]])
#
#         # time_result_max.append(time.perf_counter()-bt)
#
#         last_row = current_row.copy()
#
#     print("time_result_max!", sum(time_result_max))
#     print("time_result_green!", sum(time_result_green))
#     print("time_result_one_max", sum(time_result_one_max)/len(time_result_one_max), len(time_result_one_max))
#
#     return matrix[-1][-1], "Memory DP"



# def solve_knapsack_problem_dp_with_memory(data, weight_knapsack):
#
#     data = data.sort_values(by=["weight"], ascending=False)  # todo: now quick sort, change to another sort ???
#
#     weights_subjects = list(data["weight"])
#     weights_subjects.insert(0, weights_subjects[-1])
#     weights_subjects.pop(-1)
#
#     values_subjects = list(data["value"])
#     values_subjects.insert(0, values_subjects[-1])
#     values_subjects.pop(-1)
#
#     matrix = np.zeros((len(weights_subjects), weight_knapsack+1))
#
#     last_row = dict()
#     current_row = dict()
#
#     matrix[1][weights_subjects[1]:] = values_subjects[1]
#     last_row[weights_subjects[1]] = values_subjects[1]
#
#     for i in range(2, len(weights_subjects)):
#         ids_founded = []
#         current_row = last_row.copy()
#
#         current_row[weights_subjects[i]] = values_subjects[i] # TODO: solve assumption that all weights are unique
#         # matrix[i][weights_subjects[i]] = values_subjects[i]
#         ids_founded.append(weights_subjects[i])
#         for weight in last_row:
#             if weight+weights_subjects[i] <= weight_knapsack:
#                 # if weight+weights_subjects[i] in last_row:
#                     if last_row[weight]+values_subjects[i] > matrix[i-1][weight+weights_subjects[i]]:
#                         current_row[weight+weights_subjects[i]] = last_row[weight]+values_subjects[i]
#                         # matrix[i][weight+weights_subjects[i]] = current_row[weight+weights_subjects[i]]
#                         ids_founded.append(weight+weights_subjects[i])
#
#                 # else:
#                 #     current_row[weight + weights_subjects[i]] = last_row[weight] + values_subjects[i]
#                     # matrix[i][weight + weights_subjects[i]] = current_row[weight + weights_subjects[i]]
#                     # ids_founded.append(weight + weights_subjects[i])
#
#         ids_founded = sorted(ids_founded)
#         for ids in range(1, len(ids_founded)):
#             matrix[i][ids_founded[ids-1]:ids_founded[ids]] = current_row[ids_founded[ids-1]]
#         matrix[i][ids_founded[ids]:] = current_row[ids_founded[ids]]
#         last_row = current_row.copy()
#
#     # print(current_row)
#     return matrix[-1][-1], "Memory DP"
#     # ans = matrix[-1][-1]
#     # del matrix
#     # return ans, "Simple DP"


# def solve_knapsack_problem_dp_with_memory(data, weight_knapsack):
#
#     data = data.sort_values(by=["weight"], ascending=False)  # todo: now quick sort, change to another sort ???
#
#     weights_subjects = list(data["weight"])
#     weights_subjects.insert(0, weights_subjects[-1])
#     weights_subjects.pop(-1)
#
#     values_subjects = list(data["value"])
#     values_subjects.insert(0, values_subjects[-1])
#     values_subjects.pop(-1)
#
#     matrix = np.zeros((len(weights_subjects), weight_knapsack+1))
#
#     last_row = dict()
#     current_row = dict()
#
#     for i in range(1, len(weights_subjects)):
#         current_row = last_row.copy()
#
#         current_row[weights_subjects[i]] = values_subjects[i] # TODO: solve assumption that all weights are unique
#
#         for weight in last_row:
#             if weight+weights_subjects[i] <= weight_knapsack:
#                 if weight+weights_subjects[i] in last_row:
#                     if last_row[weight]+values_subjects[i] > last_row[weight+weights_subjects[i]]:
#                         current_row[weight+weights_subjects[i]] = last_row[weight]+values_subjects[i]
#                         # matrix[i][weight+weights_subjects[i]] = current_row[weight+weights_subjects[i]]
#                 else:
#                     current_row[weight + weights_subjects[i]] = last_row[weight] + values_subjects[i]
#                     # matrix[i][weight + weights_subjects[i]] = current_row[weight + weights_subjects[i]]
#
#         last_row = current_row.copy()
#
#     # print(current_row)
#     return current_row[weight_knapsack], "Memory DP"
#     # ans = matrix[-1][-1]
#     # del matrix
#     # return ans, "Simple DP"
