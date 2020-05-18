import numpy as np
import pandas as pd
import time
import queue
import itertools
import random


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
    # for i in matrix:
    #     print(i)
    ans = matrix[-1][-1]
    del matrix
    return ans


# def solve_knapsack_problem_dp_our():
#     data = data.sort_values(by=["weight"], ascending=False )  # todo: now quick sort, change to another sort ???


def solve_knapsack_problem_dp_our(data, weight_knapsack, na):
    data = data.sort_values(by=["weight"], ascending=False)  # todo: now quick sort, change to another sort ???

    weights_subjects = list(data["weight"])
    weights_subjects.insert(0, -1)

    values_subjects = list(data["value"])
    values_subjects.insert(0, -1)

    matrix = np.zeros((len(weights_subjects), weight_knapsack+1))
    # matrix = -matrix

    # construct left bound
    list_calc = list()

    list_calc.append([len(weights_subjects)-1, weight_knapsack])
    queue_calc_weights = np.zeros(len(weights_subjects))
    queue_calc_weights[-1] = weight_knapsack

    bt = time.perf_counter()

    while len(list_calc) != 0:  # O(n)
        current_node = list_calc.pop()
        # print(current_node)

        if current_node[0] != 0 and current_node[1] != 0:
            if current_node[1]-weights_subjects[current_node[0]] > 0:
                list_calc.append([current_node[0]-1, current_node[1]-weights_subjects[current_node[0]]])
                queue_calc_weights[current_node[0]-1]=current_node[1]-weights_subjects[current_node[0]]
            # else:
            #     list_calc.append([current_node[0]-1, current_node[1]])
            #     queue_calc_weights[current_node[0]-1]=current_node[1]

        else:
            break
    # print("Execution find REDs", time.perf_counter() - bt)

    # all_known_step = set()
    # next_row = list()
    # for i in range(1,len(weights_subjects)): # 1
    #     next, all_k_s = calc_next(i, weights_subjects, all_known_step, queue_calc_weights, weight_knapsack, next_row)
    # queue_calc_i = np.arange(0, len(queue_calc_weights))
    queue_calc_weights = queue_calc_weights.astype(int)
    queue_calc_weights = queue_calc_weights[queue_calc_weights>0].astype(int)
    queue_calc_weights = add_uniform_nodes(queue_calc_weights, na)
    # print("NUM_ADD", na)
    for i in range(1, len(weights_subjects)):
        last_weight = 0
        last_value = 0
        for i_w, w in enumerate(queue_calc_weights):
            # if i <= i_w:
                if w-weights_subjects[i] >= 0:
                    matrix[i][last_weight:] = last_value
                    matrix[i][w] = max(values_subjects[i] + matrix[i-1][w-weights_subjects[i]], matrix[i-1][w], matrix[i][w-1])
                    last_value = matrix[i][w]
                    last_weight = w

                else:
                    matrix[i][w] = matrix[i-1][w]
                    last_value = matrix[i][w]
                    last_weight = w

    # print(queue_calc_weights)
    # for i in matrix:
    #     print(i)

    return matrix[-1][-1]


def add_uniform_nodes(queue_calc_weights, percent):
    new = queue_calc_weights.copy()
    step = 0
    for i in range(1, len(queue_calc_weights)):
        number_nodes = int((queue_calc_weights[i]-queue_calc_weights[i-1])/100*percent)
        positions = np.array(sorted(random.sample(range(queue_calc_weights[i-1]+1, queue_calc_weights[i]), number_nodes)))
        if len(positions) > 0:
            new = np.concatenate([new[:i+step], positions, new[i+step:]])
            step += len(positions)
    return new


# def calc_sum_subject(ws, global_)
#
#
# def calc_next(i, ws, all_k_s, stops, weight_knapsack, next_row):  # todo: DP too
#     next = next_row.copy()
#     found = False
#     variants = np.arange(1, i+1)
#     current_var = len(variants)
#     current_var_all = current_var
#     while not found and current_var > 0:
#         var = set(itertools.combinations(variants, current_var))  # todo: need change to my algorithm2
#         # var = var - all_k_s
#         for i in var:
#             curr_w = sum([ws[j] for j in i])
#             if curr_w <= weight_knapsack:
#                 # all_k_s.add(i)
#                 if curr_w < stops[current_var_all]:
#                     next.append(curr_w)
#                     found = True
#                     break
#                 next.append(curr_w)
#         current_var -= 1
#     return next, all_k_s



def recursive_dp_evolved(data, i, w, ws, vs):
    if i == 0 or w == 0:
        return 0

    if data[i - 1][w] == -1:
        data[i-1][w] = recursive_dp_evolved(data, i-1, w, ws, vs)
    if ws[i] > w:
        data[i][w] = data[i-1][w]
    else:
        if data[i-1][w-ws[i]] == -1:
            data[i-1][w-ws[i]] = recursive_dp_evolved(data, i-1, w-ws[i], ws, vs)
        data[i][w] = max(data[i-1][w], data[i-1][w-ws[i]]+vs[i])

    return data[i][w]


def solve_knapsack_problem_dp_evolved(data, weight_knapsack):

    weights_subjects = list(data["weight"])
    tmp = weights_subjects[0]
    weights_subjects[0] = -1
    weights_subjects.append(tmp)

    values_subjects = list(data["value"])
    tmp = values_subjects[0]
    values_subjects[0] = -1
    values_subjects.append(tmp)

    matrix = np.ones((len(weights_subjects), weight_knapsack+1))
    matrix = -matrix

    recursive_dp_evolved(matrix, len(weights_subjects)-1, weight_knapsack, weights_subjects, values_subjects)

    # for i in matrix:
    #     print(i)

    return matrix[-1][-1]


def solve_knapsack_problem_dp_evolved_sorted(data, weight_knapsack):
    data = data.sort_values(by=["weight"], ascending=False)

    weights_subjects = list(data["weight"])
    tmp = weights_subjects[0]
    weights_subjects[0] = -1
    weights_subjects.append(tmp)

    values_subjects = list(data["value"])
    tmp = values_subjects[0]
    values_subjects[0] = -1
    values_subjects.append(tmp)

    matrix = np.ones((len(weights_subjects), weight_knapsack+1))
    matrix = -matrix

    recursive_dp_evolved(matrix, len(weights_subjects)-1, weight_knapsack, weights_subjects, values_subjects)

    # for i in matrix:
    #     print(i)

    return matrix[-1][-1]



def solve_knapsack_problem_dp_evolved_2(data, weight_knapsack):
    weights_subjects = list(data["weight"])
    tmp = weights_subjects[0]
    weights_subjects[0] = -1
    weights_subjects.append(tmp)

    values_subjects = list(data["value"])
    tmp = values_subjects[0]
    values_subjects[0] = -1
    values_subjects.append(tmp)

    matrix = np.ones((len(weights_subjects), weight_knapsack+1))
    matrix = -matrix

    queue_calc = list()
    list_calc = list()
    list_calc.append([len(weights_subjects)-1, weight_knapsack])

    while len(list_calc) != 0:
        current_node = list_calc.pop()
        # print(current_node)
        if not current_node in queue_calc:
            queue_calc.append(current_node)

            if current_node[0] != 0 and current_node[1] != 0:
                # queue_calc.append([current_node[0]-1, current_node[1]])
                list_calc.append([current_node[0]-1, current_node[1]])
                if current_node[1]-weights_subjects[current_node[0]] > 0:# not take item
                    # queue_calc.append([current_node[0]-1, current_node[1]-weights_subjects[current_node[0]]])
                    list_calc.append([current_node[0]-1, current_node[1]-weights_subjects[current_node[0]]])

    # print("done")
    while len(queue_calc) != 0:
        current_node = queue_calc.pop()
        i = current_node[0]
        w = current_node[1]
        if i == 0 or w == 0:
            matrix[i][w] = 0
        elif w - weights_subjects[i] >= 0:
            matrix[i][w] = max(values_subjects[i] + matrix[i - 1][w - weights_subjects[i]], matrix[i - 1][w])
        else:
            matrix[i][w] = matrix[i - 1][w]

    return matrix[-1][-1]


def solve_knapsack_problem_dp_evolved_2_sorted(data, weight_knapsack):

    data = data.sort_values(by=["weight"], ascending=False)

    weights_subjects = list(data["weight"])
    tmp = weights_subjects[0]
    weights_subjects[0] = -1
    weights_subjects.append(tmp)

    values_subjects = list(data["value"])
    tmp = values_subjects[0]
    values_subjects[0] = -1
    values_subjects.append(tmp)

    matrix = np.ones((len(weights_subjects), weight_knapsack+1))
    matrix = -matrix

    queue_calc = list()
    list_calc = list()
    list_calc.append([len(weights_subjects)-1, weight_knapsack])

    while len(list_calc) != 0:
        current_node = list_calc.pop()
        # print(current_node)
        if not current_node in queue_calc:
            queue_calc.append(current_node)

            if current_node[0] != 0 and current_node[1] != 0:
                # queue_calc.append([current_node[0]-1, current_node[1]])
                list_calc.append([current_node[0]-1, current_node[1]])
                if current_node[1]-weights_subjects[current_node[0]] > 0:# not take item
                    # queue_calc.append([current_node[0]-1, current_node[1]-weights_subjects[current_node[0]]])
                    list_calc.append([current_node[0]-1, current_node[1]-weights_subjects[current_node[0]]])

    # print("done")
    while len(queue_calc) != 0:
        current_node = queue_calc.pop()
        i = current_node[0]
        w = current_node[1]
        if i == 0 or w == 0:
            matrix[i][w] = 0
        elif w - weights_subjects[i] >= 0:
            matrix[i][w] = max(values_subjects[i] + matrix[i - 1][w - weights_subjects[i]], matrix[i - 1][w])
        else:
            matrix[i][w] = matrix[i - 1][w]

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
    weight_items.pop()
    value_items.pop()
    data = pd.DataFrame()
    data["weight"] = weight_items
    data["value"] = value_items

    return weight_knapsack, data


class Greedy:

    def __init__(self, knapsack):
        assert isinstance(knapsack, dict)
        self.capacity = knapsack['capacity'][0]
        self.weights  = knapsack['weights']
        self.profits  = knapsack['profits']
        self.n        = len(knapsack['weights'])

    def name(self):
        return 'Greedy'

    def solve(self):

        # value = [(x[0], x[2] / x[1]) for x in zip(np.arange(self.n),
        #                                           self.weights,
        #                                           self.profits)]

        value = list()
        for i in range(len(self.weights)):
            value.append((i, self.profits[i]/self.weights[i]))

        value = sorted(value, key=lambda x: x[1], reverse=True)

        cur_weight = 0
        optim_set  = np.zeros(self.n, dtype=np.int64)

        for v in value:
            if cur_weight + self.weights[v[0]] <= self.capacity:
                optim_set[v[0]] = 1
                cur_weight += self.weights[v[0]]
            else:
                continue
        return optim_set.tolist()


def compute_knapsack(knapsack, optimal, verbose=False):

    ttl_weight = sum([item[0] * item[1] for item in zip(knapsack['weights'], optimal)])
    ttl_profit = sum([item[0] * item[1] for item in zip(knapsack['profits'], optimal)])

    if ttl_weight > knapsack['capacity'][0] and verbose:
        print('Total weight exceed knapsack capacity ({} > {})'.format(
               ttl_weight, knapsack['capacity'][0]))

    return ttl_weight, ttl_profit





import sys
sys.setrecursionlimit(10**6)

if __name__ == "__main__":
    # filename = 'data/low-dimensional/f10_l-d_kp_20_879'
    filename = 'data/large_scale/knapPI_1_10000_1000_1'
    weight_knapsack, data = read_data_from_file(filename)

    ans_t = list()
    ans_v = list()

    bt = time.perf_counter()
    gred = Greedy({'capacity':[weight_knapsack],
                  'weights': list(data["weight"]),
                              'profits':list(data["value"])})
    opt = gred.solve()
    ans = compute_knapsack({'capacity':[weight_knapsack],
                  'weights': list(data["weight"]),
                              'profits':list(data["value"])}, opt)

    ans_t.append(time.perf_counter() - bt)
    ans_v.append(ans[1])
    print("Greedy", ans_v[-1], ans_t[-1] )

    # bt = time.perf_counter()
    # ans_v.append(solve_knapsack_problem_dp(data, weight_knapsack))
    # ans_t.append(time.perf_counter()-bt)
    # print("DP",ans_v[-1], ans_t[-1] )
    #
    # bt = time.perf_counter()
    # ans_v.append(solve_knapsack_problem_dp_evolved(data, weight_knapsack))
    # ans_t.append(time.perf_counter()-bt)
    # print("DP recurive",ans_v[-1], ans_t[-1] )

    # bt = time.perf_counter()
    # ans_v.append(solve_knapsack_problem_dp_evolved_2(data, weight_knapsack))
    # ans_t.append(time.perf_counter()-bt)
    # print("DP stack",ans_v[-1], ans_t[-1])

    bt = time.perf_counter()
    ans_v.append(solve_knapsack_problem_dp_our(data, weight_knapsack, na=20))
    ans_t.append(time.perf_counter()-bt)
    print("DP our 20", ans_v[-1], ans_t[-1])

    bt = time.perf_counter()
    ans_v.append(solve_knapsack_problem_dp_our(data, weight_knapsack, na=50))
    ans_t.append(time.perf_counter()-bt)
    print("DP our 50",   ans_v[-1], ans_t[-1])

    bt = time.perf_counter()
    ans_v.append(solve_knapsack_problem_dp_our(data, weight_knapsack, na=80))
    ans_t.append(time.perf_counter()-bt)
    print("DP our 80",   ans_v[-1], ans_t[-1])

    bt = time.perf_counter()
    ans_v.append(solve_knapsack_problem_dp_our(data, weight_knapsack, na=99))
    ans_t.append(time.perf_counter()-bt)
    print("DP our 99",   ans_v[-1], ans_t[-1])

    print(ans_t)
    print(ans_v)









    # bt = time.perf_counter()
    # print(solve_knapsack_problem_dp_evolved_sorted(data, weight_knapsack))
    # print("Execution time DP evolbed recurive sorted", time.perf_counter() - bt)
    #
    # bt = time.perf_counter()
    # print(solve_knapsack_problem_dp_evolved_2(data, weight_knapsack))
    # print("Execution time DP evolved stack", time.perf_counter() - bt)
    #
    # bt = time.perf_counter()
    # print(solve_knapsack_problem_dp_evolved_2(data, weight_knapsack))
    # print("Execution time DP evolved stack sorted", time.perf_counter() - bt)



