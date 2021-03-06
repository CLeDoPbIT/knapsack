from greedy import greedy
import numpy as np

def backtracking(data, weight_knapsack):
    result = {0:0}
    items = {0:[0]}
    data = data.drop([0])

    weights_subjects = list(data["weight"])
    values_subjects = list(data["value"])
    counter_cells = 0

    for i in range(0, len(weights_subjects)):
        current_weights = list(result.keys())
        i_cw = 0
        tmp_result = result.copy()
        while i_cw < len(current_weights):
            if current_weights[i_cw] + weights_subjects[i] <= weight_knapsack:
                counter_cells += 1
                if current_weights[i_cw]+weights_subjects[i] in tmp_result:
                    result[current_weights[i_cw] + weights_subjects[i]] = max(tmp_result[current_weights[i_cw] + weights_subjects[i]], tmp_result[current_weights[i_cw]] + values_subjects[i])
                    if result[current_weights[i_cw] + weights_subjects[i]] <= result[current_weights[i_cw]] + values_subjects[i]:
                        items[current_weights[i_cw] + weights_subjects[i]] = items[current_weights[i_cw]] + [i]
                else:
                    result[current_weights[i_cw]+weights_subjects[i]] = tmp_result[current_weights[i_cw]] + values_subjects[i]
                    items[current_weights[i_cw]+weights_subjects[i]] = items[current_weights[i_cw]] + [i]
            i_cw += 1
    print("Backtracking", counter_cells)

    answer = max(list(result.values()))
    return answer, "backtracking", counter_cells


def calculate_sorted_ub(current_profit, current_weight, weight_knapsack, ratio, current_item):
    if current_item < 0:
        ub = current_profit + (weight_knapsack-current_weight)*(ratio[0])  # todo: problem if weight_knapsack >> current_weight then ub will be huge. Maybe can be fixed if use min(weight_knapsack-current_weight, remaining weight of items)
    else:
        ub = current_profit + (weight_knapsack - current_weight) * (ratio[current_item])

    return ub


def backtracking_bnb(data, weight_knapsack):

    greedy_ub, _, __ = greedy(data, weight_knapsack)
    data = data.drop([0])
    data["ratio"] = data["value"]/data["weight"]
    data = data.sort_values(by=["ratio"], ascending=False)

    result = {0:0}
    items = {0:[-1]}

    weights_subjects = list(data["weight"])
    values_subjects = list(data["value"])
    ratio = list(data["ratio"])

    ub = calculate_sorted_ub(0, 0, weight_knapsack, ratio, 0)

    if greedy_ub< ub:
        ub = greedy_ub

    counter_cells = 0
    for i in range(len(weights_subjects)):
        current_weights = list(result.keys())
        i_cw = 0
        tmp_result = result.copy()
        while i_cw < len(current_weights):
            node_ub = calculate_sorted_ub(tmp_result[current_weights[i_cw]], current_weights[i_cw], weight_knapsack, ratio, i)

            if node_ub >= ub:
                if tmp_result[current_weights[i_cw]] >= ub:  # check best solution too late, maybe can speed up it. Slow DP in memory mod.Can use red for lower bound here.
                    ub = tmp_result[current_weights[i_cw]]

                if current_weights[i_cw] + weights_subjects[i] <= weight_knapsack:
                    counter_cells += 1
                    if current_weights[i_cw]+weights_subjects[i] in tmp_result:
                        result[current_weights[i_cw] + weights_subjects[i]] = max(tmp_result[current_weights[i_cw] + weights_subjects[i]], tmp_result[current_weights[i_cw]] + values_subjects[i])
                        if result[current_weights[i_cw] + weights_subjects[i]] <= result[current_weights[i_cw]] + values_subjects[i]:
                            items[current_weights[i_cw] + weights_subjects[i]] = items[current_weights[i_cw]] + [i]
                    else:
                        result[current_weights[i_cw]+weights_subjects[i]] = tmp_result[current_weights[i_cw]] + values_subjects[i]
                        items[current_weights[i_cw]+weights_subjects[i]] = items[current_weights[i_cw]] + [i]
            i_cw += 1
    print("Backtracking BnB", counter_cells)

    answer = max(list(result.values()))
    return answer, "backtracking bnb", counter_cells


def backtracking_bnb_red(data, weight_knapsack):

    greedy_ub, _, __ = greedy(data, weight_knapsack)
    data = data.drop([0])
    data["ratio"] = data["value"]/data["weight"]
    data = data.sort_values(by=["weight"], ascending=False)

    result = {0:0}
    items = {0:[0]}

    weights_subjects = list(data["weight"])
    values_subjects = list(data["value"])
    ratio = list(data["ratio"])

    ub = calculate_sorted_ub(0, 0, weight_knapsack, ratio, 0)

    if greedy_ub< ub:
        ub = greedy_ub

    counter_cells = 0


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

    for i in range(len(weights_subjects)):

        current_weights = list(result.keys())
        i_cw = 0
        tmp_result = result.copy()
        while i_cw < len(current_weights):
            node_ub = calculate_sorted_ub(tmp_result[current_weights[i_cw]], current_weights[i_cw], weight_knapsack, ratio, i)

            if node_ub >= ub:
                if tmp_result[current_weights[i_cw]] >= ub:
                    ub = tmp_result[current_weights[i_cw]]

                if queue_calc_weights[i] <= current_weights[i_cw] + weights_subjects[i] <= weight_knapsack:
                    counter_cells += 1
                    if current_weights[i_cw]+weights_subjects[i] in tmp_result:
                        result[current_weights[i_cw] + weights_subjects[i]] = max(tmp_result[current_weights[i_cw] + weights_subjects[i]], tmp_result[current_weights[i_cw]] + values_subjects[i])
                        if result[current_weights[i_cw] + weights_subjects[i]] <= result[current_weights[i_cw]] + values_subjects[i]:
                            items[current_weights[i_cw] + weights_subjects[i]] = items[current_weights[i_cw]] + [i]
                    else:
                        result[current_weights[i_cw]+weights_subjects[i]] = tmp_result[current_weights[i_cw]] + values_subjects[i]
                        items[current_weights[i_cw]+weights_subjects[i]] = items[current_weights[i_cw]] + [i]
            i_cw += 1
    print("Backtracking BnB red", counter_cells)

    answer = max(list(result.values()))
    return answer, "backtracking bnb red", counter_cells


def backtracking_red(data, weight_knapsack):
    result = {0:0}
    items = {0:{0}}
    data = data.drop([0])
    data = data.sort_values(by=["weight"], ascending=False)  # todo: now quick sort, change to another sort ???

    counter_cells = 0

    weights_subjects = list(data["weight"])
    weights_subjects.insert(0, weights_subjects[-1])
    # weights_subjects.pop(-1)

    values_subjects = list(data["value"])
    values_subjects.insert(0, values_subjects[-1])
    # values_subjects.pop(-1)

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
                list_calc.append([current_node[0] - 1, current_node[1]]) # current_node[1]
                queue_calc_weights[current_node[0]-1] = current_node[1]

        else:
            break

    queue_calc_weights = queue_calc_weights.astype(int)
    queue_calc_weights[-1] = queue_calc_weights[0]
    for i in range(1, len(weights_subjects)):
        current_weights = list(result.keys())
        i_cw = 0
        tmp_result = result.copy()
        while i_cw < len(current_weights):
            if current_weights[i_cw]+weights_subjects[i] in tmp_result:
                if (queue_calc_weights[len(items[current_weights[i_cw] + weights_subjects[i]]) - 1] <= current_weights[i_cw] + weights_subjects[i] <= weight_knapsack) :  # len(items[current_weights[i_cw] + weights_subjects[i]]) - 1 >= i
                    counter_cells += 1
                    result[current_weights[i_cw] + weights_subjects[i]] = max(tmp_result[current_weights[i_cw] + weights_subjects[i]], tmp_result[current_weights[i_cw]] + values_subjects[i])
                    items[current_weights[i_cw] + weights_subjects[i]] = (items[current_weights[i_cw]] | {i})
            else:
                if current_weights[i_cw] + weights_subjects[i] <= weight_knapsack:
                    counter_cells += 1
                    result[current_weights[i_cw]+weights_subjects[i]] = tmp_result[current_weights[i_cw]] + values_subjects[i]
                    items[current_weights[i_cw]+weights_subjects[i]] = (items[current_weights[i_cw]] | {i})

            i_cw += 1
    print("Backtracking", counter_cells)

    answer = max(list(result.values()))
    return answer, "backtracking red", counter_cells
