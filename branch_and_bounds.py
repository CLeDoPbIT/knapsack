from greedy import greedy
import queue


def calculate_ub(current_profit, current_weight, weight_knapsack, ratio, current_item):
    if current_item < 0:
        ub = current_profit + (weight_knapsack-current_weight)*max(ratio)
    else:
        ub = current_profit + (weight_knapsack-current_weight)*max(ratio[current_item:])
    return ub


def simple_bnb(data, weight_knapsack):
    weights_subjects = list(data["weight"][1:])
    values_subjects = list(data["value"][1:])
    n = len(values_subjects)
    greedy_ub, _ = greedy(data, weight_knapsack)
    queue_bnb = list()
    found_optimum = False

    node = [0, 0, 0]  # weight, profit, level of item

    # todo: sorted profit ratio and use it
    ratio = []
    for i in range(len(weights_subjects)):
        ratio.append(values_subjects[i]/weights_subjects[i])

    queue_bnb.append(node)

    ub = calculate_ub(0, 0, weight_knapsack, ratio, 0)

    if greedy_ub< ub:
        ub = greedy_ub

    while len(queue_bnb)!=0 and not found_optimum:
        current_node = queue_bnb.pop()
        node_ub = calculate_ub(current_node[1], current_node[0], weight_knapsack, ratio, current_node[2]-1)
        if node_ub >= ub:
            if current_node[1] >= ub:
                ub = current_node[1]
            if (current_node[2]+1) <= n:
                if current_node[0] + weights_subjects[current_node[2]] <= weight_knapsack:
                    queue_bnb.append([current_node[0] + weights_subjects[current_node[2]],
                                  current_node[1] + values_subjects[current_node[2]],
                                  current_node[2] + 1])
                queue_bnb.append([current_node[0],
                              current_node[1],
                              current_node[2] + 1])
    return ub, "Simple BnB"


def calculate_sorted_ub(current_profit, current_weight, weight_knapsack, ratio, current_item):
    if current_item < 0:
        ub = current_profit + (weight_knapsack-current_weight)*(ratio[0])
    else:
        ub = current_profit + (weight_knapsack - current_weight) * (ratio[current_item])

    return ub


def sorted_simple_bnb(data, weight_knapsack):
    greedy_ub, _ = greedy(data, weight_knapsack)
    data = data.drop([0])
    data["ratio"] = data["value"]/data["weight"]
    data = data.sort_values(by=["ratio"], ascending=False)  # todo: now quick sort, change to another sort ???

    weights_subjects = list(data["weight"])
    values_subjects = list(data["value"])
    ratio = list(data["ratio"])

    n = len(values_subjects)
    queue_bnb = list()
    found_optimum = False

    node = [0, 0, 0]  # weight, profit, level of item

    # todo: sorted profit ratio and use it
    # ratio = []
    # for i in range(len(weights_subjects)):
    #     ratio.append(values_subjects[i]/weights_subjects[i])
    # ratio.sort(reverse=True)
    queue_bnb.append(node)

    ub = calculate_sorted_ub(0, 0, weight_knapsack, ratio, 0)

    if greedy_ub< ub:
        ub = greedy_ub

    while len(queue_bnb)!=0 and not found_optimum:
        current_node = queue_bnb.pop()
        node_ub = calculate_sorted_ub(current_node[1], current_node[0], weight_knapsack, ratio, current_node[2]-1)
        if node_ub >= ub:
            if current_node[1] >= ub:
                ub = current_node[1]
            if (current_node[2] + 1) <= n:
                if current_node[0] + weights_subjects[current_node[2]] <= weight_knapsack:
                    queue_bnb.append([current_node[0] + weights_subjects[current_node[2]],
                                  current_node[1] + values_subjects[current_node[2]],
                                  current_node[2] + 1])
                queue_bnb.append([current_node[0],
                              current_node[1],
                              current_node[2] + 1])
    return ub, "Sorted Simple BnB"
