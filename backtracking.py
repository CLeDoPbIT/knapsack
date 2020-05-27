from greedy import greedy


def backtracking(data, weight_knapsack):
    result = {0:0}
    items = {0:[0]}
    data = data.drop([0])

    weights_subjects = list(data["weight"])
    values_subjects = list(data["value"])

    for i in range(0, len(weights_subjects)):
        current_weights = list(result.keys())
        i_cw = 0
        tmp_result = result.copy()
        while i_cw < len(current_weights):
            if current_weights[i_cw] + weights_subjects[i] <= weight_knapsack:
                if current_weights[i_cw]+weights_subjects[i] in tmp_result:
                    result[current_weights[i_cw] + weights_subjects[i]] = max(tmp_result[current_weights[i_cw] + weights_subjects[i]], tmp_result[current_weights[i_cw]] + values_subjects[i])
                    if result[current_weights[i_cw] + weights_subjects[i]] <= result[current_weights[i_cw]] + values_subjects[i]:
                        items[current_weights[i_cw] + weights_subjects[i]] = items[current_weights[i_cw]] + [i]
                else:
                    result[current_weights[i_cw]+weights_subjects[i]] = tmp_result[current_weights[i_cw]] + values_subjects[i]
                    items[current_weights[i_cw]+weights_subjects[i]] = items[current_weights[i_cw]] + [i]
            i_cw += 1
    answer = max(list(result.values()))
    return answer, "backtracking "


def calculate_sorted_ub(current_profit, current_weight, weight_knapsack, ratio, current_item):
    if current_item < 0:
        ub = current_profit + (weight_knapsack-current_weight)*(ratio[0])  # todo: problem if weight_knapsack >> current_weight then ub will be huge. Maybe can be fixed if use min(weight_knapsack-current_weight, remaining weight of items)
    else:
        ub = current_profit + (weight_knapsack - current_weight) * (ratio[current_item])

    return ub


def backtracking_bnb(data, weight_knapsack):

    greedy_ub, _ = greedy(data, weight_knapsack)
    data = data.drop([0])
    data["ratio"] = data["value"]/data["weight"]
    data = data.sort_values(by=["ratio"], ascending=False)

    result = {0:0}
    items = {0:[0]}

    weights_subjects = list(data["weight"])
    values_subjects = list(data["value"])
    ratio = list(data["ratio"])

    ub = calculate_sorted_ub(0, 0, weight_knapsack, ratio, 0)

    if greedy_ub< ub:
        ub = greedy_ub

    for i in range(len(weights_subjects)):
        current_weights = list(result.keys())
        i_cw = 0
        tmp_result = result.copy()
        while i_cw < len(current_weights):
            node_ub = calculate_sorted_ub(tmp_result[current_weights[i_cw]], current_weights[i_cw], weight_knapsack, ratio, i)

            if node_ub >= ub:
                if tmp_result[current_weights[i_cw]] >= ub:
                    ub = tmp_result[current_weights[i_cw]]

                if current_weights[i_cw] + weights_subjects[i] <= weight_knapsack:
                    if current_weights[i_cw]+weights_subjects[i] in tmp_result:
                        result[current_weights[i_cw] + weights_subjects[i]] = max(tmp_result[current_weights[i_cw] + weights_subjects[i]], tmp_result[current_weights[i_cw]] + values_subjects[i])
                        if result[current_weights[i_cw] + weights_subjects[i]] <= result[current_weights[i_cw]] + values_subjects[i]:
                            items[current_weights[i_cw] + weights_subjects[i]] = items[current_weights[i_cw]] + [i]
                    else:
                        result[current_weights[i_cw]+weights_subjects[i]] = tmp_result[current_weights[i_cw]] + values_subjects[i]
                        items[current_weights[i_cw]+weights_subjects[i]] = items[current_weights[i_cw]] + [i]
            i_cw += 1
    answer = max(list(result.values()))
    return answer, "backtracking bnb"
