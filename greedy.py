def greedy(data, weight_knapsack):

    # value = [(x[0], x[2] / x[1]) for x in zip(np.arange(self.n),
    #                                           self.weights,
    #                                           self.profits)]

    value = list()
    weights_subjects = list(data["weight"][1:])
    values_subjects = list(data["value"][1:])

    for i in range(len(weights_subjects)):
        value.append((i, values_subjects[i]/weights_subjects[i]))

    value = sorted(value, key=lambda x: x[1], reverse=True)

    cur_weight = 0
    result_value = 0
    for v, _ in enumerate(value):
        if cur_weight + weights_subjects[value[v][0]] <= weight_knapsack:
            cur_weight += weights_subjects[value[v][0]]
            result_value += values_subjects[value[v][0]]
        else:
            continue
    return result_value, "Greedy"