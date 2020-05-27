import random
import pandas as pd


def generate_uniform(number_items, boundaries_values, boundaries_weights):
    data = dict()
    data["value"] = [0] + [random.randint(boundaries_values[0], boundaries_values[1]) for i in range(number_items)]
    data["weight"] = [0] + [random.randint(boundaries_weights[0], boundaries_weights[1]) for i in range(number_items)]
    data = pd.DataFrame(data)
    return data
