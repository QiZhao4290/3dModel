import random
import numpy as np


def random_generator(type, length, seed=None):

    if type == 'T' and seed is not None:
        rng = np.random.default_rng(seed=seed)
        mu, sigma = 0, 1  # mean and standard deviation
        generated_arr = rng.normal(mu, sigma, length)
    elif type == 'T':
        mu, sigma = 0, 1  # mean and standard deviation
        generated_arr = np.random.normal(mu, sigma, length)
    elif type == 'S' and seed is not None:
        s_val = [-1, 1]
        generated_arr = np.array([])
        random.seed(seed)
        for i in range(0, length):
            generated_val = s_val[(random.randint(0, 1000) % 2)]
            generated_arr = np.hstack((generated_arr, [generated_val])) \
                if generated_arr.size else np.array([generated_val])
    else:
        pass
    return generated_arr
