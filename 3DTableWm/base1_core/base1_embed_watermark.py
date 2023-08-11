import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import sys

sys.path.append('..')
from base1_utils import random_generator


# def base1_embed_watermark(data, attr_id, wm_params, storedkey_file):
def base1_embed_watermark(data, wm_params, storedkey_file, eps = 0.01):

    attr_id = wm_params['attr_id']
    attr = data[:, attr_id]

    S = random_generator('S', attr.size, wm_params['K'])
    diff = eps * S
    wm_attr = attr + diff
    '''fixed T (a fixed seed for T)'''
    '''
    T = random_generator('T', attr.size, wm_params['T'])
    absT = np.absolute(T)
    S_absT = np.multiply(S, absT)
    XS = np.multiply(attr, S)

    stats_dict = {"mean_attr": np.mean(attr), "var_attr": sample_var(attr),
                  "mean_S_absT": np.mean(S_absT), "var_S_absT": sample_var(S_absT), "mean_XS": np.mean(XS),
                  "mean_S": np.mean(S), "mean_absT": np.mean(absT)}

    # with a known security parameter M, calculate a, b, lam
    a, b, lam = fsolve(lambda x: func(x, stats_dict, wm_params), [0.00001, -0.00001, 0.00001])
    # a,b,lam are the true solutions
    assert (np.all(np.isclose(func([a, b, lam], stats_dict, wm_params), [0.0, 0.0, 0.0])))
    
    # watermarked attribute
    wm_attr = a * attr + b + lam * S_absT
'''
    # embed watermarked attribute to data
    wm_data = data.copy()
    wm_data[:, attr_id] = wm_attr

    # M_prime = a * stats_dict["mean_XS"] + b * stats_dict["mean_S"] + lam * stats_dict["mean_absT"]
    # # print("M_prime: ", M_prime)

    #attr_abs_diff = np.absolute(np.array(attr) - np.array(wm_attr))
    wm_params['attr_abs_diff'] = diff
    #wm_params['a'] = a
    #wm_params['b'] = b
    #wm_params['lam'] = lam
    wm_params['S'] = S
    #wm_params['Sequence T'] = T
    wm_params['attr'] = attr
    wm_params['wm_attr'] = wm_attr
    wm_params['attr_id'] = attr_id
    #wm_params['M_prime'] = M_prime
    df = pd.DataFrame.from_dict(wm_params, orient='index', columns=['value'])
    df.to_csv(storedkey_file, header=False)
    print(f'save secret key at {storedkey_file}')

    return wm_data


def sample_var(data):
    return np.var(data) * np.size(data) / (np.size(data) - 1)


def func(x, stats_dict, wm_params):
    a, b, lam = x
    eq1 = a * stats_dict["mean_attr"] + b + lam * stats_dict["mean_S_absT"] - stats_dict["mean_attr"]
    eq2 = pow(a, 2) * stats_dict["var_attr"] + pow(lam, 2) * stats_dict["var_S_absT"] - stats_dict["var_attr"]
    eq3 = a * stats_dict["mean_XS"] + b * stats_dict["mean_S"] + lam * stats_dict["mean_absT"] - wm_params['M']
    return [eq1, eq2, eq3]


'''
def base1_load_wm_params(json_file, key_idx):
    with open(json_file) as json_f:
        info = json.load(json_f)
    wm_params = info[key_idx]

    return wm_params


# def embed_watermark(data, wm_params) -> wm_data
if __name__ == "__main__":
    # attr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # data
    data = np.tile(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).transpose(), (1, 3))
    data = data.astype(np.float32)
    print(data)

    # wm_params selected by key_idx
    wm_params_json = '/home/yuanqixue/pycharmProjects/watermark_proj/database_fingerprint/baseline1/base1_params.json'
    key_idx = 1
    wm_params = base1_load_wm_params(wm_params_json, str(key_idx))
    # M = wm_params['M']

    # attr_id
    attr_id = 0

    #storedkey_file
    storedkey_file = '/home/yuanqixue/pycharmProjects/watermark_proj/database_fingerprint/baseline1/base1_experiments/test/covtype/embed/storedkeys/base1_covtype_standardized_train_keyid1_keys.csv'

    wm_data = base1_embed_watermark(data, attr_id, wm_params, storedkey_file)
    print(wm_data)
'''
