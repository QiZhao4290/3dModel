import numpy as np
import csv
import sys

sys.path.append('..')
from base1_utils import random_generator, load_data, base1_load_storedkey


def base1_extract_watermark(susp_file, dataset, storedkey):
    """
    Given a susp_file and a storedkey, extract watermark from it, output result
    :param susp_file: the path to suspicious file
    :param dataset: 'covtype'
    :param stroedkey: key info, dict
    :return: extract result (the probability)
    """

    # 1. load suspicious dataset
    feat_numeric = load_data(susp_file, dataset)
    # 2. load stored key
    key_info = base1_load_storedkey(storedkey)
    # 3. extract the watermarked attribute from data
    attr = feat_numeric[:, key_info['attr_id']]
    # 4. regenerate the sequence using the stored seed K
    S = random_generator('S', attr.size, key_info['K'])
    # 5. compute M_hat
    XS = np.multiply(attr, S)
    M_hat = np.mean(XS)

    # 6. output result
    if M_hat > key_info['M'] / 2:
        result = 1
    else:
        result = 0


    '''
    # load original dataset to compute the noise
    clean_file = "$./base1_dataset_tobe_wm/base1_covtype_standardized_train.csv"
    org_data = load_data(clean_file, dataset)
    org_attr = org_data[:, key_info['attr_id']]
    SD = np.multiply(S, (attr-org_attr))
    mean_SD = np.mean(SD)
    '''

    return M_hat, result


if __name__ == '__main__':
    susp_file = '/home/yuanqixue/pycharmProjects/watermark_proj/database_fingerprint/baseline1/base1_experiments/test/covtype/embed/wm_datasets/base1_covtype_standardized_train_keyid1_wm.csv'
    dataset = 'covtype'
    storedkey = '/home/yuanqixue/pycharmProjects/watermark_proj/database_fingerprint/baseline1/base1_experiments/test/covtype/embed/storedkeys/base1_covtype_standardized_train_keyid1_keys.csv'
    result = base1_extract_watermark(susp_file, dataset, storedkey)
    print(result)
