import pandas as pd
import csv
import numpy as np


def load_data(csv_file, dataset):
    data = pd.read_csv(csv_file)
    header = list(data.columns)
    X = data.values
    # print(f"==> Load dataset {dataset}, n={X.shape[0]}, d={X.shape[1]}")

    if dataset == "covtype":
        cols_numeric_attr = list(range(1, 11))
        cols_binary_attr = list(range(11, 55))
        col_label = 55
        #binary_attributes = X[:, cols_binary_attr]
    elif dataset == "winequality":
        cols_numeric_attr = list(range(1, 13))
        col_label = 12
    elif dataset == "3Dmodel":
        cols_numeric_attr = list(range(1))
    else:
        exit()

    index_column = X[:, 0]
    feat_matrix = X[:, cols_numeric_attr]
    #label_org = X[:, col_label]

    return feat_matrix


def save_data(org_file, dataset, feat_wm, saved_file):
    org_data = pd.read_csv(org_file)
    df = org_data.copy(deep=True)
    if dataset == "covtype":
        df.iloc[:, 1:11] = feat_wm
    elif dataset == "winequality":
        df.iloc[:, 1:12] = feat_wm
    elif dataset == "3Dmodel":
        df.iloc[:, :] = feat_wm
    else:
        exit()
    df.to_csv(saved_file, index=False)
    print(f'save watermarked dataset at {saved_file}')


def base1_load_storedkey(storedkey_file):
    info_dict = {}

    with open(storedkey_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = row[0]
            val = row[1]

            # Convert the value to a number
            if key in ['M', 'K']:
                val = float(val)
            elif key == 'attr_id':
                val = int(float(val))

            info_dict[key] = val

    return info_dict


