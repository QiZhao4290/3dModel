import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_ours", type=str, default="", help="The csv file that saves the classification results")
    parser.add_argument("--res_sf", type=str, default="", help="The csv file that saves the classification results")
    parser.add_argument("--res_nr", type=str, default="", help="The csv file that saves the classification results")
    parser.add_argument("--output_dir", type=str, default="", help="The csv file that saves the extract results of baseline SF")
    parser.add_argument("--dataset", type=str, default='covtype', choices=["winequality", "covtype"])
    parser.add_argument("--attack", type=str, default='noise', choices=["noise", "delete", "alter", "insert", "none"])
    parser.add_argument("--add_info", type=str, default='rand')

    args = parser.parse_args()
    return args


def plot_classification_performance(csv_file, dataset, method, key_idx, attack, add_info, output_dir):
    # plot figures (for each watermarked dataset)
    # x-axis: attack level
    # y-axis: classification performance ('accuracy', 'precision', 'recall', 'f1 score')
    df = pd.read_csv(csv_file)

    # Filter part 1 (watermarked dataset, no attack)
    mask1 = (df['dataset'] == dataset) & (df['method'] == method) & \
            (df['attack'] == 'none') & (df['add_info'] == 'none') & \
            (df['alpha'] == 0) & (df['beta'] == 0) & (df['key_idx'] == str(key_idx))
    info_part1 = df[mask1]

    # Filter part 2
    mask2 = (df['dataset'] == dataset) & (df['method'] == method) & \
            (df['attack'] == attack) & (df['add_info'] == add_info) & \
            (df['key_idx'] == str(key_idx))
    info_part2 = df[mask2]

    # Concatenate the two parts
    df = pd.concat([info_part1, info_part2])
    df = df.drop_duplicates()
    df = df.sort_values(by='alpha', ascending=True).sort_values(by='beta', ascending=True)

    title_ = f"{dataset}_{attack}_{add_info}_keyid{key_idx}_{method}_classification"

    # Create the subplots
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(title_)

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Plot the metrics
    for ax, metric in zip([ax1, ax2, ax3, ax4],
                          ['accuracy', 'precision', 'recall', 'f1 score']):
        # Get the data for the metric
        train_metric = df[f"train_{metric}"]
        test_metric = df[f"test_{metric}"]
        alpha_values = df['alpha'].unique()
        beta_values = df['beta'].unique()

        n_alpha = len(alpha_values)
        n_beta = len(beta_values)

        if n_alpha == 1 or n_beta == 1:
            if n_alpha == 1:
                x_values = beta_values
                test_values = test_metric.values.reshape(n_beta, n_alpha)[:, 0]
                train_values = train_metric.values.reshape(n_beta, n_alpha)[:, 0]
            else:
                x_values = alpha_values
                test_values = test_metric.values.reshape(n_beta, n_alpha)[0, :]
                train_values = train_metric.values.reshape(n_beta, n_alpha)[0, :]

            sorted_indices = sorted(range(len(x_values)), key=lambda k: x_values[k])
            new_x_values = [x_values[i] for i in sorted_indices]
            new_test_values = [test_values[i] for i in sorted_indices]
            new_train_values = [train_values[i] for i in sorted_indices]


            ax.set_title(f"{metric.capitalize()} vs {['Beta', 'Alpha'][n_alpha == 1]}")
            ax.set_xlabel('Beta' if n_alpha == 1 else 'Alpha')
            ax.set_ylabel(metric)
            ax.plot(new_x_values, new_train_values, label='train', marker=".")
            ax.plot(new_x_values, new_test_values, label='test', marker="*")
            ax.set_xlim()
            ax.legend(loc='lower left')

        fig.savefig(os.path.join(output_dir, title_+".png"))
        plt.close(fig)


def extract_org_no_attack_res(df):
    mask = (df['watermarked'] == False) & (df['method'] == 'none')
    info = df[mask]
    return info


def extract_wm_no_attack_res(df, dataset, method):
    no_attack_mask = (df['dataset'] == dataset) & (df['method'] == method) & \
            (df['attack'] == 'none') & (df['add_info'] == 'none') & \
            (df['alpha'] == 0) & (df['beta'] == 0)
    info_part = df[no_attack_mask]
    return info_part


def extract_wm_with_attack_res(df, dataset, attack, add_info, method, attack_params=None):
    mask = (df['dataset'] == dataset) & (df['method'] == method) & \
            (df['attack'] == attack) & (df['add_info'] == add_info) & (df['watermarked'] == True)
    info = df[mask]

    if add_info == 'rand' and method == 'SF':
        info = info[info['alpha'].isin(attack_params)]

    return info


def extract_org_with_attack_res(df, dataset, method, attack, add_info, attack_params=None):
    '''
    print("df: ", df)
    print("dataset: ", dataset)
    print("method: ", method)
    print("attack: ", attack)
    print("add_info: ", add_info)
    print(df['dataset'],df['method'],df['watermarked'],df['attack'],df['add_info'])
    '''
    mask = (df['dataset'] == dataset) & (df['method'] == method) & \
           (df['watermarked'] == False) & (df['attack'] == attack) & (df['add_info'] == add_info)
    info = df[mask]

    if add_info == 'rand' and method == 'SF':
        info = info[info['alpha'].isin(attack_params)]
    return info


def sort_dataframe(df):
    df['alpha'] = df['alpha'].astype(float)
    df['beta'] = df['beta'].astype(float)
    df = df.sort_values(by='alpha', ascending=True).sort_values(by='beta', ascending=True)
    return df


def parse_info(info, metric):
    info = sort_dataframe(info)
    # print("info: ", info)
    test_metric = info[f"test_{metric}"]
    alpha_values = info['alpha'].unique()
    beta_values = info['beta'].unique()

    n_alpha = len(alpha_values)
    n_beta = len(beta_values)

    if n_alpha == 1:
        x_values = beta_values

        print("info: ", info)
        print("metric: ", metric)
        print("test_metric, n_beta, n_alpha: ", test_metric, n_beta, n_alpha)

        test_values = test_metric.values.reshape(n_beta, n_alpha)[:, 0]
    else:
        x_values = alpha_values

        print("info: ", info)
        print("metric: ", metric)
        print("test_metric, n_beta, n_alpha: ", test_metric, n_beta, n_alpha)

        test_values = test_metric.values.reshape(n_beta, n_alpha)[0, :]

    return x_values, test_values


def parse_cal_mean_std(info, metric, attack_params, attack, add_info):
    info = sort_dataframe(info)

    mean_list = []
    std_list = []
    for each_param in attack_params:
        if attack == "noise" and add_info == 'gaussian':
            filter_mask = (info['beta'] == each_param)
        else:
            filter_mask = (info['alpha'] == each_param)

        filter_info = info[filter_mask]

        test_metrics = filter_info[f"test_{metric}"]

        mean_val = test_metrics.mean()
        std_val = test_metrics.std()
        mean_list.append(mean_val)
        std_list.append(std_val)
    return mean_list, std_list


def base1_plot_classification_res(res_ours, res_sf, res_nr, dataset, attack, add_info, output_dir, attack_params):
    metric = 'accuracy'

    if res_ours != "":
        df_our = pd.read_csv(res_ours)
        org_with_res_our = extract_org_with_attack_res(df_our, dataset, 'Ours', attack, add_info)  # num_params
        wm_with_res_our = extract_wm_with_attack_res(df_our, dataset, attack, add_info, 'Ours')  # num_params * num_keys
        org_no_attack_res = extract_org_no_attack_res(df_our)  # 1
        wm_no_attack_our = extract_wm_no_attack_res(df_our, dataset, 'Ours')  # num_keys
        org_res_our = pd.concat([org_no_attack_res, org_with_res_our])
        wm_res_our = pd.concat([wm_no_attack_our, wm_with_res_our])
        x_values_org_our, metric_org_our = parse_info(org_res_our, metric)
        metric_wm_our_mean, metric_wm_our_std = parse_cal_mean_std(wm_res_our, metric, attack_params, attack, add_info)
    else:
        x_values_org_our = None
        metric_org_our =None
        metric_wm_our_mean = 0
        metric_wm_our_std = 0

    if res_sf != "":
        df_sf = pd.read_csv(res_sf)
        org_with_res_sf = extract_org_with_attack_res(df_sf, dataset, 'SF', attack, add_info,
                                                      attack_params)  # num_params
        wm_with_res_sf = extract_wm_with_attack_res(df_sf, dataset, attack, add_info, 'SF',
                                                    attack_params)  # num_params * num_keys
        wm_no_attack_sf = extract_wm_no_attack_res(df_sf, dataset, 'SF')  # num_keys
        org_res_sf = pd.concat([org_no_attack_res, org_with_res_sf])
        wm_res_sf = pd.concat([wm_no_attack_sf, wm_with_res_sf])
        x_values_org_sf, metric_org_sf = parse_info(org_res_sf, metric)
        metric_wm_sf_mean, metric_wm_sf_std = parse_cal_mean_std(wm_res_sf, metric, attack_params, attack, add_info)
    else:
        x_values_org_sf = None
        metric_wm_sf_mean = 0
        metric_wm_sf_std = 0

    if res_nr != "":
        df_nr = pd.read_csv(res_nr)
        org_with_res_nr = extract_org_with_attack_res(df_nr, dataset, 'NR', attack, add_info)  # num_params
        wm_with_res_nr = extract_wm_with_attack_res(df_nr, dataset, attack, add_info, 'NR')  # num_params * num_keys
        org_no_attack_res = extract_org_no_attack_res(df_nr)  # 1
        wm_no_attack_nr = extract_wm_no_attack_res(df_nr, dataset, 'NR')  # num_keys
        org_res_nr = pd.concat([org_no_attack_res, org_with_res_nr])
        wm_res_nr = pd.concat([wm_no_attack_nr, wm_with_res_nr])
        x_values_org_nr, metric_org_nr = parse_info(org_res_nr, metric)
        metric_wm_nr_mean, metric_wm_nr_std = parse_cal_mean_std(wm_res_nr, metric, attack_params, attack, add_info)
        '''
        print("df_nr: ", df_nr)
        print("org_with_res_nr: ", org_with_res_nr)
        print("wm_with_res_nr: ", wm_with_res_nr)
        print("org_no_attack_res: ", org_no_attack_res)
        print("wm_no_attack_nr: ", wm_no_attack_nr)
        print("org_res_nr: ", org_res_nr)
        print("wm_res_nr : ", wm_res_nr)
        print("x_values_org_nr: ", x_values_org_nr)
        print("metric_wm_nr_mean: ", metric_wm_nr_mean)
        '''

    else:
        x_values_org_nr = None
        metric_org_nr =None
        metric_wm_nr_mean = 0
        metric_wm_nr_std = 0


    # title = f"classification {metric} on {dataset} test set, {attack} attack, {add_info}"
    #
    # fig = plt.figure(figsize=(8, 5))
    # plt.title(title)
    # plt.xlabel('alpha')
    # plt.ylabel("classification "+metric)
    # plt.ylim([0.63, 0.73])
    #
    # if attack == "delete" or attack == "insert":
    #     plt.xticks(x_values_org_our, x_values_org_our)
    #
    # plt.plot(x_values_org_our, metric_org_our, marker="*", label='original dataset')
    # plt.errorbar(x_values_org_our, metric_wm_our_mean, yerr=metric_wm_our_std, linewidth=1, fmt='-o', capsize=4, label='watermarked dataset, Ours')
    # plt.errorbar(x_values_org_sf, metric_wm_sf_mean, yerr=metric_wm_sf_std, linewidth=1, fmt='-o', capsize=3, label='watermarked dataset, SF')
    # plt.legend(loc="lower left")
    # fig_name = f"classification_{metric}_{dataset}_{attack}_{add_info}.pdf"
    # fig.savefig(os.path.join(output_dir, fig_name))
    # plt.close(fig)

    return x_values_org_our, metric_org_our, metric_wm_our_mean, metric_wm_our_std, x_values_org_sf, metric_wm_sf_mean, metric_wm_sf_std, x_values_org_nr, metric_org_nr, metric_wm_nr_mean, metric_wm_nr_std


if __name__ == "__main__":
    args = args_parse()
    if not os.path.exists(args.output_dir):
        os.makedirs(Path(args.output_dir))

    key_idxes = ['1', '2', '3', '4', '5', '6', '7']
    if args.attack == 'noise':
        attack_params = [0, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    else:
        attack_params = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    base1_plot_classification_res(args.res_ours, args.res_sf, args.res_nr, args.dataset, args.attack, args.add_info, args.output_dir, attack_params)


