import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from base1_plot_classification_res import base1_plot_classification_res


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_ours", type=str, default="", help="The csv file that saves the classification results")
    parser.add_argument("--res_sf", type=str, default="", help="The csv file that saves the classification results")
    parser.add_argument("--res_nr", type=str, default="", help="The csv file that saves the classification results")
    parser.add_argument('--thre1_set', nargs='+', type=float, default=[0.99, 0.999, 0.9999], help="ours, threshold")
    parser.add_argument('--thre2_set', nargs='+', type=float, default=[1.0, 15/16, 14/16], help="SF, threshold")
    parser.add_argument('--thre3_set', nargs='+', type=int, default=[1], help="NR, threshold")
    parser.add_argument("--record_folder", type=str, default="")
    parser.add_argument("--save_folder", type=str, default="", help="The path to save the output file and figures")
    parser.add_argument("--attack", type=str, default="insert")
    parser.add_argument("--add_info", type=str, default="concatenate")
    parser.add_argument("--withpk", type=str, default="withoutpk")
    parser.add_argument("--dataset", type=str, default='covtype', choices=["winequality", "covtype"])

    # 0.8375, 0.95, 0.99, 0.995, 0.9997
    args = parser.parse_args()
    return args


def plot_together(x_values_org_our, metric_org_our, metric_wm_our_mean, metric_wm_our_std, x_values_org_sf, metric_wm_sf_mean, metric_wm_sf_std,
                  x_values_org_nr, metric_org_nr, metric_wm_nr_mean, metric_wm_nr_std, save_folder, record_file, record_file_no_attack, thre1_set, thre2_set, thre3_set, fig_name, attack, add_info, output_dir):
    df_1 = pd.read_csv(os.path.join(save_folder, record_file))
    df_2 = pd.read_csv(os.path.join(save_folder, record_file_no_attack))
    df = pd.concat([df_2, df_1], ignore_index=True)

    fig, axes = plt.subplots(4, 1, figsize=(8, 5 * 3))
    ax_acc = axes[0]
    ax_f1_score = axes[1]
    ax_precision = axes[2]
    ax_recall = axes[3]

    # acc
    ax_acc.plot(x_values_org_nr, metric_org_nr, marker="*", label='original dataset')
    try:
        ax_acc.errorbar(x_values_org_nr, metric_wm_nr_mean, yerr=metric_wm_nr_std, linewidth=1, fmt='-o', capsize=4,
                        label='watermarked dataset, NR')
        if x_values_org_our is None:
            pass
        else:
            ax_acc.errorbar(x_values_org_our, metric_wm_our_mean, yerr=metric_wm_our_std, linewidth=1, fmt='-o', capsize=4,
                            label='watermarked dataset, Ours')
        if x_values_org_sf is None:
            pass
        else:
            ax_acc.errorbar(x_values_org_sf, metric_wm_sf_mean, yerr=metric_wm_sf_std, linewidth=1, fmt='-o', capsize=3,
                            label='watermarked dataset, SF')
    except:
        ax_acc.plot(x_values_org_nr, metric_wm_nr_mean, label='watermarked dataset, NR')
        if x_values_org_nr is None:
            pass
        else:
            ax_acc.plot(x_values_org_our, metric_wm_our_mean, label='watermarked dataset, Ours')
        if x_values_org_sf is None:
            pass
        else:
            ax_acc.plot(x_values_org_sf, metric_wm_sf_mean, label='watermarked dataset, SF')
    ax_acc.legend(loc="lower left")

    # other 3
    for t_idx in thre1_set:
        ours_data_mask = (df['method'] == "Ours") & (df["threshold"] == t_idx)
        ours_data = df[ours_data_mask]
        ax_f1_score.plot(ours_data['attack_param1'], ours_data['f1_score'], label='Ours, thre={:.4f}'.format(t_idx), marker="o")
        ax_precision.plot(ours_data['attack_param1'], ours_data['precision'], label='Ours, thre={:.4f}'.format(t_idx),
                          marker="o")
        ax_recall.plot(ours_data['attack_param1'], ours_data['recall'], label='Ours, thre={:.4f}'.format(t_idx), marker="o")
    for tt_idx in thre2_set:
        sf_data_mask = (df['method'] == "SF") & (df["threshold"] == tt_idx)
        sf_data = df[sf_data_mask]
        ax_f1_score.plot(sf_data['attack_param1'], sf_data['f1_score'], label='SF, thre={:.4f}'.format(tt_idx), marker=">")
        ax_precision.plot(sf_data['attack_param1'], sf_data['precision'], label='SF, thre={:.4f}'.format(tt_idx), marker=">")
        ax_recall.plot(sf_data['attack_param1'], sf_data['recall'], label='SF, thre={:.4f}'.format(tt_idx), marker=">")
    for ttt_idx in thre3_set:
        nr_data_mask = (df['method'] == "NR") & (df["threshold"] == ttt_idx)
        nr_data = df[nr_data_mask]
        print("f1 scores: ", nr_data['f1_score'])
        ax_f1_score.plot(nr_data['attack_param1'], nr_data['f1_score'], label='NR, thre={:.4f}'.format(ttt_idx), marker="*")
        ax_precision.plot(nr_data['attack_param1'], nr_data['precision'], label='NR, thre={:.4f}'.format(ttt_idx), marker="*")
        ax_recall.plot(nr_data['attack_param1'], nr_data['recall'], label='NR, thre={:.4f}'.format(ttt_idx), marker="*")

    x_lim_v1 = [-0.1, 1]  # delete, insert
    x_lim_v2 = [-0.1, 1.5]  # noise

    if attack == 'noise':
        x_lim = x_lim_v2
    else:
        x_lim = x_lim_v1

    ax_acc.set_title(f"Classification Accuracy, {attack} attack ({add_info})", fontsize=10)
    ax_acc.set_xlim(x_lim)
    ax_acc.set_xlabel("alpha")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim([0.63, 0.73])

    ax_f1_score.set_xlabel('alpha')
    ax_f1_score.set_xlim(x_lim)
    # ax_f1_score.set_xticks()
    ax_f1_score.set_ylabel('F1 Score')
    ax_f1_score.set_ylim([-0.1, 1.1])
    ax_f1_score.set_title(f'Watermark extraction, F1 Score vs. alpha', fontsize=10)
    ax_f1_score.legend()

    ax_precision.set_xlabel('alpha')
    ax_precision.set_xlim(x_lim)
    ax_precision.set_ylabel('Precision')
    ax_precision.set_ylim([-0.1, 1.1])
    ax_precision.set_title(f'Watermark extraction, Precision vs. alpha', fontsize=10)
    ax_precision.legend()

    ax_recall.set_xlabel('alpha')
    ax_recall.set_xlim(x_lim)
    ax_recall.set_ylabel('Recall')
    ax_recall.set_ylim([-0.1, 1.1])
    ax_recall.set_title(f'Watermark extraction, Recall vs. alpha', fontsize=10)
    ax_recall.legend()

    fig.tight_layout()

    # fig.suptitle(f"{attack} attack ({add_info})")
    print(f'save at {os.path.join(output_dir, fig_name)}')
    plt.savefig(os.path.join(output_dir, fig_name))
    plt.close()




if __name__ == "__main__":
    args = args_parse()
    record_file = f"score_table_{args.attack}_{args.add_info}_{args.withpk}.csv"
    record_file_no_attack = f"score_table_watermarked_no_attack_{args.withpk}.csv"

    fig_name = f"overall_performance_{args.attack}_{args.add_info}_{args.withpk}.pdf"

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # key_idxes = ['1', '2', '3', '4', '5', '6', '7']
    key_idxes = ['1', '2', '3']

    if args.attack == 'noise':
        attack_params = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    else:
        attack_params = [0, 0.1, 0.2, 0.3, 0.4]

    x_values_org_our, metric_org_our, metric_wm_our_mean, metric_wm_our_std, x_values_org_sf, metric_wm_sf_mean, metric_wm_sf_std, x_values_org_nr, metric_org_nr, metric_wm_nr_mean, metric_wm_nr_std= \
        base1_plot_classification_res(args.res_ours, args.res_sf, args.res_nr, args.dataset, args.attack, args.add_info, args.save_folder, attack_params)


    print('x_values_org_nr', x_values_org_nr)
    print('metric_org_nr', metric_org_nr)
    print('metric_wm_nr_mean', metric_wm_nr_mean)
    print('metric_wm_nr_std', metric_wm_nr_std)

    metric_wm_our_std = np.nan_to_num(metric_wm_our_std, nan=0)
    print('22222metric_wm_our_std', metric_wm_our_std)
    plot_together(x_values_org_our, metric_org_our, metric_wm_our_mean, metric_wm_our_std, x_values_org_sf, metric_wm_sf_mean, metric_wm_sf_std, x_values_org_nr, metric_org_nr, metric_wm_nr_mean, metric_wm_nr_std,
                  args.record_folder, record_file, record_file_no_attack, args.thre1_set, args.thre2_set, args.thre3_set, fig_name, args.attack, args.add_info, args.save_folder)