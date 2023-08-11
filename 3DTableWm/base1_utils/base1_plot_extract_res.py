import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
import csv


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root_ours", type=str, default="")
    parser.add_argument("--exp_root_sf", type=str, default="")
    parser.add_argument("--exp_root_nr", type=str, default="")
    parser.add_argument('--thre1_set', nargs='+', type=float, default=[0.99, 0.999, 0.9999], help="ours, threshold") # [0.99, 0.997, 0.9999]
    parser.add_argument('--thre2_set', nargs='+', type=float, default=[1.0, 15/16, 14/16], help="SF, threshold")
    parser.add_argument('--thre3_set', nargs='+', type=int, default=[1], help="NR, threshold")
    # parser.add_argument('--attack_params', nargs='+', type=float, default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--save_folder", type=str, default="", help="The path to save the output file and figures")
    parser.add_argument("--attack", type=str, default="insert")
    parser.add_argument("--add_info", type=str, default="concatenate")
    parser.add_argument("--withpk", type=str, default="withoutpk")
    parser.add_argument("--dataset", type=str, default='covtype', choices=["winequality", "covtype"])
    parser.add_argument("--sum_constant", type=int, default=56, help="fp+tp+tn+fn=constant, the constant is supposed to equal to k*(k+1), where k is the number of watermarked datasets")
    # 0.8375, 0.95, 0.99, 0.995, 0.9997
    # 0.8375, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9997, 0.9999
    args = parser.parse_args()
    return args


def cal_scores(method, res, t, alpha, sum_constant, add_info):
    if method == "Ours" or method == "NR":
        if add_info == 'gaussian':
            fp_mask = (res['extract_res'] >= t) & (res['key_idx'] != res['watermark_tag']) & (
                        res['attack_param2'] == alpha)
            fn_mask = (res['extract_res'] < t) & (res['key_idx'] == res['watermark_tag']) & (
                        res['attack_param2'] == alpha)
            tp_mask = (res['extract_res'] >= t) & (res['key_idx'] == res['watermark_tag']) & (
                        res['attack_param2'] == alpha)
            tn_mask = (res['extract_res'] < t) & (res['key_idx'] != res['watermark_tag']) & (
                        res['attack_param2'] == alpha)
        else:
            fp_mask = (res['extract_res'] >= t) & (res['key_idx'] != res['watermark_tag']) & (
                        res['attack_param1'] == alpha)
            fn_mask = (res['extract_res'] < t) & (res['key_idx'] == res['watermark_tag']) & (
                        res['attack_param1'] == alpha)
            tp_mask = (res['extract_res'] >= t) & (res['key_idx'] == res['watermark_tag']) & (
                        res['attack_param1'] == alpha)
            tn_mask = (res['extract_res'] < t) & (res['key_idx'] != res['watermark_tag']) & (
                        res['attack_param1'] == alpha)
    elif method == "SF":
        if add_info == 'gaussian':
            fp_mask = (res['similarity'] >= t) & (res['key_idx'] != res['watermark_tag']) & (
                        res['attack_param2'] == alpha)
            fn_mask = (res['similarity'] < t) & (res['key_idx'] == res['watermark_tag']) & (
                        res['attack_param2'] == alpha)
            tp_mask = (res['similarity'] >= t) & (res['key_idx'] == res['watermark_tag']) & (
                        res['attack_param2'] == alpha)
            tn_mask = (res['similarity'] < t) & (res['key_idx'] != res['watermark_tag']) & (
                        res['attack_param2'] == alpha)
        else:
            fp_mask = (res['similarity'] >= t) & (res['key_idx'] != res['watermark_tag']) & (
                        res['attack_param1'] == alpha)
            fn_mask = (res['similarity'] < t) & (res['key_idx'] == res['watermark_tag']) & (
                        res['attack_param1'] == alpha)
            tp_mask = (res['similarity'] >= t) & (res['key_idx'] == res['watermark_tag']) & (
                        res['attack_param1'] == alpha)
            tn_mask = (res['similarity'] < t) & (res['key_idx'] != res['watermark_tag']) & (
                        res['attack_param1'] == alpha)
    else:
        exit()

    fp = fp_mask.sum()
    fn = fn_mask.sum()
    tp = tp_mask.sum()
    tn = tn_mask.sum()

    # assert fp + fn + tp + tn == sum_constant, 'fp={},fn={},tp={},tn={}'.format(fp, fn, tp, tn)
    print('fp={},fn={},tp={},tn={}'.format(fp, fn, tp, tn))
    if tp == 0 and fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if alpha == 0:
        recall = 1
        precision = 1
        fpr = 0
        tpr = 1
    else:
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    scores = {
        "method": method,
        "threshold": t,
        "attack_param1": alpha,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
        "f1_score": f1_score
    }

    return scores


def record_metrics(res_file_1, thre1_set, res_file_2, thre2_set, res_file_3, thre3_set, save_folder, attack_params, record_file, attack_type, add_info, sum_constant):
    if res_file_1 != "":
        df_our = pd.read_csv(res_file_1)
    else:
        df_our = None
    if res_file_2 != "":
        df_sf = pd.read_csv(res_file_2)
    else:
        df_sf = None
    if res_file_3 != "":
        df_nr = pd.read_csv(res_file_3)
    else:
        df_nr = None

    print('df_nr', df_nr)

    print("save results at {}".format(os.path.join(save_folder, record_file)))
    with open(os.path.join(save_folder, record_file), mode='w', newline='') as file:
        fieldnames = ['method', 'attack', 'add_info', 'attack_param1', 'threshold', 'precision', 'recall', 'f1_score', 'tp', 'tn', 'fp', 'fn', 'fpr', 'tpr']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for alpha in attack_params:  # for each attack level
            if res_file_1 != "":
                for each_t in thre1_set:
                    scores_dict_ours = cal_scores(method="Ours", res=df_our, t=each_t, alpha=alpha, sum_constant=sum_constant, add_info=add_info)
                    print("Ours, attack_param={}, threshold={}, precision={:.3f}, recall={:.3f}, f1_score={:.3f}".
                      format(alpha, each_t, scores_dict_ours['precision'], scores_dict_ours['recall'], scores_dict_ours['f1_score']))
                    scores_dict_ours['attack'] = attack_type
                    scores_dict_ours['add_info'] = add_info
                    writer.writerow(scores_dict_ours)
            if res_file_2 != "":
                for each_tt in thre2_set:
                    scores_dict_sf_0 = cal_scores(method="SF", res=df_sf, t=each_tt, alpha=alpha, sum_constant=sum_constant, add_info=add_info)
                    print("SF, attack_param={}, threshold={}, precision={:.3f}, recall={:.3f}, f1_score={:.3f}".
                          format(alpha, each_tt, scores_dict_sf_0['precision'], scores_dict_sf_0['recall'], scores_dict_sf_0['f1_score']))
                    scores_dict_sf_0['attack'] = attack_type
                    scores_dict_sf_0['add_info'] = add_info
                    writer.writerow(scores_dict_sf_0)
            if res_file_3 != "":
                for each_ttt in thre3_set:
                    scores_dict_nr = cal_scores(method="NR", res=df_nr, t=each_ttt, alpha=alpha, sum_constant=sum_constant, add_info=add_info)
                    print("NR, attack_param={}, threshold={}, precision={:.3f}, recall={:.3f}, f1_score={:.3f}".
                          format(alpha, each_ttt, scores_dict_nr['precision'], scores_dict_nr['recall'], scores_dict_nr['f1_score']))
                    scores_dict_nr['attack'] = attack_type
                    scores_dict_nr['add_info'] = add_info
                    writer.writerow(scores_dict_nr)
            print("")


def plot_metrics(save_folder, record_file, record_file_no_attack, thre1_set, thre2_set, thre3_set, fig_name, attack, add_info):
    df_1 = pd.read_csv(os.path.join(save_folder, record_file))
    df_2 = pd.read_csv(os.path.join(save_folder, record_file_no_attack))
    df = pd.concat([df_2, df_1], ignore_index=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 5*3))
    ax_f1_score = axes[0]
    ax_precision = axes[1]
    ax_recall = axes[2]

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
        ax_f1_score.plot(nr_data['attack_param1'], nr_data['f1_score'], label='NR, thre={:.4f}'.format(ttt_idx), marker="*")
        ax_precision.plot(nr_data['attack_param1'], nr_data['precision'], label='NR, thre={:.4f}'.format(ttt_idx), marker="*")
        ax_recall.plot(nr_data['attack_param1'], nr_data['recall'], label='NR, thre={:.4f}'.format(ttt_idx), marker="*")

    # print(ours_data['attack_param1'])

    x_lim_v1 = [-0.1, 1.1]  # delete, insert
    x_lim_v2 = [-0.1, 1.5]  # noise

    if attack == 'noise':
        x_lim = x_lim_v2
    else:
        x_lim = x_lim_v1

    ax_f1_score.set_xlabel('alpha')
    ax_f1_score.set_xlim(x_lim)
    # ax_f1_score.set_xticks(ours_data['attack_param1'])
    ax_f1_score.set_ylabel('F1 Score')
    ax_f1_score.set_ylim([-0.1, 1.1])
    ax_f1_score.set_title(f'F1 Score vs. alpha')
    ax_f1_score.legend()

    ax_precision.set_xlabel('alpha')
    ax_precision.set_xlim(x_lim)
    ax_precision.set_ylabel('Precision')
    ax_precision.set_ylim([-0.1, 1.1])
    ax_precision.set_title(f'Precision vs. alpha')
    ax_precision.legend()

    ax_recall.set_xlabel('alpha')
    ax_recall.set_xlim(x_lim)
    ax_recall.set_ylabel('Recall')
    ax_recall.set_ylim([-0.1, 1.1])
    ax_recall.set_title(f'Recall vs. alpha')
    ax_recall.legend()
    # plt.show()

    fig.suptitle(f"watermark extract performance, {attack} attack ({add_info})")

    plt.savefig(os.path.join(save_folder, fig_name))
    print(f'save at {os.path.join(save_folder, fig_name)}')
    plt.close()


if __name__ == "__main__":
    args = args_parse()
    if not os.path.exists(args.save_folder):
        os.makedirs(Path(args.save_folder))

    if args.attack == 'noise':
        # attack_params = [50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1500.0, 2000.0, 5000.0, 10000.0]
        attack_params = [0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
    elif args.attack == 'watermarked':
        attack_params = [0]
        # args.sum_constant = 49
        args.sum_constant = 9  # k*k, where k is the number of watermarked datasets
    else:
        attack_params = [0.10, 0.20, 0.30, 0.40]

    extract_res_file = f"extract_res_{args.dataset}_{args.attack}_{args.add_info}.csv"
    if args.exp_root_ours != "":
        extract_res_ours = os.path.join(args.exp_root_ours, args.attack, args.add_info, extract_res_file)
    else:
        extract_res_ours = ""
    if args.exp_root_sf != "":
        extract_res_sf = os.path.join(args.exp_root_sf, args.attack, args.add_info, extract_res_file)
    else:
        extract_res_sf = ""
    '''using extract_res_nr in the base1_test_run.sh'''
    extract_res_nr = os.path.join(args.exp_root_nr, args.attack, args.add_info, extract_res_file)

    record_file = f"score_table_{args.attack}_{args.add_info}_{args.withpk}.csv"
    fig_name = f"extract_performance_{args.attack}_{args.add_info}_{args.withpk}.png"

    record_metrics(extract_res_ours, args.thre1_set, extract_res_sf, args.thre2_set, extract_res_nr, args.thre3_set,
                   args.save_folder, attack_params, record_file, args.attack, args.add_info, args.sum_constant)

    record_file_no_attack = f"score_table_watermarked_no_attack_{args.withpk}.csv"
    if not os.path.exists(os.path.join(args.save_folder, record_file_no_attack)):
        record_metrics(extract_res_ours, args.thre1_set, extract_res_sf, args.thre2_set, extract_res_nr, args.thre3_set,
                       args.save_folder, [0], record_file_no_attack, 'watermarked', 'no_attack', args.sum_constant)

    # plot_metrics(args.save_folder, record_file, record_file_no_attack, args.thre1_set, args.thre2_set, fig_name, args.attack, args.add_info)












