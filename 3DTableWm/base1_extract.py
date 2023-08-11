import argparse
import os
import time
import fnmatch
import sys

from base1_core import base1_extract_watermark
from base1_utils import new_dir, get_keyidx_from_storedkey, get_params_from_susp_file, base1_record_extract_results


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", default="./base1_experiments", type=str, help="exp_root")
    parser.add_argument("--exp_tag", default='test', help="exp_tag")
    parser.add_argument("--target_dir", default='embed', help="the directory that saves the suspicious files")
    parser.add_argument("--dataset", default='covtype', choices=["winequality", "covtype", "3Dmodel"])
    parser.add_argument("--attack", default="watermarked", type=str, help="")
    parser.add_argument("--add_info", default="wm", type=str, help="")
    parser.add_argument("--susp_file", default="", type=str, help="the path to suspicious file")
    args = parser.parse_args()
    return args


def extract_pipeline(args):
    # collect all suspicious files
    csv_paths = []
    if args.susp_file != '':
        csv_paths.append(args.susp_file)
    elif args.target_dir != '':
        file_list = os.listdir(args.target_dir)
        file_list = sorted(file_list, key=lambda x: x.split("/")[-1])
        for filename in file_list:
            if fnmatch.fnmatch(filename, "*.csv"):
                csv_paths.append(os.path.join(args.target_dir, filename))
    else:
        print('Please specify target_dir or suspicious file.')
        exit()
    print(f'there are {len(csv_paths)} suspicious files to check.')
    print(csv_paths)

    # collect all candidate keys
    key_list = os.listdir(args.keys_folder)
    key_list = sorted(key_list, key=lambda x: x.split("/")[-1])
    candidate_keys_paths = []
    for keyname in key_list:
        if fnmatch.fnmatch(keyname, "*.csv"):
            candidate_keys_paths.append(os.path.join(args.keys_folder, keyname))
    print(f'there are {len(candidate_keys_paths)} candidate keys.')
    print(key_list)

    # extract watermark from each suspicious file using each key
    duration_collect = []
    for k in range(len(candidate_keys_paths)):  # for each key
        start_time = time.time()
        one_key = candidate_keys_paths[k]
        key_idx = get_keyidx_from_storedkey(one_key)

        for i in range(len(csv_paths)):  # for each susp_file
            one_file = csv_paths[i]
            one_file_name = os.path.splitext(os.path.basename(one_file))[0]
            susp_file_info = get_params_from_susp_file(one_file_name)

            #fig_name = 'ext_res_using_key' + key_idx + '_' + one_file_name + '.png'

            M_hat, result = base1_extract_watermark(susp_file=one_file, dataset=args.dataset, storedkey=one_key)
            print('==> extract result')
            print(
                f'secret key: {one_key} \nsuspicious dataset: {one_file_name} \nextract result (1: True; 0: False): {result}')
            base1_record_extract_results(os.path.join(args.res_folder, args.results_file), result, one_file, key_idx, M_hat,
                                   susp_file_info)
            end_time = time.time()
            duration_collect.append(end_time - start_time)

    mean_duration = sum(duration_collect) / len(duration_collect)
    print('Extracting watermark takes {:.2f}s in average for each suspicious file. (extract {} times)'.format(
        mean_duration, len(duration_collect)))  # 2.51s


if __name__ == "__main__":
    args = args_parse()
    if args.target_dir == 'embed':
        args.target_dir = os.path.join(args.exp_root, args.exp_tag, args.dataset, 'embed', 'wm_datasets')
    else:
        args.target_dir = os.path.join(args.exp_root, args.exp_tag, args.dataset, args.attack, args.add_info)

    args.keys_folder = os.path.join(args.exp_root, args.exp_tag, args.dataset, 'embed', 'storedkeys')
    args.res_folder = os.path.join(args.exp_root, args.exp_tag, args.dataset, 'extract', args.attack, args.add_info)
    new_dir(args.res_folder)
    args.figures_folder = os.path.join(args.res_folder, 'figures')
    new_dir(args.figures_folder)
    args.results_file = f"extract_res_{args.dataset}_{args.attack}_{args.add_info}.csv"
    extract_pipeline(args)
