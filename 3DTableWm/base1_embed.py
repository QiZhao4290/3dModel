import os
import argparse
import time
import json
import sys
import numpy as np
from base1_core import base1_embed_watermark
from base1_utils import load_data, save_data, new_dir, base1_load_wm_params


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_file", default="./base1_covtype_standardized_train.csv", type=str,
                        help="path to clean file (not watermarked)")
    parser.add_argument("--wm_params_json", default="./base1_params.json", type=str,
                        help="json file to save watermark parameters")
    parser.add_argument("--exp_root", default="./base1_experiments", type=str, help="exp_root")
    parser.add_argument("--exp_tag", default='test', help="exp_tag")
    parser.add_argument("--dataset", default='covtype', choices=["cps", "covtype", "3Dmodel"])
    parser.add_argument("--key_idx", nargs='+', type=int, default=[1],
                        help='the index of key, len(args.key_idx) is the number of watermarked datasets')
    args = parser.parse_args()
    return args


def embed_pipeline(args):
    
    print(f'==> embed watermark, save result at {os.path.join(args.exp_root, args.exp_tag, args.dataset, "embed")}')
    feat_org = load_data(args.org_file, args.dataset)

    embed_path = os.path.join(args.exp_root, args.exp_tag, args.dataset, 'embed')
    wm_save_path = os.path.join(embed_path, 'wm_datasets')
    keys_save_path = os.path.join(embed_path, 'storedkeys')

    new_dir(wm_save_path)
    new_dir(keys_save_path)

    duration_collect = []
    # attr_ids = list(np.random.randint(low=0, high=feat_org.shape[1], size=len(args.key_idx)))
    for key_idx in args.key_idx:
        start_time = time.time()
        wm_params = base1_load_wm_params(args.wm_params_json, str(key_idx))
        storedfile = os.path.basename(args.org_file)[:-4] + '_keyid' + str(key_idx) + '_keys.csv'
        # data_wm = base1_embed_watermark(feat_org, attr_ids[key_idx-1], wm_params, os.path.join(keys_save_path, storedfile))
        data_wm = base1_embed_watermark(feat_org, wm_params, os.path.join(keys_save_path, storedfile), eps = 0.06)
        wm_file = os.path.basename(args.org_file)[:-4] + '_keyid' + str(key_idx) + '_wmnr.csv'
        save_data(args.org_file, args.dataset, data_wm, os.path.join(wm_save_path, wm_file))
        end_time = time.time()
        duration_collect.append(end_time - start_time)

    mean_duration = sum(duration_collect) / len(duration_collect)
    print('Embedding watermark into the dataset takes {:.2f}s in average.'.format(mean_duration))  # 2.51s

if __name__ == "__main__":
    args = args_parse()
    embed_pipeline(args)
