import os
import csv


def base1_record_extract_results(csv_path, extract_res, susp_file, key_idx, M_hat, susp_file_info):
    file_exists = os.path.isfile(csv_path)

    if not file_exists:
        header = ['extract_res', 'watermark_tag', 'key_idx', 'M_hat', 'susp_file', 'attack_param1', 'attack_param2']
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)

    row = [extract_res, susp_file_info['wm_tag'], key_idx, M_hat, susp_file, susp_file_info['attack_param1'], susp_file_info['attack_param2']]

    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(row)