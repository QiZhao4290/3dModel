import json

def base1_load_wm_params(json_file, key_idx):
    with open(json_file) as json_f:
        info = json.load(json_f)
    wm_params = info[key_idx]

    return wm_params