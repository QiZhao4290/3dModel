import open3d as o3d
import extract_utils as ex_u
import csv
import subprocess
import numpy as np
import pandas as pd

# Specify the file path
file_path = './3DTableWm/base1_experiments/testing/3Dmodel/embed/wm_datasets/susp_table.csv'

# Read the single column CSV file as a NumPy array, discarding NaN values
susp_table = np.genfromtxt(file_path, delimiter='\n', skip_header=1, usecols=(0), missing_values='nan', filling_values=np.nan)

#record the valid index
valid_indices_susp = np.where(~np.isnan(susp_table))[0]

# Remove NaN values from the array
susp_data = susp_table[~np.isnan(susp_table)]

# Specify the file path
file_path = './3DTableWm/base1_experiments/testing/3Dmodel/embed/wm_datasets/table_tobe_wm_keyid1_wmnr.csv'

# Read the single column CSV file as a NumPy array
ori_wm_data = np.genfromtxt(file_path, delimiter='\n', skip_header=1, usecols=(0))
#select the entries corresponding to the valid index in susp_table
selected_ori_wm_data = ori_wm_data[valid_indices_susp]
#the difference between t2 and t1
table_diff = susp_data - selected_ori_wm_data

# Create a pandas DataFrame from the array
df = pd.DataFrame({'difference': table_diff})

# Specify the file path for saving the Excel file
file_path = 'table_diff.xlsx'

# Export the DataFrame to an Excel file with one column
df.to_excel(file_path, index=False)

