import open3d as o3d
import extract_utils as ex_u
import csv
import subprocess
def extract():
    sus_Path = "./wm_outputs/wm_bunny.ply"
    csvPath = "./extract_intermidiates/points.csv"
    valid_index_path = "./embed_intermidiates/valid_index.csv"
    valid_index_susp_path = "./extract_intermidiates/valid_index_susp.csv"
    # Open the CSV file in read mode
    with open(valid_index_path, mode='r') as file:
        reader = csv.reader(file)
        # Read the CSV file and convert it into a list
        validInx = [int(value) for row in reader for value in row]
    ex_u.pcd2csv(sus_Path, csvPath)
    ex_u.PCA()
    ex_u.scalePCAPoints()
    ex_u.tableGenerator(validInx)
    with open(valid_index_susp_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the list as a row in the CSV file
        writer.writerow(ex_u.valid_entry_index_susp)
    process = subprocess.run(['sh', 'base1_run_extract.sh'], cwd = "./3DTableWm", timeout = 10)

extract()