import open3d as o3d
import embed_utils as em_u
import extract_utils as ex_u
import csv 
import subprocess
def embed():
    pcdPath = "./bunny/reconstruction/bun_zipper_res2.ply"
    csvPath = "./embed_intermidiates/points.csv"
    valid_index_path = "./embed_intermidiates/valid_index.csv"
    em_u.pcd2csv(pcdPath, csvPath)
    em_u.PCAorReconstruction()
    em_u.scalePCAPoints()
    em_u.tableGenerator()
    with open(valid_index_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the list as a row in the CSV file
        writer.writerow(em_u.valid_entry_index)
    process = subprocess.run(['sh', 'base1_run_embed.sh'], cwd = "./3DTableWm", timeout = 10)
    em_u.wm_sclaed()
    em_u.PCAorReconstruction(reconst = True)
embed()


#visualizing the wm_model
# pcd = o3d.io.read_point_cloud("./wm_outputs/wm_bunny.ply")
# o3d.visualization.draw_geometries([pcd])
