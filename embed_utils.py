import numpy as np
import open3d as o3d
import csv 
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement


#scaler for standardizing features
scaler = StandardScaler()
pca = PCA(n_components=3)
#scaledMatrix for scaling the points after PCA, it's a 3 by 3 matrix
scaledMatrix = [[0]*3 for i in range(3)]
m = 50 # number of slices 
n = 8 # number of segments on each slice
# table stores points coresponding to each slice segment, 2D list: row: slice; element: segment
segments = [ [0]*n for i in range(m)] 
#Table used for watermarking, each element in the table is the average distance of the corresponding segment to PC1
table_2D = [ [0]*n for i in range(m)] 
table_1D = []
valid_entry_index = []

def csv2pcd(csvPointPath, pcdPath):
    # Read CSV file
    data = np.genfromtxt( csvPointPath , delimiter=' ')

    # Extract coordinates from CSV data
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Create PLY file structure
    vertex = np.array(list(zip(x, y, z)), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_element = PlyElement.describe(vertex, 'vertex')
    ply_data = PlyData([vertex_element], text=True)

    # Save PLY file
    ply_data.write(pcdPath)


def pcd2csv(pcdPath, csvPointPath):
    #pcdPath = "./bunny/reconstruction/bun_zipper_res2.ply"
    pcd = o3d.io.read_point_cloud(pcdPath)

    #convert points in ply files into a csv file
    points = np.asarray(pcd.points)
    #o3d.visualization.draw_geometries([pcd])
    with open(csvPointPath, 'w') as f:
        mywriter = csv.writer(f, delimiter=',')
        mywriter.writerows(points)   

def PCAorReconstruction(output_dir = "./wm_outputs/", intermidiate_dir = "./embed_intermidiates/", reconst = False):
    if not reconst:
        csvPointPath = intermidiate_dir + "points.csv"
        # load dataset into Pandas DataFrame
        df = pd.read_csv(csvPointPath, names=['x','y','z'])
        features = ['x','y','z']
        # Separating out the features
        x = df.loc[:, features].values
        # Standardizing the features
        #scaler = StandardScaler()
        x = scaler.fit_transform(x)
        #perform PCA
        #pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['PC1', 'PC2', 'PC3'])
        principalDf.to_csv(intermidiate_dir + "PCApoints.csv", float_format='%.6f', index=False)
    if reconst:
        global scaledMatrix
        # unscale the points in wm_scaled.csv
        scaledMatrix = np.array(scaledMatrix)
        inv_scaledMatrix = np.linalg.inv(scaledMatrix)
        X = np.genfromtxt(intermidiate_dir + 'wm_scaled.csv', delimiter=',', skip_header=1)
        result = X.dot(inv_scaledMatrix)
        np.savetxt(intermidiate_dir + 'wm_unscaled.csv', result, fmt='%.6f', delimiter=',')
        wm_unscaled_PCA = pd.read_csv(intermidiate_dir + "wm_unscaled.csv", header=None)
        X_orig = np.dot(wm_unscaled_PCA, pca.components_)
        X_orig_backscaled = scaler.inverse_transform(X_orig)
        wm_original = pd.DataFrame(data = X_orig_backscaled)
        wm_original.to_csv(output_dir + "wm_points.csv", sep = ' ', float_format='%.6f', index=False, header = False)
        csv2pcd(output_dir + "wm_points.csv", output_dir + "wm_bunny.ply")


def scalePCAPoints(src_dir = "./embed_intermidiates/", target_dir = "./embed_intermidiates/"):
    #Find the logest projection along PC1
    PCAPath = src_dir + 'PCApoints.csv'
    df=pd.read_csv(PCAPath)
    p=df['PC1'].max()
    q=df['PC1'].min()
    LongestProj = p - q
    print(LongestProj)


    #scale the PCApoints
    global scaledMatrix
    X = np.genfromtxt(PCAPath, delimiter=',', skip_header=1)
    scaledMatrix = [[1/LongestProj, 0,            0],
                    [0 ,            1/LongestProj,0],
                    [0 ,            0,            1/LongestProj]]
    result = X.dot(scaledMatrix)
    np.savetxt(target_dir + '/scaled.csv', result, fmt='%.6f', delimiter=',')

def tableGenerator(src_dir = "./embed_intermidiates/", target_dir = "./embed_intermidiates/"):
    df=pd.read_csv(src_dir + 'scaled.csv', names=['PC1','PC2','PC3'])
    max=df['PC1'].max()
    min=df['PC1'].min()
    slices = []          # a list storing points coresponding to each slice
    for i in range(m):
        if i == m-1:
            df_slice = df[ (min + i/m <= df['PC1']) & (df['PC1']<= min + (i+1)/m)]
        else:
            df_slice = df[ (min + i/m <= df['PC1']) & (df['PC1']< min + (i+1)/m)]

        slices.append(df_slice)

    #Filling the table with the corresponding segments
    global segments
    for j in range(m): 
        for i in range(8):
            if i == 0:
                slice_sub = slices[j][(0 <= slices[j]['PC2']) & (0 <= slices[j]['PC3']) & (slices[j]['PC2'] <= slices[j]['PC3'])]
                segments[j][i] = slice_sub.copy()
            elif i == 1:
                slice_sub = slices[j][(0 <= slices[j]['PC2']) & (0 <= slices[j]['PC3']) & (slices[j]['PC2'] > slices[j]['PC3'])]
                segments[j][i] = slice_sub.copy()
            elif i ==2:
                slice_sub = slices[j][(0 <= slices[j]['PC2']) & (0 >= slices[j]['PC3']) & (slices[j]['PC2'] >= -slices[j]['PC3'])]
                segments[j][i] = slice_sub.copy()
            elif i ==3:
                slice_sub = slices[j][(0 <= slices[j]['PC2']) & (0 >= slices[j]['PC3']) & (slices[j]['PC2'] < -slices[j]['PC3'])]
                segments[j][i] = slice_sub.copy()
            elif i ==4:
                slice_sub = slices[j][(0 >= slices[j]['PC2']) & (0 >= slices[j]['PC3']) & (slices[j]['PC2'] >= slices[j]['PC3'])]
                segments[j][i] = slice_sub.copy()
            elif i ==5:
                slice_sub = slices[j][(0 >= slices[j]['PC2']) & (0 >= slices[j]['PC3']) & (slices[j]['PC2'] < slices[j]['PC3'])]
                segments[j][i] = slice_sub.copy()
            elif i ==6:
                slice_sub = slices[j][(0 >= slices[j]['PC2']) & (0 <= slices[j]['PC3']) & (-slices[j]['PC2'] >= slices[j]['PC3'])]
                segments[j][i] = slice_sub.copy()
            elif i ==7:
                slice_sub = slices[j][(0 >= slices[j]['PC2']) & (0 <= slices[j]['PC3']) & (-slices[j]['PC2'] < slices[j]['PC3'])]
                segments[j][i] = slice_sub.copy()

    #for each segment in segments, calculate the distance of each point to PC1 axis, 
    # and store the distance into a new column in each dataFrame
    for slice in segments:
        for seg in slice:
            seg.loc[:, 'dis'] = np.sqrt((seg['PC2']**2) + (seg['PC3']**2))

    #storing the average distance of each segment into table
    global table_2D
    for row in range(m):
        for col in range(n):
            table_2D[row][col] = segments[row][col].loc[:, 'dis'].mean()

    #convert 2d table into 1d
    global table_1D
    for row in range(m):
        for col in range(n):
            table_1D.append(table_2D[row][col])
    #record the valid indice in table_1D
    table_tobe_wm = []
    global valid_entry_index
    for i in range(m*n):
        if not math.isnan(table_1D[i]):
            valid_entry_index.append(i)
            table_tobe_wm.append(table_1D[i])

    with open("./3DTableWm/base1_dataset_tobe_wm/table_tobe_wm.csv", "w") as f:
        writer = csv.writer(f)
        header = [0] # Adding column index 
        writer.writerow(header)
        for ele in table_tobe_wm:
            writer.writerow([ele]) # multiply each entry by 100 to enable embedding 
    

# After embedding watermark (by running the sh file in 3DTableWm), 
# we need to change the segments to ensure the points in it reflect the embedding:
def subtract_lists(list1, list2):
    return [x - y for x, y in zip(list1, list2)]

def wm_sclaed():
    wm_table = np.genfromtxt("./3DTableWm/base1_experiments/testing/3Dmodel/embed/wm_datasets/table_tobe_wm_keyid1_wmnr.csv", delimiter=',', skip_header=1)
    #calculate the difference matrix between wm_table and original table
    wm_table_list = list(wm_table)
    # in order to subtract two lists, we need to maintian a list of original valid entries
    ori_table_list = []
    for inx in valid_entry_index:
        ori_table_list.append(table_1D[inx])
    diff = subtract_lists(wm_table_list, ori_table_list) 

    for i in range(len(diff)):
        ele_float = float(diff[i])
        diff[i] = "{:.8f}".format(ele_float)

    pt = 0
    for row in range(m):
        for col in range(n):
            #delete the 'dis' column to prepare the dataframe for tranformation
            del segments[row][col]['dis']
            if not segments[row][col].empty:
                t = float(diff[pt])
                if col == 0 or col == 1:
                    segments[row][col]['addi'] = [1] * len(segments[row][col])
                    T = (math.sqrt(2)/2) * t
                    Tmatrix = [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0], 
                            [0, T, T, 1]]
                    segments[row][col] = segments[row][col].dot(Tmatrix)
                    # drop the additional column used in translation
                    segments[row][col].drop(segments[row][col].columns[3], axis=1, inplace=True) 
                elif col == 2 or col == 3:
                    segments[row][col]['addi'] = [1] * len(segments[row][col])
                    T = (math.sqrt(2)/2) * t
                    Tmatrix = [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0], 
                            [0, T, -T, 1]]
                    segments[row][col] = segments[row][col].dot(Tmatrix)
                    # drop the additional column used in translation
                    segments[row][col].drop(segments[row][col].columns[3], axis=1, inplace=True) 
                elif col == 4 or col == 5:
                    segments[row][col]['addi'] = [1] * len(segments[row][col])
                    T = (math.sqrt(2)/2) * t
                    Tmatrix = [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0], 
                            [0, -T, -T, 1]]
                    segments[row][col] = segments[row][col].dot(Tmatrix)
                    # drop the additional column used in translation
                    segments[row][col].drop(segments[row][col].columns[3], axis=1, inplace=True) 
                elif col == 6 or col == 7:
                    segments[row][col]['addi'] = [1] * len(segments[row][col])
                    T = (math.sqrt(2)/2) * t
                    Tmatrix = [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0], 
                            [0, -T, T, 1]]
                    segments[row][col] = segments[row][col].dot(Tmatrix)
                    # drop the additional column used in translation
                    segments[row][col].drop(segments[row][col].columns[3], axis=1, inplace=True) 
                pt += 1
            else:
                continue

    # convert dataframes in segments into a csv file
    f = open('./embed_intermidiates/wm_scaled.csv', 'w')
    headerList = ['PC1', 'PC2', 'PC3']
    writer = csv.writer(f)
    writer.writerow(headerList)
    f.close()
    for row in range(m):
        for col in range(n):
            segments[row][col].to_csv('./embed_intermidiates/wm_scaled.csv', mode='a', header = False, index=False, float_format='%.6f')

