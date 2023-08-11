import numpy as np
import open3d as o3d
import csv 
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
valid_entry_index_susp = []

def pcd2csv(pcdPath, csvPointPath):
    #pcdPath = "./bunny/reconstruction/bun_zipper_res2.ply"
    pcd = o3d.io.read_point_cloud(pcdPath)

    #convert points in ply files into a csv file
    points = np.asarray(pcd.points)
    #o3d.visualization.draw_geometries([pcd])
    with open(csvPointPath, 'w') as f:
        mywriter = csv.writer(f, delimiter=',')
        mywriter.writerows(points)   

def PCA(intermidiate_dir = "./extract_intermidiates/"):

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
    

def scalePCAPoints(src_dir = "./extract_intermidiates/", target_dir = "./extract_intermidiates/"):
    #Find the logest projection along PC1
    PCAPath = src_dir + 'PCApoints.csv'
    df=pd.read_csv(PCAPath)
    p=df['PC1'].max()
    q=df['PC1'].min()
    LongestProj = p - q


    #scale the PCApoints
    global scaledMatrix
    X = np.genfromtxt(PCAPath, delimiter=',', skip_header=1)
    scaledMatrix = [[1/LongestProj, 0,            0],
                    [0 ,            1/LongestProj,0],
                    [0 ,            0,            1/LongestProj]]
    result = X.dot(scaledMatrix)
    np.savetxt(target_dir + '/scaled.csv', result, fmt='%.6f', delimiter=',')

def tableGenerator(validInx, src_dir = "./extract_intermidiates/"):
    df=pd.read_csv(src_dir + 'scaled.csv', names=['PC1','PC2','PC3'])
    max=df['PC1'].max()
    min=df['PC1'].min()
    slices = []          # a list storing points coresponding to each slice
    for i in range(m):
        if i == m-1:
            df_slice = df[ (min + i/m <= df['PC1']) & (df['PC1']<= min + (i+1)/m)]
        else:
            df_slice = df[ (min + i/m <= df['PC1']) & (df['PC1']< min + (i+1)/m)]
        #print(df_slice)
        slices.append(df_slice)
    #print(slices)
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
    #print(segments)    
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
    #extract from table_1D entries corresponding to previouse valid indice
    susp_table = []
    for inx in validInx:
        susp_table.append(table_1D[inx])
    
    #record valid indice in susp_table
    global valid_entry_index_susp
    for i in range(len(susp_table)):
        if not math.isnan(susp_table[i]):
            valid_entry_index_susp.append(i)

    with open("./3DTableWm/base1_experiments/testing/3Dmodel/embed/wm_datasets/susp_table.csv", "w") as f:
        writer = csv.writer(f)
        header = [0] # Adding column index 
        writer.writerow(header)
        for ele in susp_table:
            writer.writerow([ele])