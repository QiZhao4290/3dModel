import numpy as np
import open3d as o3d
import csv 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load dataset into Pandas DataFrame
df = pd.read_csv("points.csv", names=['x','y','z'])
features = ['x','y','z']
# Separating out the features
x = df.loc[:, features].values
# print("                    Original:", x[0])
# Standardizing the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

#perform PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2', 'PC3'])

principalDf.to_csv("PCApoints.csv", float_format='%.6f', index=False)




# print("                   PCA space:", principalComponents[0])
# print("Original from PCA backscaled:", X_orig_backscaled[0])