# -*- coding: utf-8 -*-
"""
@author: gkredila
"""

"""
-------------------------------------------------------------------------------
-- Import Packages --
-------------------------------------------------------------------------------
"""
import pandas as pd # data manipulation
import numpy as np
import matplotlib.pyplot as plt # clustering plot
from sklearn.cluster import KMeans # clustering algo
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.spatial import ConvexHull # create cluster bounds
import folium # map viz
from datashader.utils import lnglat_to_meters as webm # convert coords to Mercator
import seaborn as sns
sns.set(style='whitegrid')
import warnings
warnings.filterwarnings("ignore")

import postgresql_connection
# if the console cannot find this file, please execute the file once
from postgresql_connection import db_connect


"""
How do we maximize the impact of marketing campaigns focused on targeted customers?

Customer Segmentation via Recency, Frequency, Monetary (RFM) Analysis
-- Create higher rates of customer response, expanding loyalty and customer lifetime
"""

"""
-------------------------------------------------------------------------------
-- Load Data --
-------------------------------------------------------------------------------
"""
connection = db_connect()
rfm = pd.read_sql_table('rfm_top3_prodcat', connection)


# Geospatial Work

rfm['geoloc_lat'] = pd.to_numeric(rfm['geoloc_lat'])
rfm['geoloc_long'] = pd.to_numeric(rfm['geoloc_long'])


"""
-------------------------------------------------------------------------------
-- Data Preprocessing --
1. Make sure Kmeans input data is numeric
2. Remove noise/outliers
3. Ensure Normal distribution
4. Ensure variables on same scale (normalized or standardized to have same mean/variance)
5. Remove collinearity
6. Reduce dimensions
-------------------------------------------------------------------------------
"""



"""
-------------------------------------------------------------------------------
-- EDA --
-------------------------------------------------------------------------------
"""
# Remove coordinates not within the bounds of Brazil's borders
# Brazil North = 5 deg 16′ 27.8″ N latitude.;
rfm = rfm[rfm.geoloc_lat <= 5.269582]
# Brazil West = 73 deg, 58′ 58.19″W Long.
rfm = rfm[rfm.geoloc_long >= -73.98306]
# Brazil South = 33 deg, 45′ 04.21″ S Latitude.
rfm = rfm[rfm.geoloc_lat >= -33.750936]
# Brazil East =  34 deg, 47′ 35.33″ W Long.
rfm = rfm[rfm.geoloc_long <=  -34.793015]

# transform coordinates to Mercator x/y Coords

#------------------------------------------------------------------------------

rfm_only = pd.DataFrame(data = rfm, columns = ['cust_id','frequency','recency','monetary'])
rfm_only = rfm_only[['cust_id','frequency','recency','monetary']].drop_duplicates()

df = pd.DataFrame(data = rfm_only, columns = ['frequency','recency','monetary'])
sns.boxplot(x='variable', y="value", data = pd.melt(df))
plt.show()

#need to standardize due to outliers
columns = ['frequency','recency','monetary']
rfm_std = rfm_only.copy()
features = rfm_std[columns]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
rfm_std[columns] = features

rfm_df_std = pd.DataFrame(data = rfm_std, columns = ['frequency','recency','monetary'])
sns.boxplot(x='variable', y="value", data = pd.melt(rfm_df_std))
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 808)
    kmeans.fit(rfm_df_std)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Plot indicates either cluster of 2 or 3

range_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_cluster in range_clusters:
    kmeans = KMeans(n_clusters = num_cluster, max_iter = 50)
    kmeans.fit(rfm_df_std)
    
    cluster_labs = kmeans.labels_
    
    silh_avg = silhouette_score(rfm_df_std, cluster_labs)
    print("For n_cluster = {0}, the silhouette score is {1}".format(num_cluster, silh_avg))
"""
For n_cluster = 2, the silhouette score is 0.4676789839579955
For n_cluster = 3, the silhouette score is 0.5005401196282795
For n_cluster = 4, the silhouette score is 0.5124271121076909
For n_cluster = 5, the silhouette score is 0.438287736894351
For n_cluster = 6, the silhouette score is 0.4573107617226523
For n_cluster = 7, the silhouette score is 0.45779552888285957
For n_cluster = 8, the silhouette score is 0.4140360928254224
"""
# 3 clusters

kmeans = KMeans(n_clusters = 4, max_iter = 50, random_state = 808)
kmeans.fit(rfm_df_std)
rfm_df_std.loc[:,'cust_id'] = rfm_only['cust_id']
rfm_df_std['cluster'] = kmeans.labels_

sns.boxplot(x='cluster', y = 'monetary', data = rfm_df_std)
sns.boxplot(x='cluster', y = 'frequency', data = rfm_df_std)
sns.boxplot(x='cluster', y = 'recency', data = rfm_df_std)

sns.scatterplot(rfm_df_std[:,0], rfm_df_std[:,1], hue='cluster', palette=['red', 'blue', 'purple', 'green'], alpha=0.5, s=7)


N = 1000
X0 = np.random.normal(np.repeat(np.random.uniform(0, 20, 4), N), 1)
X1 = np.random.normal(np.repeat(np.random.uniform(0, 10, 4), N), 1)
X = np.vstack([X0, X1]).T
y = np.repeat(range(4), N)
colors = ['red', 'blue', 'purple', 'green']
ax = sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette=colors, alpha=0.5, s=7)

means = np.vstack([X[y == i].mean(axis=0) for i in range(4)])
ax = sns.scatterplot(means[:, 0], means[:, 1], hue=range(4), palette=colors, s=20, ec='black', legend=False, ax=ax)
plt.show()


# Geospatial Work

rfm['geoloc_lat'] = pd.to_numeric(rfm['geoloc_lat'])
rfm['geoloc_long'] = pd.to_numeric(rfm['geoloc_long'])

# Remove coordinates not within the bounds of Brazil's borders
# Brazil North = 5 deg 16′ 27.8″ N latitude.;
rfm = rfm[rfm.geoloc_lat <= 5.269582]
# Brazil West = 73 deg, 58′ 58.19″W Long.
rfm = rfm[rfm.geoloc_long >= -73.98306]
# Brazil South = 33 deg, 45′ 04.21″ S Latitude.
rfm = rfm[rfm.geoloc_lat >= -33.750936]
# Brazil East =  34 deg, 47′ 35.33″ W Long.
rfm = rfm[rfm.geoloc_long <=  -34.793015]

# transform coordinates to Mercator x/y Coords
