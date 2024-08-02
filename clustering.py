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
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from sklearn.cluster import KMeans # clustering algo
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from scipy.spatial import ConvexHull # create cluster bounds
from yellowbrick.cluster import silhouette_visualizer
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

R — Higher score = customer has recently made a purchase.
Most likely respond to current promotion. 
Low R score = the possibility of being churned

F — Higher score = customer has made repeated purchasing at higher frequency. 
(High demand / Loyalty)

M — Higher score = purchasing at larger amount. (High value customer)
"""

"""
-------------------------------------------------------------------------------
-- Load Data --
-------------------------------------------------------------------------------
"""
connection = db_connect()
rfm = pd.read_sql_table('rfm', connection)
rfm.info()
rfm.shape # (96469, 7)

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 96469 entries, 0 to 96468
Data columns (total 7 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   cust_id             96469 non-null  object 
 1   cust_zip_cd_prefix  96469 non-null  object 
 2   cust_city           96469 non-null  object 
 3   cust_state          96469 non-null  object 
 4   frequency           96469 non-null  int64  
 5   recency             96469 non-null  int64  
 6   monetary            96469 non-null  float64
dtypes: float64(1), int64(2), object(4)
memory usage: 5.2+ MB
"""
rfm_only = rfm[['recency', 'frequency', 'monetary']].copy()


"""
-------------------------------------------------------------------------------
-- Data Preprocessing --
1. Check for nulls
2. Check for negative values
3. Make sure Kmeans input data is numeric
4. Remove noise/outliers
5. Remove collinearity (if any)
-------------------------------------------------------------------------------
"""
null_check = 0
for col in list(rfm.columns):
    if any(rfm[col].isnull()):
        null_check += 1

print(f'There are {null_check} NULL values.')

# For negatives check, ignore the first 4 columns in the rfm DataFrame
neg_check = 0
neg_col_start = 4

for neg_col_start in list(rfm.columns):
    if any(rfm[col] < 0):
        neg_check += 1

print(f'There are {neg_check} negative values.')

rfm_only.describe()
"""
            recency     frequency      monetary
count  96469.000000  96469.000000  96469.000000
mean     240.660844      1.500316    159.855320
std      152.831793      0.500002    218.820934
min        1.000000      1.000000      9.590000
25%      117.000000      1.000000     61.880000
50%      222.000000      2.000000    105.280000
75%      350.000000      2.000000    176.330000
max      696.000000      2.000000  13664.080000
"""

# Use IQR method to remove outliers
def outliers_iqr(df, column):
    quartile_1 = df[column].quantile(0.25)
    quartile_3 = df[column].quantile(0.75)
    iqr = quartile_3 - quartile_1
    
    df = df.loc[lambda df: ~((df[column] < (quartile_1 - 1.5 * iqr)) | (df[column] > (quartile_3 + 1.5 * iqr)))]
    return df


# remove outliers recency and monetary columns using IQR method
rfm_outliers_rem = rfm_only.pipe(outliers_iqr, 'monetary').pipe(outliers_iqr, 'recency')
rfm_outliers_rem.describe()

# check collinearity
sns.heatmap(rfm_outliers_rem.iloc[:, 0:3].corr())
    
"""
-------------------------------------------------------------------------------
-- Visualize RFM variables and Prepare Data for K-Means Clustering --
1. Ensure Normal distribution
2. Ensure variables on same scale (normalized or standardized to have same mean/variance)
-------------------------------------------------------------------------------
"""
scale = StandardScaler()
rfm_norm = pd.DataFrame(scale.fit_transform(rfm_outliers_rem))
rfm_norm.columns = ['norm_Recency', 'norm_Frequency', 'norm_Monetary']
rfm_norm.describe()
"""
       norm_Recency  norm_Frequency  norm_Monetary
count  8.889400e+04    8.889400e+04   8.889400e+04
mean  -1.998287e-17   -1.251727e-16  -2.856750e-16
std    1.000006e+00    1.000006e+00   1.000006e+00
min   -1.570254e+00   -1.000968e+00  -1.461550e+00
25%   -8.097490e-01   -1.000968e+00  -7.820302e-01
50%   -1.213614e-01    9.990330e-01  -2.488639e-01
75%    7.178159e-01    9.990330e-01   5.599634e-01
max    2.986217e+00    9.990330e-01   3.228151e+00
"""

figs, ax = plt.subplots(1, 3, figsize = (20, 5))
for i, feature in enumerate(list(rfm_norm.columns)):
    sns.distplot(rfm_norm[feature], ax = ax[i])

    
"""
-------------------------------------------------------------------------------
--Segmentation using K-Means Clustering --
-------------------------------------------------------------------------------
"""
sse_k = []
range_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_cluster in range_clusters:
    kmeans = KMeans(n_clusters = num_cluster, random_state = 808, max_iter = 50).fit(rfm_norm)
    sse_k.append(kmeans.inertia_)
    
    cluster_labs = kmeans.labels_
    
    silh_avg = silhouette_score(rfm_norm, cluster_labs)
    print("For n_cluster = {0}, the silhouette score is {1}".format(num_cluster, silh_avg))
"""
For n_cluster = 2, the silhouette score is 0.36933829230075793
For n_cluster = 3, the silhouette score is 0.37394991022444046
For n_cluster = 4, the silhouette score is 0.3668156724459828
For n_cluster = 5, the silhouette score is 0.375814978471408
For n_cluster = 6, the silhouette score is 0.39589210869763936
For n_cluster = 7, the silhouette score is 0.39342433407926597
For n_cluster = 8, the silhouette score is 0.3779786858042015
"""
    
sns.pointplot(x = list(range(1, 8)), y = sse_k)
plt.show()

# visualize silhouette score
silhouette_visualizer(KMeans(5, random_state=808), rfm_norm, colors='yellowbrick')

# Identify Clusters
model = KMeans(n_clusters = 5, random_state = 808).fit(rfm_norm)
centers = model.cluster_centers_
rfm_rev = pd.DataFrame(scale.inverse_transform(rfm_outliers_rem))
rfm_rev.columns = rfm_outliers_rem.columns
rfm_rev['cust_id'] = rfm_outliers_rem.index
rfm_rev['cluster'] = model.labels_

melted_rfm_norm = pd.melt(rfm_rev.reset_index(),
                          id_vars = ['cust_id', 'cluster'],
                          value_vars = ['recency', 'frequency', 'monetary'],
                          var_name = 'Features',
                          value_name = 'Value')
sns.lineplot(data = melted_rfm_norm, x = 'Features', y = 'Value', hue = 'cluster')
plt.legend()

stat_summary = rfm_rev.groupby('cluster').agg({
    'recency': ['mean', 'min', 'max'],
    'frequency': ['mean', 'min', 'max'],
    'monetary': ['mean', 'min', 'max', 'count']})




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

For n_cluster = 2, the silhouette score is 0.4676789839579955
For n_cluster = 3, the silhouette score is 0.5005401196282795
For n_cluster = 4, the silhouette score is 0.5124271121076909
For n_cluster = 5, the silhouette score is 0.438287736894351
For n_cluster = 6, the silhouette score is 0.4573107617226523
For n_cluster = 7, the silhouette score is 0.45779552888285957
For n_cluster = 8, the silhouette score is 0.4140360928254224

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
"""