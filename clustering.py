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
from sklearn.preprocessing import StandardScaler, RobustScaler 
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
sns.set(style='whitegrid')
import warnings
warnings.filterwarnings("ignore")

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
rfm.set_index('cust_id', inplace = True)

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
4. Remove noise/outliers (if any)
5. Remove collinearity (if any)
-------------------------------------------------------------------------------
"""
null_check = 0
for col in list(rfm_only.columns):
    if any(rfm_only[col].isnull()):
        null_check += 1

print(f'There are {null_check} NULL values.') #0

# For negatives check, ignore the first 4 columns in the rfm DataFrame
neg_check = 0
neg_col_start = 4

for neg_col_start in list(rfm_only.columns):
    if any(rfm_only[col] < 0):
        neg_check += 1

print(f'There are {neg_check} negative values.') #0

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

# Use IQR method to remove outliers from monetary field
def outliers_iqr(df, column):
    quartile_1 = df[column].quantile(0.25)
    #print(quartile_1)
    quartile_3 = df[column].quantile(0.75)
    #print(quartile_3)
    iqr = quartile_3 - quartile_1
    #print(iqr)
    
    df = df.loc[lambda df: ~((df[column] < (quartile_1 - 1.5 * iqr)) | (df[column] > (quartile_3 + 1.5 * iqr)))]
    return df

# remove outliers recency and monetary columns using IQR method
rfm_outliers_rem = outliers_iqr(rfm_only, 'monetary')
rfm_outliers_rem.describe()
"""
            recency     frequency      monetary
count  88894.000000  88894.000000  88894.000000
mean     240.511294      1.500484    115.046487
std      152.531194      0.500003     72.154254
min        1.000000      1.000000      9.590000
25%      117.000000      1.000000     58.620000
50%      222.000000      2.000000     97.090000
75%      350.000000      2.000000    155.450000
max      696.000000      2.000000    347.970000
"""
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

scale2 = RobustScaler ()
rfm_norm2 = pd.DataFrame(scale.fit_transform(rfm_only))
rfm_norm2.columns = ['norm_Recency', 'norm_Frequency', 'norm_Monetary']
rfm_norm2.describe()
"""
StandardScaler
       norm_Recency  norm_Frequency  norm_Monetary
count  8.889400e+04    8.889400e+04   8.889400e+04
mean  -1.998287e-17   -1.251727e-16  -2.856750e-16
std    1.000006e+00    1.000006e+00   1.000006e+00
min   -1.570254e+00   -1.000968e+00  -1.461550e+00
25%   -8.097490e-01   -1.000968e+00  -7.820302e-01
50%   -1.213614e-01    9.990330e-01  -2.488639e-01
75%    7.178159e-01    9.990330e-01   5.599634e-01
max    2.986217e+00    9.990330e-01   3.228151e+00

vs

RobustScaler
       norm_Recency  norm_Frequency  norm_Monetary
count  9.646900e+04    9.646900e+04   9.646900e+04
mean  -2.850450e-17   -9.530961e-17  -1.065052e-16
std    1.000005e+00    1.000005e+00   1.000005e+00
min   -1.568143e+00   -1.000633e+00  -6.867081e-01
25%   -8.091346e-01   -1.000633e+00  -4.477443e-01
50%   -1.221012e-01    9.993679e-01  -2.494076e-01
75%    7.154252e-01    9.993679e-01   7.528880e-02
max    2.979364e+00    9.993679e-01   6.171391e+01
"""

figs, ax = plt.subplots(1, 3, figsize = (20, 5))
for i, measure in enumerate(list(rfm_norm.columns)):
    sns.distplot(rfm_norm[measure], ax = ax[i])

    
"""
-------------------------------------------------------------------------------
--Segmentation using K-Means Clustering --
-------------------------------------------------------------------------------
"""
# Choosing the optimal cluster size
"""
sse_k = []

for num_cluster in range(0, 8):
    kmeans = KMeans(n_clusters = num_cluster + 1, random_state = 808, max_iter = 50).fit(rfm_norm)
    sse_k.append(kmeans.inertia_)
    
sns.pointplot(x = list(range(1, 9)), y = sse_k)
plt.show()
"""
kmeans = KMeans(n_init=10, random_state = 808) # num of times KMeans is run w/ different centroid seeds
visualizer = KElbowVisualizer(kmeans, k = (2, 9)) 
visualizer.fit(rfm_norm)
visualizer.show()
# According to this graph, the optimal cluster size is 5

# Now, check against silhouette score and plot
silh_avg_scores = []
range_clusters = list(range(2, 9))

for num_cluster in range_clusters:
    kmeans = KMeans(n_clusters = num_cluster, n_init = 10, random_state = 808)
    cluster_labs = kmeans.fit_predict(rfm_norm)
    
    silh_avg = silhouette_score(rfm_norm, cluster_labs)
    silh_avg_scores.append(silh_avg)
    print("For cluster size = {0}, the silhouette score is {1}".format(num_cluster, silh_avg))
"""
For cluster size = 2, the silhouette score is 0.36933829230075793
For cluster size = 3, the silhouette score is 0.37394991022444046
For cluster size = 4, the silhouette score is 0.3668156724459828
For cluster size = 5, the silhouette score is 0.375814978471408
For cluster size = 6, the silhouette score is 0.39589210869763936
For cluster size = 7, the silhouette score is 0.39342433407926597
For cluster size = 8, the silhouette score is 0.3779786858042015
"""    
plt.figure(figsize = (10,6))
plt.plot(range(2, 9), silh_avg_scores, marker = 'o')
plt.title('Avg Silhouette Scores for Various Cluster Sizes')
plt.xlabel('Cluster Size')
plt.ylabel('Score')
plt.show()
# The silhouette score implies that a cluster size of 6 offers a fair clustering in comparison with the other sizes
# Using the elbow method, either a cluster size of 5 or 6 can be chosen
# To compromise between the two suggestions, 6 was chosen



# Identify Clusters
model = KMeans(n_clusters = 6, random_state = 808).fit(rfm_norm)
centers = model.cluster_centers_
labels = model.labels_
rfm_pre_norm = rfm_outliers_rem.copy()

"""
# Visualize Clusters
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create a 3D scatter plot, colored by cluster
scatter = ax.scatter(rfm_pre_norm.iloc[:, 0], rfm_pre_norm.iloc[:, 1], rfm_pre_norm.iloc[:, 2], c=labels, cmap='coolwarm', edgecolor='k', s=50, alpha=0.8, label='Customer Clusters')
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='o', color='red', s=200, edgecolor='k', label='Cluster Centers')

# Labels/title
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary Value')
ax.set_title('K-Means Clustering 3D Visualization', size=20)

# Enhancements
ax.grid(True)
ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax.view_init(elev=20, azim=120)  # Adjust for best angle

# Legend
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
ax.legend(['Cluster Centers'], loc='upper left')

plt.tight_layout()
plt.show()
"""

rfm_pre_norm['cluster'] = labels
melted_rfm_pre_norm = pd.melt(rfm_pre_norm.reset_index(),
                              id_vars = ['cust_id', 'cluster'],
                              value_vars = ['recency', 'frequency', 'monetary'],
                              var_name = 'Measure',
                              value_name = 'Value')
sns.lineplot(data = melted_rfm_pre_norm, x = 'Measure', y = 'Value', hue = 'cluster')
plt.legend(title = 'Clusters')

stat_summary = rfm_pre_norm.groupby('cluster').agg({
    'recency': ['mean', 'min', 'max'],
    'frequency': ['mean', 'min', 'max'],
    'monetary': ['mean', 'min', 'max', 'count']})
"""
	recency                     frequency		monetary	
	mean	            min	max	mean min max	mean	            min	    max	    count 
cluster										
0	129.34278996063998	1	269	1.0	 1	 1	    81.40977764146604	9.59	175.41	19563
1	403.97120076981236	261	696	2.0	 2	 2	    89.14134510962953	10.07	267.49	14549
2	400.91107247927255	261	696	1.0	 1	 1	    86.58895159133459	11.63	266.11	14956
3	214.14321012057718	1	695	2.0	 2	 2	    221.6418926665349	140.08	347.97	10118
4	133.303788528477	1	279	2.0	 2	 2	    80.16642435554658	10.89	171.12	19823
5	219.23045017703592	2	695	1.0	 1	 1	    223.63879312089026	141.23	347.89	9885

From the line plot and statistics summary, cluster 3 is the group of customers
to target as they have the highest purchase mean, a relatively short time since their
last order, and more transactions. Customers either had 1 or 2 orders placed, so 
greater emphasis was pllaced on the recency and monetary metrics in interpreting
the results.
"""


"""
-------------------------------------------------------------------------------
--Segmentation - Combing RFM Scoring and K-Means Clustering --

Using the original dataset, (not scaled using StandardScale), apply a scoring
system of 1 to 5 (5+5+5=15 is the ideal customer, 1+1+1=3 is a churn risk customer) 
-------------------------------------------------------------------------------
"""
rfm_w_scores = rfm_outliers_rem.copy()

rfm_w_scores['score_Recency'] = pd.qcut(rfm_w_scores['recency'], 5, 
                                        labels = [5, 4, 3, 2, 1])
rfm_w_scores['score_Frequency'] = pd.qcut(rfm_w_scores['frequency'].rank(method="first"), 5, 
                                          labels=[1, 2, 3, 4, 5])
rfm_w_scores['score_Monetary'] = pd.qcut(rfm_w_scores['monetary'], 5, 
                                         labels=[1, 2, 3, 4, 5])
rfm_w_scores[['score_Recency', 'score_Frequency', 'score_Monetary']] = \
    rfm_w_scores[['score_Recency', 'score_Frequency', 'score_Monetary']].apply(pd.to_numeric)

# Choosing the optimal cluster size
"""
sse_k_rfm = []

for num_cluster in range(0, 8):
    kmeans = KMeans(n_clusters = num_cluster + 1, random_state = 808, 
                    max_iter = 50).fit(rfm_w_scores.iloc[:, 3:])
    sse_k_rfm.append(kmeans.inertia_)
    
sns.pointplot(x = list(range(1, 9)), y = sse_k_rfm)
plt.show()
"""
kmeans = KMeans(n_init=10, random_state = 808) # num of times KMeans is run w/ different centroid seeds
visualizer = KElbowVisualizer(kmeans, k = (2, 9)) 
visualizer.fit(rfm_w_scores.iloc[:, 3:])
visualizer.show()
# According to this graph, the optimal cluster size is 5


# Now, check against silhouette score and plot
silh_avg_scores_rfmK = []
range_clusters = list(range(2, 9))

for num_cluster in range_clusters:
    kmeans = KMeans(n_clusters = num_cluster, n_init = 10, random_state = 808)
    cluster_labs = kmeans.fit_predict(rfm_w_scores.iloc[:, 3:])
    
    silh_avg_rfm = silhouette_score(rfm_w_scores.iloc[:, 3:], cluster_labs)
    silh_avg_scores_rfmK.append(silh_avg_rfm)
    print("For cluster size = {0}, the silhouette score is {1}".format(num_cluster, silh_avg_rfm))
"""
For cluster size = 2, the silhouette score is 0.24461560559334516
For cluster size = 3, the silhouette score is 0.25042562260191803
For cluster size = 4, the silhouette score is 0.27037835821642353
For cluster size = 5, the silhouette score is 0.2810535622082644
For cluster size = 6, the silhouette score is 0.3018595630335999
For cluster size = 7, the silhouette score is 0.299916182117888
For cluster size = 8, the silhouette score is 0.30263441504036825
"""    
plt.figure(figsize = (10,6))
plt.plot(range(2, 9), silh_avg_scores_rfmK, marker = 'o')
plt.title('Avg Silhouette Scores for Various Cluster Sizes')
plt.xlabel('Cluster Size')
plt.ylabel('Score')
plt.show()
# The silhouette score implies that a cluster size of 6 offers a fair clustering in comparison with the other sizes
# Using the elbow method, either a cluster size of 5 or 6 can be chosen
# To compromise between the two suggestions, 6 was chosen in this scenario as well.

# Identify Clusters
model2 = KMeans(n_clusters = 6, random_state = 808).fit(rfm_w_scores.iloc[:, 3:])
centers_rfmK = model2.cluster_centers_
labels_rfmK = model2.labels_
rfm_w_rfm_kclusters = rfm_outliers_rem.copy()

rfm_w_rfm_kclusters['cluster'] = labels_rfmK
melted_rfm_kclusters = pd.melt(rfm_w_rfm_kclusters.reset_index(),
                              id_vars = ['cust_id', 'cluster'],
                              value_vars = ['recency', 'frequency', 'monetary'],
                              var_name = 'Measure',
                              value_name = 'Value')
sns.lineplot(data = melted_rfm_kclusters, x = 'Measure', y = 'Value', hue = 'cluster')
plt.legend(title = 'Clusters')

stat_summary_2 = rfm_w_rfm_kclusters.groupby('cluster').agg({
    'recency': ['mean', 'min', 'max'],
    'frequency': ['mean', 'min', 'max'],
    'monetary': ['mean', 'min', 'max', 'count']})

