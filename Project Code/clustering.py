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
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.manifold import TSNE
pt = PowerTransformer(method = 'box-cox')
ss = StandardScaler()
rs = RobustScaler()
from sklearn.linear_model import LinearRegression #score standardization techniques
lr = LinearRegression()
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
4. Remove collinearity (if any)
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
# check collinearity
sns.heatmap(rfm_only.iloc[:, 0:3].corr())

    
"""
-------------------------------------------------------------------------------
-- Visualize RFM variables and Prepare Data for K-Means Clustering --
1. Ensure Normal distribution
2. Ensure variables on same scale (normalized or standardized to have same mean/variance)
-------------------------------------------------------------------------------
"""
figs, ax = plt.subplots(1, 3, figsize = (20, 5))
# Compare standardization techniques
ss_rfm = pd.DataFrame(ss.fit_transform(rfm_only))
ss_rfm.columns = ['norm_Recency', 'norm_Frequency', 'norm_Monetary']
for i, measure in enumerate(list(ss_rfm.columns)):
    sns.displot(ss_rfm[measure], ax = ax[i])
    
rs_rfm = pd.DataFrame(rs.fit_transform(rfm_only))
rs_rfm.columns = ['norm_Recency', 'norm_Frequency', 'norm_Monetary']
for i, measure in enumerate(list(rs_rfm.columns)):
    sns.displot(rs_rfm[measure], ax = ax[i])
    
pt_rfm = pd.DataFrame(pt.fit_transform(rfm_only))
pt_rfm.columns = ['norm_Recency', 'norm_Frequency', 'norm_Monetary']
for i, measure in enumerate(list(pt_rfm.columns)):
    sns.displot(pt_rfm[measure], ax = ax[i])
    
# The Box-Cox technique produced the most normal distributions
pt_rfm.describe()
"""
PowerTransformer (Box-Cox)
       norm_Recency  norm_Frequency  norm_Monetary
count  9.646900e+04    9.646900e+04   9.646900e+04
mean   2.027723e-16   -1.502563e-17  -7.725676e-16
std    1.000005e+00    1.000005e+00   1.000005e+00
min   -2.377072e+00   -1.000633e+00  -3.907618e+00
25%   -7.140577e-01   -1.000633e+00  -6.755527e-01
50%    4.443087e-02    9.993679e-01   3.983715e-02
75%    7.715654e-01    9.993679e-01   6.629792e-01
max    2.280559e+00    9.993679e-01   3.950724e+00
"""
    
    
"""
-------------------------------------------------------------------------------
--Segmentation using K-Means Clustering --
-------------------------------------------------------------------------------
"""
# Choosing the optimal cluster size
kmeans = KMeans(n_init=10, random_state = 808) # num of times KMeans is run w/ different centroid seeds
visualizer = KElbowVisualizer(kmeans, k = (2, 9)) 
visualizer.fit(pt_rfm)
visualizer.show()
# According to this graph, the optimal cluster size is 5

# Now, check against silhouette score and plot
silh_avg_scores = []
range_clusters = list(range(2, 9))

for num_cluster in range_clusters:
    kmeans = KMeans(n_clusters = num_cluster, n_init = 10, random_state = 808)
    cluster_labs = kmeans.fit_predict(pt_rfm)
    
    silh_avg = silhouette_score(pt_rfm, cluster_labs)
    silh_avg_scores.append(silh_avg)
    print("For cluster size = {0}, the silhouette score is {1}".format(num_cluster, silh_avg))
"""
For cluster size = 2, the silhouette score is 0.3618521116744522
For cluster size = 3, the silhouette score is 0.31541023666359924
For cluster size = 4, the silhouette score is 0.31748101466042367
For cluster size = 5, the silhouette score is 0.3257072646016416
For cluster size = 6, the silhouette score is 0.34067210416371485
For cluster size = 7, the silhouette score is 0.3334874042237079
For cluster size = 8, the silhouette score is 0.32477104772151466
"""    
plt.figure(figsize = (10,6))
plt.plot(range(2, 9), silh_avg_scores, marker = 'o')
plt.title('Avg Silhouette Scores for Various Cluster Sizes')
plt.xlabel('Cluster Size')
plt.ylabel('Score')
plt.show()
# The silhouette score implies that a cluster size of 6 offers a fair clustering in comparison with the other sizes
# We can ignore a cluster size of 2 as it would be hard to interpret in the context of this case study
# Using the elbow method, either a cluster size of 5 or 6 can be chosen

# Identify Clusters (experiment with 5 and 6)
kmeans = KMeans(n_clusters = 6, random_state = 808).fit(pt_rfm)
centers = kmeans.cluster_centers_
labels = kmeans.labels_
rfm_pre_norm = rfm_only.copy()

# Create a cluster label column in original dataset
rfm_pre_norm = rfm_pre_norm.assign(cluster = labels)
melted_rfm_pre_norm = pd.melt(rfm_pre_norm.reset_index(),
                              id_vars = ['cust_id', 'cluster'],
                              value_vars = ['recency', 'frequency', 'monetary'],
                              var_name = 'Measure',
                              value_name = 'Value')
sns.lineplot(data = melted_rfm_pre_norm, x = 'Measure', y = 'Value', hue = 'cluster')
plt.legend(title = 'Clusters')

# create flattened graph
rfm_pre_norm = rfm_pre_norm.rename(str, axis = "columns")
model_tsne = TSNE(random_state = 808)
model_transf = model_tsne.fit_transform(rfm_pre_norm)
plt.title('Flattened Graph of {} Clusters'.format(6))
sns.scatterplot(x = model_transf[:,0], y = model_transf[:,1], hue = labels, style = labels, palette = "Set1")

stat_summary = rfm_pre_norm.groupby('cluster').agg({
    'recency': ['mean', 'min', 'max'],
    'frequency': ['mean', 'min', 'max'],
    'monetary': ['mean', 'min', 'max', 'count']})
"""
K = 6
	recency                     frequency         monetary	
	mean	            min	max	mean min max      mean	                min	    max        count 
cluster										
0	327.97541126547696	46	696	2.0	2	2         259.24778643259896	85.08	7274.88    17203
1	311.81190080531763	5	696	1.0	1	1         55.722123865524736	9.59	98.79      15646
2	82.71401634192331	1	207	1.0	1	1         145.25061785040856	14.29	4163.51    15910
3	81.69059390048155	1	203	2.0	2	2         152.8825573033708	    16.29	6922.21    15575
4	326.051898125901	59	695	1.0	1	1         273.28848570398844	91.16	13664.08   16648
5	302.13198166203915	17	696	2.0	2	2         54.73138503260799	    10.07	97.88	   15487

From the line plot and statistics summary, Cluster 6 has the most distinct clusters
"""


"""
-------------------------------------------------------------------------------
--Segmentation - Combing RFM Scoring and K-Means Clustering --
- For further exploration

Using the original dataset, (not scaled using PowerTransformer), apply a scoring
system of 1 to 5 (5+5+5=15 is the ideal customer, 1+1+1=3 is a churn risk customer) 
-------------------------------------------------------------------------------
"""
rfm_w_scores = rfm_only.copy()

rfm_w_scores['score_Recency'] = pd.qcut(rfm_w_scores['recency'], 5, 
                                        labels = [5, 4, 3, 2, 1])
rfm_w_scores['score_Frequency'] = pd.qcut(rfm_w_scores['frequency'].rank(method="first"), 5, 
                                          labels=[1, 2, 3, 4, 5])
rfm_w_scores['score_Monetary'] = pd.qcut(rfm_w_scores['monetary'], 5, 
                                         labels=[1, 2, 3, 4, 5])
rfm_w_scores[['score_Recency', 'score_Frequency', 'score_Monetary']] = \
    rfm_w_scores[['score_Recency', 'score_Frequency', 'score_Monetary']].apply(pd.to_numeric)

# Choosing the optimal cluster size
kmeans = KMeans(n_init=10, random_state = 808) # num of times KMeans is run w/ different centroid seeds
visualizer = KElbowVisualizer(kmeans, k = (2, 9)) 
visualizer.fit(rfm_w_scores.iloc[:, 3:])
visualizer.show()
# According to this graph, the optimal cluster size is 4


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
For cluster size = 2, the silhouette score is 0.2458697281139218
For cluster size = 3, the silhouette score is 0.24817175658550733
For cluster size = 4, the silhouette score is 0.27162201236850425
For cluster size = 5, the silhouette score is 0.2818675301801122
For cluster size = 6, the silhouette score is 0.3023566314697015
For cluster size = 7, the silhouette score is 0.3007331451905604
For cluster size = 8, the silhouette score is 0.3034281082550997
"""    
plt.figure(figsize = (10,6))
plt.plot(range(2, 9), silh_avg_scores_rfmK, marker = 'o')
plt.title('Avg Silhouette Scores for Various Cluster Sizes')
plt.xlabel('Cluster Size')
plt.ylabel('Score')
plt.show()

# Identify Clusters (experiment with 5 and 6)
kmeans2 = KMeans(n_clusters = 6, random_state = 808).fit(rfm_w_scores.iloc[:, 3:])
centers_rfmK = kmeans2.cluster_centers_
labels_rfmK = kmeans2.labels_
rfm_w_rfm_kclusters = rfm_only.copy()

#rfm_w_rfm_kclusters['cluster'] = labels_rfmK
rfm_w_rfm_kclusters = rfm_w_rfm_kclusters.assign(cluster = labels_rfmK)
melted_rfm_kclusters = pd.melt(rfm_w_rfm_kclusters.reset_index(),
                              id_vars = ['cust_id', 'cluster'],
                              value_vars = ['recency', 'frequency', 'monetary'],
                              var_name = 'Measure',
                              value_name = 'Value')
sns.lineplot(data = melted_rfm_kclusters, x = 'Measure', y = 'Value', hue = 'cluster')
plt.legend(title = 'Clusters')

# create flattened graph
rfm_w_rfm_kclusters = rfm_w_rfm_kclusters.rename(str, axis = "columns")
model2_tsne = TSNE(random_state = 808)
model2_transf = model2_tsne.fit_transform(rfm_w_rfm_kclusters)
plt.title('Flattened Graph of {} Clusters'.format(6))
sns.scatterplot(x = model2_transf[:,0], y = model2_transf[:,1], hue = labels_rfmK, style = labels_rfmK, palette = "Set1")

stat_summary_2 = rfm_w_rfm_kclusters.groupby('cluster').agg({
    'recency': ['mean', 'min', 'max'],
    'frequency': ['mean', 'min', 'max'],
    'monetary': ['mean', 'min', 'max', 'count']})

"""
K = 6
	recency                     frequency                    monetary	
	mean	            min	max	mean                min max  mean	            min     max         count 
cluster																			
0	366.7488984230056	181	696	1.1363636363636365	1	2	 67.83307108070501	10.07	128.31      17248
1	127.85308719669953	1	271	1.7388993776658974	1	2	 51.18797077127474	9.59	85.27       14301
2	352.1508620689655	181	695	1.2698832035595107	1	2	 314.347650166852	128.44	13664.08	14384
3	371.69853077924495	181	696	2.0	                2	2	 124.30396317649247	11.62	4950.34     16131
4	114.36326574401888	1	271	1.0619568715803025	1	2	 162.4280871151164	11.56	4681.78     18642
5	118.57083042568038	1	271	1.899448074605088	1	2	 251.49721753473324	85.36	7274.88     15763

"""