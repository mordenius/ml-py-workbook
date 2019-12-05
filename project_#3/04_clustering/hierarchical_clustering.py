import datasets.dataset_provider as data_provider
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

dataset = data_provider.get_mall_customers()
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find
# tge optimal number of clusters
# dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
#
# plt.title('Dendogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean distances')
# plt.show()

# Fitting Heirarchical clustering to the dataset
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising the clusterts
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label = 'Clusetr 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label = 'Clusetr 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label = 'Clusetr 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label = 'Clusetr 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label = 'Clusetr 5')

# plt.scatter(hc.cluster_centers_[:,0], hc.cluster_centers_[:,1], s=300, c='yellow', label = 'Centroids 5')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()