from sklearn.cluster import KMeans
import numpy as np








def elbowKmeans(data1):
  variances = []
  for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data1)
    variances.append(kmeans.inertia_)

 
  # Détermination du nombre optimal de classes
  diff_variances = np.diff(variances)
  diff2_variances = np.diff(diff_variances)
  n_clusters = np.argmin(diff2_variances) + 2

  return n_clusters,variances