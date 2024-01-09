from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt





def kMeanCustom(k,data):
  # Cluster data
  kmeans = KMeans(n_clusters=k, n_init=10)
  label = kmeans.fit_predict(data)
  

  # Dictionnaire pour stocker les indices de chaque cluster
  clusters = {}
  for i, label in enumerate(kmeans.labels_):
    if label not in clusters:
      clusters[label] = [i]
    else:
      clusters[label].append(i)


  # Calcul de l'intraclasse
  intraclasse = kmeans.inertia_

  # Calcul de l'interclasse
  interclasse = pairwise_distances(kmeans.cluster_centers_, Y=None, metric='euclidean').sum()

  

  return clusters,intraclasse,interclasse,label



