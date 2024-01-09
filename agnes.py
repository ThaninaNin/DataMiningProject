from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import scipy.cluster.hierarchy as shc





def agnes(df):
  agg = AgglomerativeClustering(n_clusters=6, linkage='ward')
  agg.fit(df)

  

  agg_labels = agg.labels_


  # Calculate the distance matrix using ward linkage
  agg_dist = shc.distance.pdist(df, metric='euclidean')
  agg_linkage = shc.linkage(agg_dist, method='ward')
  # calculate the intraclass distance for each cluster
  intraclass_distances = []
  for label in set(agg_labels):
      cluster_points = df[agg_labels == label]
      distance_matrix = pairwise_distances(cluster_points)
      intraclass_distance = distance_matrix.sum() / distance_matrix.size
      intraclass_distances.append(intraclass_distance)

  # calculate the interclass distance between each pair of clusters
  interclass_distances = []
  for i in range(len(set(agg_labels))):
      for j in range(i+1, len(set(agg_labels))):
          cluster_points1 = df[agg_labels == i]
          cluster_points2 = df[agg_labels == j]
          distance_matrix = pairwise_distances(cluster_points1, cluster_points2)
          interclass_distance = distance_matrix.sum() / distance_matrix.size
          interclass_distances.append(interclass_distance)

  return intraclass_distance,interclass_distance,agg_labels,agg_linkage