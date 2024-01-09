import scipy.cluster.hierarchy as shc
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering



def Diana(df):


  diana = AgglomerativeClustering(n_clusters=3, linkage='single')
  diana.fit(df)
  diana_labels = diana.labels_

  # Calculate the distance matrix using complete linkage
  diana_dist = shc.distance.pdist(df, metric='euclidean')
  diana_linkage = shc.linkage(diana_dist, method='complete')



  # calculate the intraclass distance for each cluster
  intraclass_distances = []
  for label in set(diana_labels):
    cluster_points = df[diana_labels == label]
    distance_matrix = pairwise_distances(cluster_points)
    intraclass_distance = distance_matrix.sum() / distance_matrix.size
    intraclass_distances.append(intraclass_distance)

  # calculate the interclass distance between each pair of clusters
  interclass_distances = []
  for i in range(len(set(diana_labels))):
    for j in range(i+1, len(set(diana_labels))):
      cluster_points1 = df[diana_labels == i]
      cluster_points2 = df[diana_labels == j]
      distance_matrix = pairwise_distances(cluster_points1, cluster_points2)
      interclass_distance = distance_matrix.sum() / distance_matrix.size
      interclass_distances.append(interclass_distance)
  

  return diana_dist,intraclass_distance,interclass_distance,diana_linkage,diana_labels