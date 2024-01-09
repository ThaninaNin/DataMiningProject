import pandas as pd
import numpy as np
import csv
import random
from scipy.spatial.distance import cdist








def kmedoids(X, k, tmax=100):
    m, n = X.shape

    # Initialisation aléatoire des médoides
    medoids = np.array(random.sample(list(X), k))
    old_medoids = np.zeros((k, n))

    for i in range(tmax):
        # Étape d'affectation : attribution de chaque point au médoid le plus proche
        distances = cdist(X, medoids, metric='euclidean')
        labels = np.argmin(distances, axis=1)

        for i in range(k):
            indices = np.where(labels == i)[0]
            cluster_points = X[indices]
            old_medoids[i, :] = medoids[i]
            medoids[i, :] = cluster_points[np.argmin(cdist(cluster_points, cluster_points, metric='euclidean').sum(axis=1)), :]

        if np.all(old_medoids == medoids):
            break
            
    # Calcul de la somme des distances entre chaque point et son médoid
    distances = cdist(X, medoids, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    total_distance = np.sum(distances[range(len(labels)), labels])
    
    # Calcul de l'inertie interclasse
    global_centroid = np.mean(X, axis=0)
    medoid_distances = cdist(medoids, [global_centroid], metric='euclidean')
    inter_cluster_distance = np.sum(medoid_distances)**2

    # Calcul de l'inertie intraclasse
    intra_cluster_distances = np.zeros(k)
    for i in range(k):
        cluster_points = X[labels == i]
        intra_cluster_distances[i] = np.sum(cdist(cluster_points, [medoids[i]], metric='euclidean'))
    intra_cluster_distance = (np.sum(intra_cluster_distances))*1/k 

    return medoids, labels, intra_cluster_distance, inter_cluster_distance
   # return medoids, labels
