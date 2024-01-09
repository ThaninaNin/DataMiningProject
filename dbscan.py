import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN











def dbscan_custom(data,eps,min_samples):
  data1= data
  # Sélection des colonnes numériques
  numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

  # Remplacement des valeurs manquantes par la moyenne
  data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

  # Encodage One-Hot des colonnes non numériques
  non_numeric_cols = data.select_dtypes(include=['object']).columns
  data = pd.get_dummies(data, columns=non_numeric_cols)

  # Normalisation des données
  scaler = StandardScaler()
  data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

  # DBSCAN avec paramètres par défaut
  #dbscan = DBSCAN()
  dbscan = DBSCAN(eps, min_samples=10)

  dbscan.fit(data)

  # Affichage des clusters
  #print(dbscan.labels_)

  # nombre de clusters (excluant le bruit)
  n_clusters = len(np.unique(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
  print(f"Nombre de clusters: {n_clusters}")

  # indices des points considérés comme des outliers (-1)
  outliers_idx = np.where(dbscan.labels_ == -1)[0]
  print("out layers",outliers_idx)

  outliers_data = data1.iloc[outliers_idx]
  print("Données des outliers : ")
  print(data1.loc[outliers_data.index])

  #print("Données des outliers : ")
  #print("out layer data",outliers_data)

  print("\n ")
  print("************ ")

  # Inertie intra-classe et
  groups = {}
  for label in np.unique(dbscan.labels_):
      if label != -1:
          groups[label] = data1[dbscan.labels_ == label]
          
          # Création d'un DataFrame pour ce cluster
          cluster_df = pd.DataFrame(groups[label], columns=data1.columns)
          
          # Affichage des données de ce cluster
          print(f"Cluster {label}:")
          print(cluster_df)

  # Inertie intra-classe et inter-classe
  groups = {}
  for label in np.unique(dbscan.labels_):
      if label != -1:
          groups[label] = data[dbscan.labels_ == label]
          
  inertia_inter = 0
  inertia_intra = 0
  for label in groups:
      group = groups[label]
      group_mean = group.mean(axis=0)
      group_size = len(group)
      group_inertia = ((group - group_mean) ** 2).sum().sum()
      inertia_intra += group_inertia
      inertia_inter += group_size * ((group_mean - data.mean(axis=0)) ** 2).sum()

  print(f"Inertie intra-classe : {inertia_intra}")
  print(f"Inertie inter-classe : {inertia_inter}")

  return n_clusters,outliers_idx,inertia_intra,inertia_inter,data1,outliers_data
    
 