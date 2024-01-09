import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt
import preprocessing
import kMeans
import time
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import preprocessing_Zscore
import elbow
from sklearn.cluster import KMeans
import agnes
import scipy.cluster.hierarchy as shc
import diana
from scipy.spatial.distance import cdist
import kmedoids
import dbscan
import os
import Remplacer
import knn_nominale
import plotly.express as px
import naivebayse
import decision_tree
import svm
import Rn
import AD



# The code below is for the title and logo for this page.
st.set_page_config(page_title="Fouille de données", page_icon="📊")


st.title("Projet Feuille de données Classification non supervisée et classification supervisée")

st.write("")



with st.expander("Description de l'application."):

    st.write("")

    st.markdown(
        """
    Etudier des benchmarks de données (dataset) et appliqué les methodes de classification non supervisée(clusturing) et la classification supervisée sur elles ,
    avec l'affichage de performances de classification
    """
    )

    st.write("")


uploaded_file = st.file_uploader("Choisisez votre base de données(dataset)")
if uploaded_file is not None:
    

    dataInit = pd.read_csv(uploaded_file, delimiter=';',na_values='?')
    df_copy=dataInit.copy()
   
    algorithme_type = st.sidebar.selectbox("Choisissez le type d'algorithme", ['Non supervisée', 'Supervisée'])   
    if algorithme_type == 'Non supervisée':
        dataMid,dataFinal= preprocessing.preprocessingMinMax(dataInit)
        algorithme = st.sidebar.radio(
        "Algorithme : ",
        ('K-means', 'Elbow', 'K-Medoid', 'Dbscan','Agnes','Diana','Etude comparative'))

        placeholder = st.empty()


        
        col0, col1,col2 = st.columns(3)
        

        with col0 :
            with st.expander("Les données avant le processing"):
                st.write(dataInit)

        with col1 :
            with st.expander("Les données aprés le processing avec méthode Min Max"):
                st.write(dataFinal)

        with col2 : 
            with st.expander("Les données aprés le processing avec la methode z score"):
                datta = preprocessing_Zscore.preprocessing_Zscore(dataInit)
                st.write(datta)

        if (algorithme == 'K-means'):
            container1 = st.container()
            sizeData = dataFinal.shape[0]
            k = st.sidebar.slider('Nombre de cluster : ',1, sizeData,1)

            with container1 :
                col3,col4 = st.columns(2)
                clusters,intraclasse,interclasse,label = kMeans.kMeanCustom(k,dataFinal)

                with col3 :
                    for label, indices in clusters.items():
                        st.write("Cluster : ",label,dataInit.iloc[indices])

                
            
                with col4 : 
                    
                    st.metric(label="Intraclasse", value=intraclasse)
                    st.metric(label="Interclasse", value=interclasse)

                
                

        if (algorithme == 'Elbow'):
            container1 = st.container()
            

            with container1 :

                number_cluster,variances = elbow.elbowKmeans(dataFinal)
                st.metric(label="elbow", value=number_cluster)

                chart_data = pd.DataFrame({'k': range(1, 11),'variance':variances})
                
                # Basic Altair line chart where it picks automatically the colors for the lines
                basic_chart = alt.Chart(chart_data).mark_line().encode(
                    x='k',
                    y='variance',
                    # legend=alt.Legend(title='Animals by year')
                )
                st.altair_chart(basic_chart)
                #st.line_chart(chart_data,x= "Nombre de clusters", y="Variance)


        if(algorithme == 'Agnes'):
            container1 = st.container()
            col0, col1 = st.columns(2)

            with container1 :
                intraclass_distances,interclass_distances,agg_labels,agg_linkage = agnes.agnes(dataFinal)
                with col0:
                    with st.expander("Dendogram pour algorithme agnes"):
                        plt.figure(figsize=(10, 7))
                        plt.title("Dendrogramme d'AGNES")
                        dend = shc.dendrogram(agg_linkage, labels=agg_labels)
                        plt.show()
                        st.pyplot()
                with col1:
                    st.metric(label="intraclasse", value=intraclass_distances)
                    st.metric(label="interclasse", value=interclass_distances)
                    


        if(algorithme == 'Diana'):
            container1 = st.container()
            col0, col1 = st.columns(2)

            with container1 :
                diana_dist,intraclass_distances,interclass_distances,diana_linkage,diana_labels = diana.Diana(dataFinal)
                with col0:
                    with st.expander("Dendogram pour algorithme Diana"):
                        # Create the dendrogram for DIANA
                        plt.figure(figsize=(10, 7))
                        plt.title("Dendrogramme de DIANA")
                        dend = shc.dendrogram(diana_linkage, labels=diana_labels)
                        plt.show()
                        st.pyplot()
                with col1:
                    st.metric(label="intraclasse", value=intraclass_distances)
                    st.metric(label="interclasse", value=interclass_distances)


        if(algorithme == 'K-Medoid'):
            maxCluster = dataInit.shape[0]
            k = st.sidebar.slider('Choisir le nombre de cluster', 1, maxCluster, 1)
            X = np.array(dataInit.iloc[:, :-1])
            attribute_names = list(dataInit.columns)[:-1]
            medoids, labels, intra_cluster_distance, inter_cluster_distance = kmedoids.kmedoids(X, k, tmax=100)
            medoids_df = pd.DataFrame(medoids, columns=attribute_names)
            container1 = st.container()
            col0, col1 = st.columns(2)

            with container1 :
                with col0:
                    with st.expander("les clusters de algorithme K-Medoids"):
                         for i in range(k):
                            cluster_indices = np.where(labels == i)[0]
                            cluster_df = dataInit.iloc[cluster_indices, :]
                            st.write("Cluster ", i, " :")
                            st.write(cluster_df)
                with col1:
                    st.metric(label="intraclasse", value=intra_cluster_distance)
                
                    st.metric(label="interclasse", value=inter_cluster_distance)


        if(algorithme == 'Dbscan'):
            eps = st.sidebar.slider('Choisir epsilon', 0.0, 100.0, 0.1)
            maxmin_samples = dataFinal.shape[0]
            st.write(maxmin_samples)
            min_samples = st.sidebar.slider('Choisir min_samples', 1, maxmin_samples, 10)
            n_clusters, outliers_idx, inertia_intra, inertia_inter, data1, outliers_data = dbscan.dbscan_custom(dataFinal,eps,min_samples)
            container1 = st.container()
            col0, col1,col2 = st.columns(3)

            with container1 :
                with col0:
                    st.title("Out layers")
                    st.write(outliers_idx)
                with col1:
                    st.title("Données des outliers")
                    st.write(data1.loc[outliers_data.index])

                with col2 :
                    st.metric(label="nombre de cluster", value=n_clusters)

                    st.metric(label="intraclasse", value=inertia_intra)
                    st.metric(label="interclasse", value=inertia_inter)



        if(algorithme == 'Etude comparative'):
            st.title("Etude comparative :")
            maxCluster = dataFinal.shape[0]
            st.sidebar.title("Paramétres de l'étude comparative")
            k = st.sidebar.slider('Nombre de cluster', 1, maxCluster, 1)
            eps = st.sidebar.slider('Epsilon : ', 0.0, 100.0, 0.1)
            min_samples = st.sidebar.slider('Choisir min_samples', 1, maxCluster, 10)

            X = np.array(dataFinal.iloc[:, :-1])
            clusters,intraclasse_kmean,interclasse_kmeans,label = kMeans.kMeanCustom(k,dataFinal)
            intraclass_distances_agnes,interclass_distances_agnes,agg_labels,agg_linkage = agnes.agnes(dataFinal)
            diana_dist,intraclass_distances_diana,interclass_distances_diana,diana_linkage,diana_labels = diana.Diana(dataFinal)
            medoids, labels, intra_cluster_distance_kmedoids, inter_cluster_distance_kmedoids = kmedoids.kmedoids(X, k, tmax=100)
            n_clusters, outliers_idx, inertia_intra_dbscan, inertia_inter_dbscan, data1, outliers_data = dbscan.dbscan_custom(dataFinal,eps,min_samples)
            data_chart_intra = pd.DataFrame({
            'methode': ['K-means', 'Agnes', 'Diana','K-Medoid', 'Dbscan'],
            'intraclasse value': [intraclasse_kmean, intraclass_distances_agnes, intraclass_distances_diana, intra_cluster_distance_kmedoids,inertia_intra_dbscan]
            })    
            # Create a bar chart using Altair
            chart = alt.Chart(data_chart_intra).mark_bar().encode(
                x='methode',
                y='intraclasse value'
            )

            # Render the chart using Streamlit
            st.altair_chart(chart, use_container_width=True)  


            data_chart_inter = pd.DataFrame({
            'methode': ['K-means', 'Agnes', 'Diana','K-Medoid', 'Dbscan'],
            'interclasse value': [interclasse_kmeans, interclass_distances_agnes,interclass_distances_diana, inter_cluster_distance_kmedoids,inertia_inter_dbscan]
            })    

            # Create a bar chart using Altair
            chart = alt.Chart(data_chart_inter).mark_bar().encode(
                x='methode',
                y='interclasse value'
            )

            # Render the chart using Streamlit
            st.altair_chart(chart, use_container_width=True) 
     
  ###################################################################################################################################################################           
    else:
        
         
        algorithme = st.sidebar.radio(
        "Algorithme : ",
        ('KNN', 'Naive Bayes', 'Arbre de decision', 'Reseaux de Neuron','SVM'))
        placeholder = st.empty()


        
        col0, col1,col2 = st.columns(3)
        

        with col0 :
            with st.expander("Les données sans preprocessing"):
             st.write(dataInit)
        
                

        with col1 :
            with st.expander("Les données aprés le processing(remplacage des valeurs manquante)"):
                dataR= Remplacer.remplacer_valeurs_manquantes_csv(dataInit)
                
                st.write(dataR)

        with col2 : 
            with st.expander("Le nuage des points"):
                    naivebayse.scatter_plot(dataInit)
        if (algorithme == 'KNN'):
            container1 = st.container()
            sizeData = dataInit.shape[0]
            k = st.sidebar.slider('Nombre de voisin les plus proche K: ',1, sizeData,1)

            with container1 :
                col3,col4 = st.columns(2)
                #appelle a la  fonction 

                with col3 :
                    # Exploratory Data Analysis
                    st.subheader("Analyse exploratoire des données")

# Statistique Descriptifs
                    st.write("Statistique Descriptifs:")
                    st.write(dataInit.describe())

# Distribution de la classe
                    target_column = dataInit.columns[-1]
                    class_counts = dataInit[target_column].value_counts()

# Créer un diagramme à barres pour la distribution de la classe cible
                    fig = px.bar(x=class_counts.index, y=class_counts.values, labels={'x': 'Classe', 'y': 'Nombre'}, title='Distribution de la classe cible', width=350)

# Afficher le graphique dans la première colonne
                    col3.plotly_chart(fig)

                with col4 : 
                   dataR= Remplacer.remplacer_valeurs_manquantes_csv(dataInit)
                   knn_nominale.main(df_copy,k) 
                     

                
                

        if (algorithme == 'Naive Bayes'):
            container1 = st.container()   
            with container1 :
                col0,col1 = st.columns(2)
                #appelle a la  fonction 

                with col0 :
                    # Exploratory Data Analysis
                     st.subheader("Analyse exploratoire des données")
                    
                     st.write("Statistique Descriptifs:")
                     st.write(dataInit.describe())

                     st.write("la distribution de la class:")
                     st.write(df_copy[df_copy.columns[-1]].value_counts())
                    # En supposant que la colonne cible s'appelle 'target_column'
                     target_column = df_copy.columns[-1]

# Créer un diagramme à barres pour la distribution de la classe cible
                     fig = px.bar(df_copy[target_column].value_counts(), x=df_copy[target_column].value_counts().index, y=df_copy[target_column].value_counts().values, labels={'x': 'Classe', 'y': 'Nombre'}, title='Distribution de la classe cible',width=350)


# Afficher le graphique dans la première colonne
                     col0.plotly_chart(fig)
            
                with col1 : 
                   
                   naivebayse.main(df_copy) 

                


        if(algorithme == 'Arbre de decision'):
            container1 = st.container()
            col0, col1 = st.columns(2)

            with container1 :
                
                with col0:
                    
                    st.subheader("Analyse exploratoire des données")

# Statistique Descriptifs
                    st.write("Statistique Descriptifs:")
                    st.write(dataInit.describe())

# Distribution de la classe
                    target_column = dataInit.columns[-1]
                    class_counts = dataInit[target_column].value_counts()

# Créer un diagramme à barres pour la distribution de la classe cible
                    fig = px.bar(x=class_counts.index, y=class_counts.values, labels={'x': 'Classe', 'y': 'Nombre'}, title='Distribution de la classe cible', width=350)

# Afficher le graphique dans la première colonne
                    col0.plotly_chart(fig)
                with col1:
                    decision_tree.train_and_visualize_decision_tree_csv(df_copy)
                   





        if(algorithme == 'SVM'):
            container1 = st.container()
            col0, col1 = st.columns(2)

            with container1 :
                
                with col0:
                    
                    st.subheader("Analyse exploratoire des données")

# Statistique Descriptifs
                    st.write("Statistique Descriptifs:")
                    st.write(dataInit.describe())

# Distribution de la classe
                    target_column = dataInit.columns[-1]
                    class_counts = dataInit[target_column].value_counts()

# Créer un diagramme à barres pour la distribution de la classe cible
                    fig = px.bar(x=class_counts.index, y=class_counts.values, labels={'x': 'Classe', 'y': 'Nombre'}, title='Distribution de la classe cible', width=350)

# Afficher le graphique dans la première colonne
                    col0.plotly_chart(fig)
                with col1:
                    svm.train_evaluate_svm(df_copy)
        if(algorithme == 'Reseaux de Neuron'):
            container1 = st.container()
            col0, col1 = st.columns(2)

            with container1 :
                
                with col0:
                    
                    st.subheader("Analyse exploratoire des données")

# Statistique Descriptifs
                    st.write("Statistique Descriptifs:")
                    st.write(dataInit.describe())

# Distribution de la classe
                    target_column = dataInit.columns[-1]
                    class_counts = dataInit[target_column].value_counts()

# Créer un diagramme à barres pour la distribution de la classe cible
                    fig = px.bar(x=class_counts.index, y=class_counts.values, labels={'x': 'Classe', 'y': 'Nombre'}, title='Distribution de la classe cible', width=350)

# Afficher le graphique dans la première colonne
                    col0.plotly_chart(fig)
                with col1:
                    Rn.rn(df_copy)


           




       
