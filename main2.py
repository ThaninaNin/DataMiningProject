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

# ... (votre code existant)

# Ajoutez un menu déroulant pour choisir l'algorithme
algorithme_type = st.sidebar.selectbox("Choisissez le type d'algorithme", ['Non supervisé', 'Supervisé'])

if algorithme_type == 'Non supervisé':
    # Menu déroulant pour les algorithmes de classification non supervisée
    algorithme = st.sidebar.selectbox(
        "Choisissez l'algorithme de classification non supervisée",
        ('K-means', 'Elbow', 'K-Medoid', 'Dbscan', 'Agnes', 'Diana', 'Etude comparative')
    )
else:
    # Menu déroulant pour les algorithmes de classification supervisée
    algorithme = st.sidebar.selectbox(
        "Choisissez l'algorithme de classification supervisée",
        ('Votre_Algorithme_Supervisé_1', 'Votre_Algorithme_Supervisé_2', 'Autre')
    )

placeholder = st.empty()

# ... (le reste de votre code en fonction de l'algorithme sélectionné)
