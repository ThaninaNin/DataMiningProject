import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import streamlit as st

def train_evaluate_svm(data):
    # Séparer les caractéristiques (X) de la cible (y)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Définir les colonnes catégorielles et numériques
    colonnes_categorielles = X.select_dtypes(include=['object']).columns
    colonnes_numeriques = X.select_dtypes(exclude=['object']).columns

    # Créer les transformers pour le traitement des colonnes
    transformateur_numerique = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    transformateur_categoriel = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Créer le ColumnTransformer
    preprocesseur = ColumnTransformer(
        transformers=[
            ('num', transformateur_numerique, colonnes_numeriques),
            ('cat', transformateur_categoriel, colonnes_categorielles)
        ])

    # Ajouter le classifieur SVM à la fin du pipeline
    pipeline = Pipeline(steps=[('preprocesseur', preprocesseur),
                                 ('classifieur', SVC(kernel='linear', probability=True))])

    # Adapter le pipeline sur les données d'entraînement
    pipeline.fit(X_train, y_train)

    # Évaluer le modèle sur les données de test
    exactitude = pipeline.score(X_test, y_test)
    st.subheader('L\'accuracy')
    st.write(f'Exactitude : {exactitude}')

    # Prédire les scores des probabilités
    y_scores = pipeline.decision_function(X_test)

    # Convertir les étiquettes en format binaire (0 ou 1)
    y_test_bin = label_binarize(y_test, classes=pipeline.classes_)

    # Calculer la courbe ROC
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_scores.ravel())
    roc_auc = auc(fpr, tpr)

    # Afficher la courbe ROC
    st.subheader('Courbe ROC:')
    fig_roc = plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Courbe ROC (aire = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    st.pyplot(fig_roc)

    # Prédire les étiquettes sur l'ensemble de test
    y_pred = pipeline.predict(X_test)

    # Calculer la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Afficher la matrice de confusion avec seaborn
    st.subheader('Matrice de Confusion:')
    fig_conf_matrix = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.title('Matrice de Confusion')
    st.pyplot(fig_conf_matrix)

