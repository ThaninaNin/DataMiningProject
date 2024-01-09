import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import streamlit as st

def Ad(df):
    # Vérifier si une colonne cible est présente dans le dataframe
    target_column = df.columns[-1]

    # Divisez vos données en ensembles d'entraînement et de test
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Imputation pour toutes les colonnes (utilisation de la valeur la plus fréquente par défaut)
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Encodage one-hot des variables catégorielles
    X_encoded = pd.get_dummies(X_imputed, columns=X_imputed.select_dtypes(include='object').columns, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Créez un modèle SVM avec scikit-learn
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    # Prédire les scores des probabilités
    y_scores = clf.decision_function(X_test)

    # Convertir les étiquettes en format binaire (0 ou 1)
    y_test_bin = label_binarize(y_test, classes=clf.classes_)

    # Calculer la courbe ROC
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_scores.ravel())
    roc_auc = auc(fpr, tpr)

    # Afficher la courbe ROC
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    st.pyplot()

    # Prédire les étiquettes sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Évaluez les performances du modèle
    accuracy = accuracy_score(y_test, y_pred)

    # Titre pour les performances du modèle
    st.subheader("Performance du Modèle")

    # Afficher l'accuracy
    st.write(f"Précision (Accuracy) : {accuracy:.2%}")

    # Rapport de Classification
    st.subheader("Rapport de Classification")
    st.table(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    # Matrice de Confusion
    st.subheader("Matrice de Confusion")
    st.table(pd.DataFrame(confusion_matrix(y_test, y_pred), index=clf.classes_, columns=clf.classes_))


