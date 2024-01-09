import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import streamlit as st

def train_and_visualize_decision_tree_csv(df):
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

    # Créez un modèle d'arbre de décision avec scikit-learn
    clf = DecisionTreeClassifier(criterion='entropy')  # Vous pouvez utiliser 'gini' comme critère également
    clf.fit(X_train, y_train)

    # Convertissez les noms de classe en chaînes de caractères
    class_names_str = list(map(str, clf.classes_))

    # Affichez l'arbre de décision avec matplotlib
    st.pyplot(plt.figure(figsize=(20, 10)))
    plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=class_names_str, fontsize=10, max_depth=5)
    st.pyplot()

    # Faites des prédictions sur l'ensemble de test
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
 