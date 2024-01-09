import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt  # Ajoutez cette ligne

def plot_roc_curve(y_true, y_prob, label='(Courbe ROC)'):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'{label} (aire = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de faux positif')
    ax.set_ylabel('Taux de vrai positif')
    ax.set_title(f'Courbe Caractéristique de Fonctionnement du Récepteur ({label})')
    ax.legend(loc='lower right')

    st.pyplot(fig)

def rn(df):
    # Extract features (X) and target variable (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Définir les caractéristiques numériques et catégorielles
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Créer un pipeline de prétraitement
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Imputer les valeurs manquantes avec la moyenne
        ('scaler', StandardScaler())  # Standardiser les caractéristiques numériques
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputer les valeurs manquantes avec la valeur la plus fréquente
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encodage one-hot des caractéristiques catégorielles
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Adapter et transformer les données
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Encoder les étiquettes catégorielles
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Réseau neuronal
    st.subheader("Entraînement du Réseau Neuronal:")

    # Construire le modèle de réseau neuronal en utilisant Keras
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=X_train_preprocessed.shape[1]))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=2, activation='relu'))
    model.add(Dense(units=len(pd.unique(y_train)), activation='softmax'))  # Utiliser softmax pour la classification multi-classe

    # Compiler le modèle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle
    history = model.fit(X_train_preprocessed, y_train_encoded, epochs=20, batch_size=32, validation_data=(X_test_preprocessed, y_test_encoded))

    # Afficher l'historique d'entraînement
    """st.subheader("Précision d'Entraînement et de Validation au Fil des Époques:")
    st.line_chart(history.history['accuracy'])
    st.line_chart(history.history['val_accuracy'])

    # Évaluer le modèle sur l'ensemble de test"""
    loss, accuracy = model.evaluate(X_test_preprocessed, y_test_encoded)
    st.subheader(f"Précision sur l'Ensemble de Test: {accuracy}")
    #st.subheader("Précision d'Entraînement et de Validation au Fil des Époques:")
    fig_accuracy, ax_accuracy = plt.subplots(figsize=(8, 6))
    ax_accuracy.plot(history.history['accuracy'], label='Précision d\'Entraînement')
    #ax_accuracy.plot(history.history['val_accuracy'], label='Précision de Validation')
    ax_accuracy.set_xlabel('Époques')
    ax_accuracy.set_ylabel('Précision')
    #ax_accuracy.set_title('Précision d\'Entraînement et de Validation au Fil des Époques')
    ax_accuracy.legend()
    st.pyplot(fig_accuracy)
    # Tracer la Courbe ROC
    st.subheader("La Courbe de (ROC):")
    y_prob = model.predict(X_test_preprocessed)[:, 1]
    plot_roc_curve(y_test_encoded, y_prob)


