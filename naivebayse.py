import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype('category')

    df = pd.get_dummies(df, columns=df.select_dtypes(include=['category']).columns)

    missing_values = df.isnull().sum()
    if missing_values.any():
        #st.warning("Missing values detected. Imputing...")

        #st.write("Attributes with missing values:")
        #for col, missing_count in missing_values[missing_values > 0].items():
           # st.write(f"{col}: {missing_count} missing values")

        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(df.iloc[:, :-1]), columns=df.iloc[:, :-1].columns)
        X = X_imputed.values
    else:
        X = df.iloc[:, :-1].values

    y = df.iloc[:, -1].values

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_naive_bayes(X_train, y_train):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    return naive_bayes

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("L'accuracy:")
    st.write(f"Accuracy: {accuracy:.2%}")

    return accuracy

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Naive Bayes')
    plt.legend(loc='lower right')

    st.pyplot()

def scatter_plot(df):
    plt.figure(figsize=(8, 6))
    
    # Utilisez des colonnes appropriées pour les axes x et y (ajustez selon votre jeu de données)
    x_column = df.columns[0]
    y_column = df.columns[1]

    sns.scatterplot(x=x_column, y=y_column, hue=df.columns[-1], data=df)
    plt.title('Scatter Plot of Data Points by Class')
    st.pyplot()

def main(df):
      
        
        X, y = preprocess_data(df)

        num_instances, num_attributes = X.shape[0], X.shape[1]
        st.subheader("Le nombre d'instance:")
        st.write(f" {num_instances}")
        st.subheader("Le nombre d'attribut:")
        st.write(f" {num_attributes}")

        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_naive_bayes(X_train, y_train)

        accuracy = evaluate_model(model, X_test, y_test)

        y_prob = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob)
        

     

        

if __name__ == "__main__":
    main()
