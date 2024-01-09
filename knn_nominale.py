import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st

# Load the dataset from CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def prepare_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Label encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=X.select_dtypes(include='object').columns, drop_first=True)

    return X, y

def prepare_data(df):
    # Exclude the first row and select features
    X = df.iloc[1:, :-1]
    
    # Select target variable (excluding the first row)
    y = df.iloc[1:, -1]

    # Label encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=X.select_dtypes(include='object').columns, drop_first=True)

    return X, y

# Split data into separate training and test set
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Handle missing values and feature scaling
def scale_features(X_train, X_test):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')  # You can choose 'median' or 'constant' as well
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled

# Train kNN classifier
def train_knn(X_train_scaled, y_train, k=3):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    return knn_classifier

# Predict the test-set results
def predict(knn_classifier, X_test_scaled):
    y_pred = knn_classifier.predict(X_test_scaled)
    return y_pred

# Check the accuracy score
def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAccuracy Score: {accuracy:.4f}')

# Rebuild kNN classification model using different values of k

def rebuild_knn(X_train_scaled, y_train, X_test_scaled, y_test, max_k):
    k_values = range(1, max_k + 1)
    accuracy_scores = []

    for k in k_values:
        knn_classifier = train_knn(X_train_scaled, y_train, k)
        y_pred = predict(knn_classifier, X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

    # Plot accuracy vs k
    plt.plot(k_values, accuracy_scores, marker='o')
    plt.title('Accuracy vs k for kNN')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.show()
    st.pyplot()


# Confusion matrix and classification metrics
def classification_metrics(y_test, y_pred):
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    st.subheader('Matrice de Confusion:')
    st.write(confusion_mat)

    st.subheader('Rapport de Classification:')
    st.table(pd.DataFrame(classification_rep).transpose())

# ROC - AUC
def roc_auc_curve(y_test, y_pred_proba):
    classes = np.unique(y_test)
    n_classes = len(classes)

    # Initialize aggregate arrays
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = 0

    plt.figure(figsize=(8, 6))

    for cls in classes:
        cls_idx = (y_test == cls).astype(int)
        fpr, tpr, _ = roc_curve(cls_idx, y_pred_proba[:, cls])
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= n_classes

    # Compute aggregate AUC
    mean_auc = roc_auc_score((y_test == classes[-1]).astype(int), y_pred_proba[:, -1])

    plt.plot(all_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.2f})', linewidth=2)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.title('Mean ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # Intégration avec Streamlit
    st.pyplot()

# Main function to choose the best k and display kNN plots
def main(df,k):
    # Load data (replace 'your_dataset.csv' with the path to your CSV file)
    #
    #df = load_data(file_path)

    # Exploratory Data Analysis
      

    # Prepare data
    X, y = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Handle missing values and feature scaling
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    knn_classifier = train_knn(X_train_scaled, y_train, k)
    y_pred = predict(knn_classifier, X_test_scaled)
    accuracy1 = accuracy_score(y_test, y_pred)
    st.subheader("L'accuracy:")
    st.write(f'Pour k={k}, Précision: {accuracy1:.2%}')

    # Search for the best k value
    best_k = None
    best_accuracy = 0.0
     # Train kNN classifier with the specified k
    #knn_classifier = train_knn(X_train_scaled, y_train, k)
    #y_pred = predict(knn_classifier, X_test_scaled)
    #accuracy = accuracy_score(y_test, y_pred)

    #st.write(f'Pour k={k}, Précision: {accuracy:.4f}')
    #max_k = min(len(X_train), len(X_test)) // 2
    for k in range(1,20):  # Search over k values from 1 to 10
        knn_classifier = train_knn(X_train_scaled, y_train,k )
        y_pred = predict(knn_classifier, X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    st.subheader("meillieur K pour la meillieur précision")
    st.write(f'\nMeilleure valeur de k: {best_k} avec Précision: {best_accuracy:.4f}')

    # Train kNN classifier with the best k
    best_knn_classifier = train_knn(X_train_scaled, y_train, best_k)
   

    # Predict on test set
    y_pred = predict(best_knn_classifier, X_test_scaled)

    # Evaluate model
    evaluate(y_test, y_pred)

    # Rebuild kNN with different values of k
    rebuild_knn(X_train_scaled, y_train, X_test_scaled, y_test,max_k=20)

    # Classification metrics
    classification_metrics(y_test, y_pred)

    # ROC - AUC
   
    st.subheader("La courbe de ROC :")
    y_pred_proba = best_knn_classifier.predict_proba(X_test_scaled)
    roc_auc_curve(y_test, y_pred_proba)

if __name__ == '__main__':
    main()
