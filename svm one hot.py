import numpy as np
import pandas as pd
from scipy.io import arff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load ARFF file
# Replace 'your_dataset.arff' with the actual path to your ARFF file
data, _ = arff.loadarff('breast-cancer.arff')
df = pd.DataFrame(data)

# Dataset information
print("Dataset information:")
print(df.info())

# Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Class distribution
print("\nClass distribution:")
print(df[df.columns[-1]].value_counts())

# Print column names and descriptions
attributes = df.columns[:-1]  # Exclude the last column (target variable)
print("Column Names and Descriptions:")
for attr in attributes:
    attr_type = df[attr].dtype
    print(f"{attr} ({attr_type})")

# Print class names and their counts
class_counts = df.iloc[:, -1].value_counts()
print("\nClass Names and Counts:")
print(class_counts)

# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=df.columns[-1], data=df)
plt.title('Class Distribution')
plt.show()

# Print missing values information
missing_values_info = df.isnull().sum()
print("\nMissing Values Information:")
print(missing_values_info)

# Extract features (X) and target variable (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(str)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and categorical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Standardize numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Encode categorical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Build the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Train the SVM model
svm_model.fit(X_train_preprocessed, y_train_encoded)

# Predictions on the test set
y_pred = svm_model.predict(X_test_preprocessed)

# Evaluate the SVM model
accuracy = accuracy_score(y_test_encoded, y_pred)
conf_matrix = confusion_matrix(y_test_encoded, y_pred)
class_report = classification_report(y_test_encoded, y_pred)

print(f'\nAccuracy on Test Set: {accuracy}')
print('\nConfusion Matrix:\n', conf_matrix)
print('\nClassification Report:\n', class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
