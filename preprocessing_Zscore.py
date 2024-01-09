import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt




def preprocessing_Zscore(data):
  
  # Preprocess data
  le = LabelEncoder()
  for col in data.columns:
      if data[col].dtype == 'object':
          data[col] = le.fit_transform(data[col].astype(str))
  imp = SimpleImputer(strategy='mean')
  data = pd.DataFrame(imp.fit_transform(data), columns=data.columns)
  scaler = StandardScaler()
  data2 = data.copy()
  
  data2[data.select_dtypes(include="number").columns] = scaler.fit_transform(data.select_dtypes(include="number"))
  return data2