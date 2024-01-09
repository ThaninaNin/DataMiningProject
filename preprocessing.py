import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt









#encodage + remplacage des valeurs manquantes par la moyenne + la normalisation min max entre 0 et 1 
def preprocessingMinMax(dataInit):

  #encode value 
  le = LabelEncoder()
  for col in dataInit.columns:
    if dataInit[col].dtype == 'object':
      dataInit[col] = le.fit_transform(dataInit[col].astype(str))
  
  #Remplace missing value with mean value
  dataGap = SimpleImputer(strategy='mean')
  dataMid = pd.DataFrame(dataGap.fit_transform(dataInit), columns=dataInit.columns)
  scaler = MinMaxScaler()
  dataFinal = pd.DataFrame(scaler.fit_transform(dataMid.select_dtypes(include=np.number)), columns=dataMid.select_dtypes(include=np.number).columns)
  return dataMid,dataFinal





