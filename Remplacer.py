import pandas as pd
from sklearn.impute import SimpleImputer

def remplacer_valeurs_manquantes_csv(df):
    # Charger le fichier CSV
    #df = pd.read_csv(input_file_path)

    # Séparer les colonnes numériques et catégorielles
    numeriques = df.select_dtypes(include='number')
    categorielles = df.select_dtypes(exclude='number')

    # Remplacer les valeurs manquantes numériques par la moyenne de chaque colonne numérique
    if not numeriques.empty:
        imputer_numerique = SimpleImputer(strategy='mean')
        numeriques_remplace = pd.DataFrame(imputer_numerique.fit_transform(numeriques), columns=numeriques.columns)
    else:
        numeriques_remplace = pd.DataFrame()

    # Remplacer les valeurs manquantes catégorielles par la valeur la plus fréquente de chaque colonne catégorielle
    if not categorielles.empty:
        imputer_categorique = SimpleImputer(strategy='most_frequent')
        categorielles_remplace = pd.DataFrame(imputer_categorique.fit_transform(categorielles), columns=categorielles.columns)
    else:
        categorielles_remplace = pd.DataFrame()

    # Concaténer les DataFrames numériques et catégoriels pour obtenir le DataFrame final
    df_remplace = pd.concat([numeriques_remplace, categorielles_remplace], axis=1)

    # Retourner le DataFrame mis à jour
    return df_remplace


