"""
Fonctions utilitaires communes pour les modèles de machine learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(data_path='data/day.csv'):
    """
    Charge et préprocesse les données du dataset bike sharing
    
    Args:
        data_path (str): Chemin vers le fichier de données
        
    Returns:
        tuple: (X, y_casual, y_registered, y_total, feature_columns)
    """
    # Chargement des données
    df = pd.read_csv(data_path)
    
    # Création d'une copie pour le preprocessing
    df_processed = df.copy()
    
    # Conversion de la date
    df_processed['dteday'] = pd.to_datetime(df_processed['dteday'])
    
    # Feature engineering : variables cycliques pour capturer la saisonnalité
    df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['mnth'] / 12)
    df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['mnth'] / 12)
    df_processed['weekday_sin'] = np.sin(2 * np.pi * df_processed['weekday'] / 7)
    df_processed['weekday_cos'] = np.cos(2 * np.pi * df_processed['weekday'] / 7)
    
    # Encodage des variables catégorielles
    le_season = LabelEncoder()
    le_weather = LabelEncoder()
    
    df_processed['season_encoded'] = le_season.fit_transform(df_processed['season'])
    df_processed['weather_encoded'] = le_weather.fit_transform(df_processed['weathersit'])
    
    # Sélection des features pour la modélisation
    # atemp retiré car redondant avec temp
    feature_columns = [
        'temp', 'hum', 'windspeed',
        'season_encoded', 'weather_encoded',
        'holiday', 'yr',
        'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
    ]
    
    target_columns = ['casual', 'registered', 'cnt']
    
    X = df_processed[feature_columns]
    y_casual = df_processed['casual']
    y_registered = df_processed['registered']
    y_total = df_processed['cnt']
    
    return X, y_casual, y_registered, y_total, feature_columns




def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Évalue un modèle et retourne les métriques
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        model_name: Nom du modèle pour l'affichage
        
    Returns:
        dict: Dictionnaire contenant les métriques
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    
    return metrics


def compare_models(models_results):
    """
    Compare les modèles et retourne les résultats
    
    Args:
        models_results: Dictionnaire avec les résultats des modèles
        
    Returns:
        dict: Résultats de la comparaison
    """
    # Déterminer le meilleur modèle
    best_model = min(models_results.items(), key=lambda x: x[1]['rmse'])
    
    return {
        'best_model_name': best_model[0],
        'best_model_rmse': best_model[1]['rmse'],
        'models_results': models_results
    }
