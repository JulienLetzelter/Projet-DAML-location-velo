"""
Modèle Random Forest pour la prédiction du nombre de locations de vélos
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


class RandomForestModel:
    """
    Classe pour le modèle Random Forest
    """
    
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1, **kwargs):
        """
        Initialise le modèle Random Forest
        
        Args:
            n_estimators (int): Nombre d'arbres
            random_state (int): Seed pour la reproductibilité
            n_jobs (int): Nombre de jobs parallèles
            **kwargs: Autres paramètres pour RandomForestRegressor
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Paramètres par défaut
        default_params = {
            'n_estimators': n_estimators,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
        # Fusionner avec les paramètres fournis
        params = {**default_params, **kwargs}
        
        self.model = RandomForestRegressor(**params)
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        """
        Entraîne le modèle
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
    
    
    def cross_validate(self, X, y, cv=5):
        """
        Effectue une cross-validation
        
        Args:
            X: Features
            y: Target
            cv: Nombre de folds
            
        Returns:
            dict: Résultats de la cross-validation
        """
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error')
        
        rmse_cv = np.sqrt(-cv_scores.mean())
        rmse_std = np.sqrt(cv_scores.std())
        
        # Analyse de la stabilité
        if rmse_std < 100:
            stability = "stable"
        elif rmse_std < 200:
            stability = "modérément stable"
        else:
            stability = "instable"
        
        return {
            'rmse_mean': rmse_cv,
            'rmse_std': rmse_std,
            'scores': cv_scores,
            'stability': stability
        }
    
    def get_feature_importance(self, feature_columns):
        """
        Retourne l'importance des features
        
        Args:
            feature_columns: Liste des noms des features
            
        Returns:
            DataFrame: Importance des features
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant d'obtenir l'importance des features")
        
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        
        return feature_importance
    
    def get_model_info(self):
        """
        Retourne les informations sur le modèle
        
        Returns:
            dict: Informations du modèle
        """
        return {
            'model_type': 'Random Forest',
            'parameters': {
                'n_estimators': self.n_estimators,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'max_depth': getattr(self.model, 'max_depth', None),
                'min_samples_split': getattr(self.model, 'min_samples_split', 2),
                'min_samples_leaf': getattr(self.model, 'min_samples_leaf', 1)
            },
            'is_fitted': self.is_fitted,
            'n_features': len(self.model.feature_importances_) if self.is_fitted else None
        }


