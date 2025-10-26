"""
Modèle de Régression Linéaire Multiple pour la prédiction du nombre de locations de vélos
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


class LinearRegressionModel:
    """
    Classe pour le modèle de régression linéaire multiple
    """
    
    def __init__(self, random_state=42):
        """
        Initialise le modèle de régression linéaire
        
        Args:
            random_state (int): Seed pour la reproductibilité
        """
        self.model = LinearRegression()
        self.random_state = random_state
        self.feature_columns = None
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
        Retourne l'importance des features (coefficients)
        
        Args:
            feature_columns: Liste des noms des features
            
        Returns:
            DataFrame: Importance des features
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant d'obtenir l'importance des features")
        
        coefficients = pd.DataFrame({
            'feature': feature_columns,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        
        return coefficients
    
    def get_model_info(self):
        """
        Retourne les informations sur le modèle
        
        Returns:
            dict: Informations du modèle
        """
        return {
            'model_type': 'Linear Regression',
            'parameters': {
                'fit_intercept': self.model.fit_intercept,
                'normalize': getattr(self.model, 'normalize', False),
                'copy_X': getattr(self.model, 'copy_X', True)
            },
            'is_fitted': self.is_fitted,
            'n_features': len(self.model.coef_) if self.is_fitted else None
        }


