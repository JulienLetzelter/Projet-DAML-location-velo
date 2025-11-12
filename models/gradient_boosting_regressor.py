"""
Modèle Random Forest pour la prédiction du nombre de locations de vélos
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


class GradientBoostingModel:
    """
    Classe pour le modèle Random Forest
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        **kwargs
    ):
        """
        Initialise le modèle Gradient Boosting

        Args:
            n_estimators (int): Nombre d'arbres
            learning_rate (float): Taux d'apprentissage
            max_depth (int): Profondeur maximale d'un arbre
            subsample(int): Fraction d'échantillons utilisés à chaque itération
            random_state (int): Seed pour la reproductibilité
            n_jobs (int): Nombre de jobs parallèles
            **kwargs: Autres paramètres pour RandomForestRegressor
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state

        # Paramètres par défaut
        default_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "random_state": random_state,
        }

        # Fusionner avec les paramètres fournis
        params = {**default_params, **kwargs}

        self.model = GradientBoostingRegressor(**params)
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
        cv_scores = cross_val_score(
            self.model, X, y, cv=cv, scoring="neg_mean_squared_error"
        )

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
            "rmse_mean": rmse_cv,
            "rmse_std": rmse_std,
            "scores": cv_scores,
            "stability": stability,
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
            raise ValueError(
                "Le modèle doit être entraîné avant d'obtenir l'importance des features"
            )

        feature_importance = pd.DataFrame(
            {"feature": feature_columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return feature_importance

    def get_model_info(self):
        """
        Retourne les informations sur le modèle

        Returns:
            dict: Informations du modèle
        """
        return {
            "model_type": "Random Forest",
            "parameters": {
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "subsample": self.subsample,
                "min_samples_split": getattr(self.model, "min_samples_split", 2),
                "min_samples_leaf": getattr(self.model, "min_samples_leaf", 1),
            },
            "is_fitted": self.is_fitted,
            "n_features": (
                len(self.model.feature_importances_) if self.is_fitted else None
            ),
        }
