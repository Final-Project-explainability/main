import xgboost as xgb
import joblib
import pandas as pd
import json
from sklearn.model_selection import cross_val_score


class FeatureFilteredXGB(xgb.XGBClassifier):
    def __init__(self, selected_features=None, **kwargs):
        """
        Custom XGBoost model with feature filtering.
        :param selected_features: List of features to use for training and prediction.
        :param kwargs: Additional parameters for XGBClassifier.
        """
        super().__init__(**kwargs)
        self.selected_features = selected_features

    def fit(self, X, y, **kwargs):
        """
        Fit the model using only the selected features.
        :param X: Training data (pandas DataFrame).
        :param y: Target variable.
        :param kwargs: Additional arguments for XGBClassifier's fit method.
        """
        if self.selected_features is not None:
            X = X[self.selected_features]
        super().fit(X, y, **kwargs)

    def predict_proba(self, X, **kwargs):
        """
        Predict probabilities using only the selected features.
        :param X: Input data (pandas DataFrame).
        :param kwargs: Additional arguments for XGBClassifier's predict_proba method.
        :return: Predicted probabilities.
        """
        if self.selected_features is not None:
            X = X[self.selected_features]
        return super().predict_proba(X, **kwargs)

    def save(self, filepath):
        """
        Save the model to a file.
        :param filepath: Path to save the model.
        """
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        """
        Load the model from a file.
        :param filepath: Path to the saved model.
        :return: Loaded FeatureFilteredXGB object.
        """
        return joblib.load(filepath)
