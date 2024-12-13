import json
import os
import joblib
import hashlib


class ModelManager:
    CONFIG_PATH = "jsons/model_configs.json"  # Static path for configuration file

    @staticmethod
    def _load_config():
        """Load the configuration file."""
        if os.path.exists(ModelManager.CONFIG_PATH):
            with open(ModelManager.CONFIG_PATH, 'r') as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save_config(config):
        """Save the configuration to file."""
        with open(ModelManager.CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def _hash_model(model):
        """
        Calculate a hash for the model, ensuring the hash is unique
        for models trained on different features or having different parameters.

        Parameters:
            model: The machine learning model.

        Returns:
            A hash string representing the model.
        """
        # Get model parameters as string
        model_params = str(model.get_params()) if hasattr(model, 'get_params') else str(model)

        # Combine model parameters and feature count
        combined_str = model_params + str(getattr(model, 'n_features_in_', ''))

        # Generate and return the hash
        return hashlib.sha256(combined_str.encode('utf-8')).hexdigest()

    @staticmethod
    def get_path(model):
        model_hash = ModelManager._hash_model(model)

        # Determine paths for saving
        model_type = type(model).__name__
        model_dir = f"models/{model_type}"
        os.makedirs(model_dir, exist_ok=True)
        return f"{model_dir}/{model_type}_{model_hash}"

    @staticmethod
    def save_model_and_shap(model, shap_values):
        """Save the model and its SHAP values."""

        path = ModelManager.get_path(model)
        model_path = path + ".joblib"
        shap_path = path + "_shap.pkl"

        # Save model and SHAP values
        joblib.dump(model, model_path)
        with open(shap_path, 'wb') as f:
            joblib.dump(shap_values, f)

        # Update configuration
        config = ModelManager._load_config()
        model_type = type(model).__name__
        config[model_type] = {
            "latest_model": model_path,
            "latest_shap": shap_path,
        }
        ModelManager._save_config(config)
        print(f"Model and SHAP values saved for {model_type}. Paths updated in config.")

    @staticmethod
    def load_model(model_type):
        """Load the latest saved model of the given type."""
        config = ModelManager._load_config()
        if model_type in config:
            model_path = config[model_type]["latest_model"]
            print(f"loading model from {model_path}")
            return joblib.load(model_path)
        raise ValueError(f"No model of type {model_type} found in configuration.")

    @staticmethod
    def load_shap(model):
        """Load SHAP values for a given model."""
        shap_path = ModelManager.get_path(model) + "_shap.pkl"
        with open(shap_path, 'rb') as f:
            return joblib.load(f)

    @staticmethod
    def save_fbt(model, fbt):
        fbt_path = ModelManager.get_path(model) + "_fbt.joblib"
        joblib.dump(fbt, fbt_path)
        print("FBT model saved as 'fbt_model.joblib'.")

    @staticmethod
    def load_fbt(model):
        fbt_path = ModelManager.get_path(model) + "_fbt.joblib"
        return joblib.load(fbt_path)
