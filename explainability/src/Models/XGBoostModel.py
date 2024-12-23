import json
import os
import tempfile
import webbrowser
import lime
import lime.lime_tabular
import optuna
import shap
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from explainability.src.ModelManager import ModelManager
from explainability.src.Models.Model import Model
import pandas as pd
from XGBoostTreeApproximator.FBT import FBT


class XGBoostModel(Model):

    def __init__(self):
        super().__init__()
        self.model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

    def train(self, X_train, y_train, tune_hyperparameter=False):
        """
            Train an XGBoost model with optional hyperparameter tuning, feature selection, and long-run optimization.
            Args:
                X_train: Training feature set.
                y_train: Training labels.
                tune_hyperparameter: Whether to perform long-running optimization using Optuna.
            Returns:
                Trained XGBoost model.
            """
        print("Training XGBoost model...")
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

        # Split training data into training and validation sets
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        params_path = "../data/jsons/best_params.json"

        if tune_hyperparameter:
            print("Starting long-run optimization with Optuna...")

            def hyperparameter_tuning(trial):
                param = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': trial.suggest_int('max_depth', 4, 5),  # Focusing on a narrow range for depth
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01, log=True),
                    # Very low learning rate
                    'n_estimators': trial.suggest_int('n_estimators', 5000, 20000),  # Allowing for very large trees
                    'min_child_weight': trial.suggest_int('min_child_weight', 4, 6),
                    'gamma': trial.suggest_float('gamma', 1e-8, 0.01, log=True),  # Smaller gamma values for fine-tuning
                    'subsample': trial.suggest_float('subsample', 0.6, 0.8),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                    'scale_pos_weight': scale_pos_weight  # Balancing class imbalance
                }

                model = xgb.XGBClassifier(**param)
                cv_scores = cross_val_score(model, X_train_split, y_train_split, scoring='roc_auc', cv=3)
                return cv_scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(hyperparameter_tuning, n_trials=50)

            best_params = study.best_params
            best_params['objective'] = 'binary:logistic'
            best_params['eval_metric'] = 'logloss'
            best_params['scale_pos_weight'] = scale_pos_weight

            print("Best parameters found: ", best_params)

            # Save the best parameters to a JSON file
            with open(params_path, 'w') as f:
                json.dump(best_params, f)
            print(f"Optimized parameters saved to {params_path}")

        else:
            # Load previous best parameters if available
            if os.path.exists(params_path):
                with open(params_path, 'r') as file:
                    best_params = json.load(file)
                print("Loaded previous best parameters: ", best_params)
            else:
                best_params = {
                    'max_depth': 4,
                    'learning_rate': 0.048382856372731146,
                    'n_estimators': 477,
                    'min_child_weight': 6,
                    'gamma': 0.0001605071172415074,
                    'subsample': 0.7501140945156216,
                    'colsample_bytree': 0.7481462648709857,
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'scale_pos_weight': scale_pos_weight
                }
                print("Using default parameters: ", best_params)

        model = xgb.XGBClassifier(**best_params)

        model.fit(X_train, y_train)

        self.model = model
        self.set_name()

        return model

    def local_explain(self, X_train, X_instance, predicted_probability):
        self.local_explain_with_shap(X_instance, predicted_probability)
        self.explain_with_lime(X_train, X_instance)

    def global_explain(self, X_train,y_train):
        self.global_explain_with_shap(X_train=X_train)
        self.train_and_visualize_fbt(X_train = X_train, y_train=y_train, xgb_model= self)

    def train_and_visualize_fbt(self,X_train,y_train, xgb_model, max_depth=5, min_forest_size=10,
                                max_number_of_conjunctions=100, pruning_method='auc'):

        try:
            fbt = ModelManager.load_fbt(xgb_model)
            print(" ss fbt")
        except:  # train fbt base on the given model
            # Combine X_train and y_train into a single DataFrame
            train_data = X_train.copy()
            train_data['hospital_death'] = y_train  # Add the label column

            # Extract feature names
            feature_cols = X_train.columns.tolist()
            label_col = 'hospital_death'  # Name of the label column

            # Prepare FBT
            fbt = FBT(max_depth=max_depth,
                      min_forest_size=min_forest_size,
                      max_number_of_conjunctions=max_number_of_conjunctions,
                      pruning_method=pruning_method)

            X_train_sample = train_data.sample(frac=0.1, random_state=42)

            # Fit the FBT model to the training data
            fbt.fit(X_train_sample, feature_cols, label_col, xgb_model)

            print("FBT model trained successfully.")

            ModelManager.save_fbt(xgb_model, fbt)

        # Visualize the tree (example code for visualization)
        try:
            print("Generating visualization...")
            train_data = X_train.copy()
            train_data['hospital_death'] = y_train  # Add the label column
            X_train_sample = train_data.sample(frac=0.05, random_state=42)
            # print(fbt.get_decision_paths(X_train_sample))
            paths = fbt.get_decision_paths(X_train_sample)
            for i, path in enumerate(paths):
                print(f" path  {i + 1}:")
                for step in path:
                    print(f"  {step}")

        except Exception as e:
            print(f"Failed to generate visualization: {e}")

        return fbt

    def global_explain_with_shap(self, X_train):
        """
            Explain the model with SHAP values, using pre-saved values if available.
            If SHAP values are not found, compute them and save both model and SHAP.
            Args:
                model: The trained model object.
                X_train: The dataset used for SHAP computation.
            Returns:
                shap_values: The SHAP values for the model.
            """
        try:
            # Attempt to load SHAP values if they exist
            shap_values = ModelManager.load_shap(self)
            print("Loaded existing SHAP values.")
        except:
            explainer = shap.TreeExplainer(self.model)
            print("Using TreeExplainer for tree-based model.")
            shap_values = explainer.shap_values(X_train)
            # Save the SHAP values and the model type
            ModelManager.save_shap(self, shap_values)

        shap.summary_plot(shap_values, X_train)

        return shap_values

    def local_explain_with_shap(self, X_instance, predicted_probability):
        """
        Explains a prediction using SHAP for complex models like XGBoost, LightGBM, and Random Forest.
        Displays the 10 most important features and the sum of the remaining features in descending order of importance.

        Args:
            model: The trained model.
            X_instance: A pandas DataFrame row representing the instance to explain.
            predicted_probability: The predicted probability from the model.

        Returns:
            None. Displays the SHAP explanation.
        """
        # Create a SHAP explainer object
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_instance)

        # Extract the base value (expected value of the model predictions)
        base_value = explainer.expected_value

        # Extract SHAP values for class 1 (assuming binary classification)
        shap_values_class_1 = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Combine feature names and SHAP values into a DataFrame
        feature_contributions = pd.DataFrame({
            "Feature": X_instance.columns,
            "Contribution": shap_values_class_1[0],
            "Absolute Contribution": np.abs(shap_values_class_1[0])
        })

        # Sort features by absolute contribution to highlight the most impactful ones
        feature_contributions = feature_contributions.sort_values(by="Absolute Contribution", ascending=False)

        # Select the top 10 most important features
        top_10_features = feature_contributions.head(10)
        top_10_features = top_10_features.sort_values(by="Absolute Contribution")

        # Calculate the sum of the remaining features' contributions
        other_features_contribution = feature_contributions.iloc[10:]["Contribution"].sum()

        # Add a row representing the sum of the other features
        other_features_row = pd.DataFrame({
            "Feature": ["Other Features Contribution"],
            "Contribution": [other_features_contribution],
            "Absolute Contribution": [np.abs(other_features_contribution)]
        })

        # Combine top 10 features and the other features' contribution into a single DataFrame
        explanation_df = pd.concat([other_features_row, top_10_features], ignore_index=True)

        # Create a horizontal bar chart to visualize positive and negative contributions
        plt.figure(figsize=(10, 7))

        # Set colors: blue for positive contributions, red for negative contributions
        colors = ['#1f77b4' if x < 0 else '#d62728' for x in explanation_df["Contribution"]]

        # Plot horizontal bars
        plt.barh(explanation_df["Feature"], explanation_df["Contribution"], color=colors)

        # Add a vertical line for the base value (expected value)
        plt.axvline(x=base_value, color='gray', linestyle='--', label=f"Base Value: {base_value:.4f}")

        # Add a vertical line for the predicted probability
        plt.axvline(x=predicted_probability, color='green', linestyle='-',
                    label=f"Predicted Probability: {predicted_probability:.4f}")

        # Add labels, title, and legend
        plt.xlabel("Contribution to Prediction")
        plt.title("Feature Contributions Using SHAP")
        plt.legend()

        # Annotate each bar with its contribution value
        for i, v in enumerate(explanation_df["Contribution"]):
            plt.text(v, i, f"{v:.2f}", va='center', ha='left' if v > 0 else 'right', color='black')

        # Adjust layout for better display
        plt.tight_layout()
        plt.show()

    def explain_with_lime(self, X_train, X_instance, save_html_path=None):
        """
        Explains a prediction for an XGBoost or LGBM model using LIME.
        Always saves and displays the explanation as an HTML file and an image (bar plot).

        Args:
            model: The trained model (e.g., XGBoost or LightGBM).
            X_train: A pandas DataFrame representing the training data.
            X_instance: A pandas DataFrame row representing the instance to explain.
            save_image_path: Optional path to save the explanation figure as an image.
            save_html_path: Optional path to save the explanation as an HTML file.

        Returns:
            None. Saves and displays both HTML and image explanations.
        """
        # Set default class names
        class_names = ['Survive', 'Death']

        # Prepare the feature names
        feature_names = X_train.columns.tolist()

        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,  # Training data
            feature_names=feature_names,  # Feature names
            class_names=class_names,  # Class names for the output
            mode='classification',  # Model type
            discretize_continuous=True  # Discretize continuous features
        )

        # Explain the single instance
        explanation = explainer.explain_instance(
            X_instance.values[0],  # Instance to explain
            self.model.predict_proba  # Prediction function
        )

        # Save explanation as HTML (always)
        if not save_html_path:
            # Generate a temporary HTML file if save path not provided
            html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            save_html_path = html_file.name

        explanation.save_to_file(save_html_path)
        print(f"LIME explanation saved as HTML at: {os.path.abspath(save_html_path)}")

        # Automatically open the HTML in the default browser
        webbrowser.open(f"file://{os.path.abspath(save_html_path)}")

        # Generate and save/display the explanation as a bar plot
        fig = explanation.as_pyplot_figure()
        plt.title("LIME Explanation - Feature Contributions")
        plt.tight_layout()
        plt.show()
