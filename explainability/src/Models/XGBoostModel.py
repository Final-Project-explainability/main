import json
import os
import tempfile
import webbrowser
from lime.lime_tabular import LimeTabularExplainer
import lime
import lime.lime_tabular
import optuna
import shap
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from explainability.src.ModelManager import ModelManager
from explainability.src.Models.Model import Model
import pandas as pd

from sklearn.model_selection import ParameterGrid
import random


class XGBoostModel(Model):

    def __init__(self):
        super().__init__()
        self.model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

    def global_explain_inherent(self, X_train):
        """
        Compute global feature importance from an XGBoost model using 'gain' as the importance type.

        Args:
            X_train (pd.DataFrame): The training data (only used to extract feature names).

        Returns:
            pd.DataFrame: DataFrame with feature importances (gain), unnormalized.
        """
        booster = self.model.get_booster()
        importance_dict = booster.get_score(importance_type='gain')

        all_features = list(X_train.columns)
        importance_data = [
            {
                "Feature": feature,
                "Contribution": importance_dict.get(feature, 0.0)
            }
            for feature in all_features
        ]

        df = pd.DataFrame(importance_data)
        df = df.sort_values(by="Contribution", ascending=False).reset_index(drop=True)
        return df

    # def backend_inherent(self, X_instance):
    #     """
    #     Calculate the contribution of each feature for a single instance prediction
    #     in an XGBoost model using the inherent Gain metric.
    #
    #     Parameters:
    #         X_instance (DataFrame): The single instance to analyze, shape (1, n_features).
    #
    #     Returns:
    #         DataFrame: A DataFrame with feature names and their Gain contributions.
    #     """
    #     # Ensure the model is trained
    #     if not self.model.get_booster():
    #         raise ValueError("The model must be trained before calling this method.")
    #
    #     # Get feature importances based on Gain
    #     booster = self.model.get_booster()
    #     gain_importances = booster.get_score(importance_type='gain')
    #
    #     # Normalize gain importances
    #     total_gain = sum(gain_importances.values())
    #     normalized_gain = {}
    #     for feature in gain_importances:
    #         normalized_gain[feature] = gain_importances[feature] / total_gain
    #
    #     # Create DataFrame for contributions
    #     contributions_df = pd.DataFrame({
    #         'Feature': list(normalized_gain.keys()),
    #         'Contribution': list(normalized_gain.values())
    #     }).sort_values(by='Contribution', ascending=False).reset_index(drop=True)
    #
    #     return contributions_df

    def backend_inherent(self, X_instance):
        """
        Compute local feature contributions (inherent explanation) for a single instance
        using XGBoost's pred_contribs=True.

        Args:
            X_instance (pd.DataFrame): A single-row DataFrame representing one example.

        Returns:
            pd.DataFrame: DataFrame with each feature's contribution and the bias term.
        """

        # Create DMatrix with feature names
        dmatrix = xgb.DMatrix(X_instance, feature_names=list(X_instance.columns))

        # Get per-feature contributions (including bias)
        contribs = self.model.get_booster().predict(dmatrix, pred_contribs=True)[0]

        # Remove the last entry (bias)
        contribs = contribs[:-1]

        # Prepare output DataFrame
        feature_names = list(X_instance.columns)
        df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contribs
        })

        # # Optional: compute predicted logit and probability
        # logit = np.sum(contribs)
        # probability = 1 / (1 + np.exp(-logit))
        #
        # print(f"Predicted logit: {logit:.4f}")
        # print(f"Predicted probability: {probability:.4f}")

        return df


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

    # def train_and_visualize_fbt(self,X_train,y_train, xgb_model, max_depth=5, min_forest_size=10,
    #                             max_number_of_conjunctions=100, pruning_method='auc'):
    #
    #     try:
    #         fbt = ModelManager.load_fbt(xgb_model)
    #         print(" ss fbt")
    #     except:  # train fbt base on the given model
    #         # Combine X_train and y_train into a single DataFrame
    #         train_data = X_train.copy()
    #         train_data['hospital_death'] = y_train  # Add the label column
    #
    #         # Extract feature names
    #         feature_cols = X_train.columns.tolist()
    #         label_col = 'hospital_death'  # Name of the label column
    #
    #         # Prepare FBT
    #         fbt = FBT(max_depth=max_depth,
    #                   min_forest_size=min_forest_size,
    #                   max_number_of_conjunctions=max_number_of_conjunctions,
    #                   pruning_method=pruning_method)
    #
    #         X_train_sample = train_data.sample(frac=0.1, random_state=42)
    #
    #         # Fit the FBT model to the training data
    #         fbt.fit(X_train_sample, feature_cols, label_col, xgb_model)
    #
    #         print("FBT model trained successfully.")
    #
    #         ModelManager.save_fbt(xgb_model, fbt)
    #
    #     # Visualize the tree (example code for visualization)
    #     try:
    #         print("Generating visualization...")
    #         train_data = X_train.copy()
    #         train_data['hospital_death'] = y_train  # Add the label column
    #         X_train_sample = train_data.sample(frac=0.05, random_state=42)
    #         # print(fbt.get_decision_paths(X_train_sample))
    #         print(fbt.predict_proba(X_train_sample[0]))
    #         print("************")
    #         paths = fbt.get_decision_paths(X_train_sample)
    #         for i, path in enumerate(paths):
    #             print(f" path  {i + 1}:")
    #             for step in path:
    #                 print(f"  {step}")
    #
    #     except Exception as e:
    #         print(f"Failed to generate visualization: {e}")
    #
    #     return fbt

    def global_explain_with_shap(self, X_train):
        """
        Explain the model with SHAP values, using pre-saved values if available.
        If SHAP values are not found, compute them and save both model and SHAP.
        Additionally, save feature importance as percentages in a sorted JSON file.

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

        # Compute mean absolute SHAP values for each feature
        shap_means = np.abs(shap_values).mean(axis=0)

        # Normalize to percentages
        shap_percentages = (shap_means / shap_means.sum()) * 100

        # Convert to a Python float (from np.float32) for JSON compatibility
        shap_percentages = shap_percentages.astype(float)

        # Prepare a dictionary of features and their importance percentages
        feature_importance = {
            feature: percentage
            for feature, percentage in zip(X_train.columns, shap_percentages)
        }

        # Sort the dictionary by values (importance) in descending order
        sorted_feature_importance = dict(
            sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
        )

        # Save to JSON file
        with open("XGBOOST_global_shap.json", "w") as f:
            json.dump(sorted_feature_importance, f, indent=4)

        print("Sorted feature importance saved to 'XGBOOST_global_shap.json'.")

        # Display the SHAP summary plot
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
        return feature_contributions

    def explain_with_lime(self, X_train, X_instance, save_html_path=None):
        """
        Explains a prediction for an XGBoost or LGBM model using LIME.
        Always saves and displays the explanation as an HTML file and an image (bar plot).

        Args:
            model: The trained model (e.g., XGBoost or LightGBM).
            X_train: A pandas DataFrame representing the training data.
            X_instance: A pandas DataFrame row representing the instance to explain.
            save_html_path: Optional path to save the explanation as an HTML file.

        Returns:
            explanation_list: List of feature contributions (weights).
        """
        # Set default class names
        class_names = ['Survive', 'Death']

        # Normalize the training data and the instance using MinMaxScaler
        # scaler = MinMaxScaler()
        # X_train_normalized = scaler.fit_transform(X_train)
        # X_instance_normalized = scaler.transform(X_instance)

        # Prepare the feature names
        feature_names = X_train.columns.tolist()

        # Create LIME explainer with normalized data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,  # training data for LIME
            feature_names=feature_names,  # Feature names
            class_names=class_names,  # Class names for the output
            mode='classification',  # Model type
            discretize_continuous=False,  # Discretize continuous features
            kernel_width=5
        )

        # Explain the single instance (normalized)
        explanation = explainer.explain_instance(
            X_instance.values[0],  # instance to explain
            self.model.predict_proba,  # Prediction function
            num_samples=1000,
            num_features=183  # להציג את כל הפיצ'רים
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

        # Extract the intercept and explanation list
        explanation_list = explanation.as_list()
        intercept = explanation.intercept[1]  # הטיה עבור המחלקה 'Death'
        weights_sum = sum([w[1] for w in explanation_list])  # סכום התרומות של התכונות
        predicted_value = intercept + weights_sum  # חישוב החיזוי לפי המודל המקומי

        print(f"Intercept: {intercept}")
        print(f"Sum of weights: {weights_sum}")
        print(f"Predicted value: {predicted_value}")
        return explanation_list

    def global_explanations_with_lime(self, X_train):
        """
        Aggregate LIME explanations for the entire dataset to create a global summary.
        Save feature importance in a sorted JSON file and visualize the top 20 features.

        Args:
            model: Trained model to explain.
            X_train_sample: Training dataset.
            y_train: Training labels.
        Returns:
            aggregated_importance: A DataFrame with feature importance aggregated across all samples.
        """
        X_train_sample = X_train.sample(frac=0.02, random_state=42)

        # Initialize the LIME explainer
        explainer = LimeTabularExplainer(
            X_train_sample.values,
            feature_names=X_train_sample.columns,
            class_names=['alive', 'dead'],
            verbose=True,
            mode='classification'
        )

        # Initialize a dictionary to store cumulative importance for each unique feature
        cumulative_importance = {}

        # Run LIME for all samples in the dataset
        print("Running LIME on 2% of the dataset...")
        for i in range(X_train_sample.shape[0]):
            exp = explainer.explain_instance(X_train_sample.iloc[i].values, self.model.predict_proba,
                                             num_features=len(X_train_sample.columns))
            explanation_list = exp.as_list()

            # Accumulate the importance values
            for feature, importance in explanation_list:
                if feature not in cumulative_importance:
                    cumulative_importance[feature] = 0
                cumulative_importance[feature] += abs(importance)  # Accumulate absolute importance

        # Normalize the importance values to percentages
        total_importance = sum(cumulative_importance.values())
        importance_percentages = {feature: (importance / total_importance) * 100 for feature, importance in
                                  cumulative_importance.items()}

        # Sort features by importance in descending order
        sorted_importance = dict(sorted(importance_percentages.items(), key=lambda item: item[1], reverse=True))

        # Save the sorted importance values to a JSON file in SHAP-like format
        with open("lime_feature_importance.json", "w") as f:
            json.dump(sorted_importance, f, indent=4)
        print("Feature importance saved to 'lime_feature_importance.json'.")

        # Convert sorted importance to a DataFrame
        importance_df = pd.DataFrame(list(sorted_importance.items()), columns=['Feature', 'Importance (%)'])

        # Select the top 20 features for visualization
        top_20_features = importance_df.head(20)

        # Plot the top 20 features
        plt.figure(figsize=(8, 9.5))  # Adjust size as needed
        plt.barh(top_20_features['Feature'][::-1], top_20_features['Importance (%)'][::-1], color='skyblue')
        plt.xlabel('Importance (%)')
        plt.title('Top 20 Features by LIME')
        plt.tight_layout()  # Ensures labels fit properly
        plt.show()

        return sorted_importance

    def backend_get_name(self):
        return "XGBOOST"


def tune_lime_parameters(model, X_train, num_samples=500):
    """
    Perform parameter tuning for LIME on a subset of the data to maximize R².

    Args:
        model: Trained XGBoost model.
        X_train: Training data as a pandas DataFrame.
        num_samples: Number of samples to evaluate in the experiment.

    Returns:
        best_params: Dictionary of the best parameters.
        results: DataFrame of all parameter combinations and their R² averages.
    """
    # Subset of the data
    random_indices = random.sample(range(len(X_train)), num_samples)
    X_subset = X_train.iloc[random_indices]

    # Define parameter grid
    param_grid = {
        'kernel_width': [3, 5, 10],
        'num_samples': [1000, 5000, 10000],
        'num_features': [10, 20, 30],
        'discretize_continuous': [True, False],
    }
    grid = list(ParameterGrid(param_grid))

    # Initialize results storage
    results = []

    for params in grid:
        r2_scores = []

        # Set up LIME explainer with current parameters
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['Survive', 'Death'],
            mode='classification',
            kernel_width=params['kernel_width'],
            discretize_continuous=params['discretize_continuous']
        )

        # Iterate over the subset of data
        for idx, row in X_subset.iterrows():
            explanation = explainer.explain_instance(
                data_row=row.values,
                predict_fn=model.predict_proba,
                num_features=params['num_features'],
                num_samples=params['num_samples']
            )
            # Collect R² score
            r2_scores.append(explanation.score)

        # Store results for this parameter combination
        avg_r2 = np.mean(r2_scores)
        results.append({'params': params, 'avg_r2': avg_r2})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Find the best parameters
    best_params = results_df.loc[results_df['avg_r2'].idxmax()]['params']
    return best_params, results_df



