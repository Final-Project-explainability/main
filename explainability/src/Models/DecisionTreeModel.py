import json
import os
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import graphviz

from explainability.src.Models.Model import Model

from sklearn.tree import _tree
class DecisionTreeModel(Model):


    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train):
        """
            Train a single Decision Tree classifier with hyperparameter tuning and save the best parameters.
            Args:
                X_train: Training feature set.
                y_train: Training labels.
            Returns:
                model: Trained Decision Tree model.
            """
        print("Training a single Decision Tree with hyperparameter tuning...")

        params_path = "../data/jsons/decision_tree_params.json"

        # Check if we have pre-saved parameters
        if os.path.exists(params_path):
            with open(params_path, 'r') as file:
                best_params = json.load(file)
            print(f"Loaded best parameters from {params_path}: {best_params}")
            model = DecisionTreeClassifier(**best_params, random_state=42)
            model.fit(X_train, y_train)
        else:
            # Define hyperparameter grid
            param_grid = {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            # Perform grid search
            grid_search = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid,
                scoring='roc_auc',
                cv=5,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Get the best parameters and model
            best_params = grid_search.best_params_
            print(f"Best Parameters: {best_params}")

            # Save the best parameters to a JSON file
            os.makedirs(os.path.dirname(params_path), exist_ok=True)  # Create directory if it doesn't exist
            with open(params_path, 'w') as f:
                json.dump(best_params, f)
            print(f"Saved best parameters to {params_path}")

            model = grid_search.best_estimator_

        self.model = model
        self.set_name()

        return model

    def local_explain(self, X_train, X_instance, predicted_probability):
        """
        Explains a prediction for a DecisionTreeClassifier by analyzing feature contributions.

        Args:
            X_train: A pandas DataFrame representing the training data (for context).
            X_instance: A pandas DataFrame row representing the instance to explain.
            predicted_probability: The predicted probability for the given instance.

        Returns:
            None. Displays the feature contributions.
        """

        self.visualize_decision_tree(feature_names=X_train.columns)

        try:
            # Ensure X_instance is a 2D array with valid feature names
            if hasattr(X_train, 'columns') and not hasattr(X_instance, 'columns'):
                X_instance = X_train.iloc[[X_instance.name]]
            elif not isinstance(X_instance, (np.ndarray, list)):
                X_instance = X_instance.to_numpy().reshape(1, -1)

            # Access the decision tree structure
            tree = self.model.tree_

            # Extract feature importances along the decision path
            feature_contributions = np.zeros(X_train.shape[1])
            node_indicator = self.model.decision_path(X_instance)
            leaf_id = self.model.apply(X_instance)[0]

            for node_index in node_indicator.indices:
                if node_index == leaf_id:
                    break
                feature = tree.feature[node_index]
                if feature != _tree.TREE_UNDEFINED:
                    threshold = tree.threshold[node_index]
                    value = X_instance.iloc[0, feature] if hasattr(X_instance, 'iloc') else X_instance[0, feature]
                    contribution = (value - threshold) if value <= threshold else (threshold - value)
                    feature_contributions[feature] += contribution

            # Normalize contributions
            total_contribution = np.sum(feature_contributions)
            feature_contributions /= total_contribution

            # Select top 10 features
            top_features_indices = np.argsort(np.abs(feature_contributions))[-10:][::-1]
            top_contributions = feature_contributions[top_features_indices]
            top_feature_names = (
                X_train.columns[top_features_indices]
                if hasattr(X_train, 'columns') else [f'Feature {i}' for i in top_features_indices]
            )

            # Plot the contributions in a horizontal bar chart with values on bars
            plt.figure(figsize=(12, 8))
            colors = ['green' if contrib > 0 else 'red' for contrib in top_contributions]
            bars = plt.barh(range(len(top_contributions)), top_contributions, color=colors, align='center')
            plt.yticks(range(len(top_contributions)), top_feature_names)
            plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xlabel("Contribution to Decision")
            plt.title(f"Top 10 Feature Contributions\nPredicted Probability: {predicted_probability:.4f}")

            # Add text labels to each bar
            for bar, contrib in zip(bars, top_contributions):
                plt.text(
                    bar.get_width() + (0.02 if contrib > 0 else -0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f'{contrib:.3f}',
                    va='center',
                    ha='left' if contrib > 0 else 'right',
                    color='black'
                )

            plt.tight_layout()
            plt.show()

        except AttributeError as e:
            print("Error: Ensure the model is a DecisionTreeClassifier or compatible model.")
            print(f"Details: {e}")
        except Exception as e:
            print("An unexpected error occurred.")
            print(f"Details: {e}")

        self.local_explain2(X_train=X_train, X_instance=X_instance, predicted_probability=predicted_probability)

    def local_explain2(self, X_train, X_instance, predicted_probability):
        X_instance = pd.DataFrame(X_instance, columns=X_train.columns)
        # Get the decision path and leaf node
        decision_path = self.model.decision_path(X_instance)
        leaf_id = self.model.apply(X_instance)[0]

        # Access the tree structure
        tree = self.model.tree_
        feature = tree.feature
        threshold = tree.threshold
        impurity = tree.impurity

        # Initialize contributions dictionary
        feature_contributions = {}

        # Iterate over the decision path
        node_indicator = decision_path.indices
        for node_id in node_indicator[:-1]:  # Skip the last node (leaf)
            # Get the feature and threshold for the current node
            split_feature = feature[node_id]
            if split_feature == -2:  # Skip if the node is not splitting on a feature
                continue

            split_threshold = threshold[node_id]
            feature_name = X_train.columns[split_feature]

            # Calculate the impurity decrease
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            # Determine which child the instance went to
            if X_instance.iloc[0, split_feature] <= split_threshold:
                child_node = left_child
            else:
                child_node = right_child

            # Impurity decrease
            impurity_decrease = impurity[node_id] - impurity[child_node]

            # Accumulate contributions
            if feature_name not in feature_contributions:
                feature_contributions[feature_name] = 0
            feature_contributions[feature_name] += impurity_decrease

        # Normalize contributions to sum to the prediction probability
        total_contribution = sum(feature_contributions.values())
        normalized_contributions = {k: (v / total_contribution) * predicted_probability for k, v in
                                    feature_contributions.items()}

        # Create a DataFrame for better visualization
        explanation_df = pd.DataFrame(list(normalized_contributions.items()), columns=["Feature", "Contribution"])
        explanation_df["Absolute Contribution"] = explanation_df["Contribution"].abs()

        # Sort by absolute contribution and select top 10 features
        explanation_df = explanation_df.sort_values(by="Absolute Contribution", ascending=False)
        top_10_features = explanation_df.head(10).sort_values(by="Contribution")

        # Calculate the sum of the remaining features' contributions
        other_features_contribution = explanation_df.iloc[10:]["Contribution"].sum()

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
        colors = ['#1f77b4' if x > 0 else '#d62728' for x in explanation_df["Contribution"]]

        # Plot horizontal bars
        plt.barh(explanation_df["Feature"], explanation_df["Contribution"], color=colors)

        # Add a vertical line for the base value (expected value)
        base_value = predicted_probability - sum(explanation_df["Contribution"])
        plt.axvline(x=base_value, color='gray', linestyle='--', label=f"Base Value: {base_value:.4f}")

        # Add a vertical line for the predicted probability
        plt.axvline(x=predicted_probability, color='green', linestyle='-',
                    label=f"Predicted Probability: {predicted_probability:.4f}")

        # Add labels, title, and legend
        plt.xlabel("Contribution to Prediction")
        plt.title("Feature Contributions Using Decision Tree")
        plt.legend()

        # Annotate each bar with its contribution value
        for i, v in enumerate(explanation_df["Contribution"]):
            plt.text(v, i, f"{v:.2f}", va='center', ha='left' if v > 0 else 'right', color='black')

        # Adjust layout for better display
        plt.tight_layout()
        plt.show()

        return explanation_df

    def visualize_decision_tree(self, feature_names, class_names=None, save_as_png=False):
        """
        Visualize a Decision Tree using Graphviz or Matplotlib.

        Args:
            model: Trained DecisionTreeClassifier model.
            feature_names: List of feature names.
            class_names: List of class names (optional).
            save_as_png: If True, save the tree as a PNG file.
        """
        # Using sklearn's built-in plot_tree (Matplotlib)
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title("Decision Tree Visualization")
        plt.show()

        # Optionally save the tree as PNG using Graphviz
        if save_as_png:
            dot_data = export_graphviz(
                self.model,
                out_file=None,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                special_characters=True
            )
            graph = graphviz.Source(dot_data)
            graph.format = "png"
            graph.render("decision_tree")
            print("Decision Tree saved as 'decision_tree.png'")

    def global_explain(self, X_train, y_train):
        """
        Display and plot the feature importances from a trained Decision Tree model,
        and save feature importance in a sorted JSON file.

        Args:
            X_train: The dataset used for the model's training.
        Returns:
            feature_importances: The sorted feature importances for all features.
        """
        feature_names = X_train.columns

        # Extract feature importances
        importances = self.model.feature_importances_

        # Combine feature names and their importances into a DataFrame
        feature_importances = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        # Compute relative importance as percentages
        feature_importances['Importance (%)'] = (feature_importances['Importance'] / feature_importances[
            'Importance'].sum()) * 100

        # Sort features by importance in descending order
        feature_importances = feature_importances.sort_values(by="Importance (%)", ascending=False)

        # Save all feature importances to JSON
        feature_importance_dict = {
            row['Feature']: float(row['Importance (%)'])
            for _, row in feature_importances.iterrows()
        }

        # with open("decision_tree_feature_importance.json", "w") as f:
        #     json.dump(feature_importance_dict, f, indent=4)

        print("Feature importance saved to 'decision_tree_feature_importance.json'.")

        # Select the top 20 features for visualization
        top_20_features = feature_importances.head(20)

        # Plot the feature importances as a horizontal bar chart
        plt.figure(figsize=(8, 9.5))  # Adjust this ratio as needed

        # Create horizontal bar plot with reversed order so the most important feature is at the top
        plt.barh(top_20_features['Feature'][::-1], top_20_features['Importance (%)'][::-1], color='skyblue')

        plt.xlabel('Importance (%)')
        plt.title('Top 20 Feature Importances')

        # Ensure labels are horizontal for better readability
        plt.yticks(rotation=0)
        plt.tight_layout()  # Ensures the labels and titles fit in the plot

        plt.show()

        return feature_importances