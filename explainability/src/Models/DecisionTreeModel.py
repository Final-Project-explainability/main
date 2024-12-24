import json
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from explainability.src.Models.Model import Model


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
        Explains a prediction for a DecisionTreeClassifier by visualizing the decision path.

        Args:
            X_train: A pandas DataFrame representing the training data (for context).
            X_instance: A pandas DataFrame row representing the instance to explain.
            predicted_probability: The predicted probability for the given instance.

        Returns:
            None. Displays the decision path explanation.
        """
        try:
            # Ensure X_instance is a sparse matrix or convert it
            if not isinstance(X_instance, (np.ndarray, scipy.sparse.spmatrix)):
                X_instance = X_instance.to_numpy()

            # Compute the decision path
            decision_path = self.model.decision_path(X_instance)
            print(f"Decision Path indices: {decision_path.indices}")
            print(f"Decision Path indptr: {decision_path.indptr}")

            # Visualization of the decision path (Placeholder for actual visualization logic)
            plt.figure(figsize=(10, 6))
            plt.title(f"Decision Path Visualization\nPredicted Probability: {predicted_probability:.4f}")
            plt.xlabel("Node Index")
            plt.ylabel("Decision Path Depth")
            plt.plot(decision_path.indices, label="Decision Path", marker='o', linestyle='-')
            plt.legend()
            plt.tight_layout()
            plt.show()

        except AttributeError as e:
            print("Error: Ensure the model is a DecisionTreeClassifier or compatible model.")
            print(f"Details: {e}")
        except Exception as e:
            print("An unexpected error occurred.")
            print(f"Details: {e}")
    #
    # def local_explain(self, X_train, X_instance, predicted_probability):
    #     """
    #         Explains a prediction for a DecisionTreeClassifier by visualizing the decision path.
    #
    #         Args:
    #             X_instance: A pandas DataFrame row representing the instance to explain.
    #
    #         Returns:
    #             None. Displays the decision path explanation.
    #         """
    #     decision_path = self.model.decision_path(X_instance)
    #     print(f"Decision Path for the instance: {decision_path}")
    #     # Visualize decision path using matplotlib or other visualization libraries
    #     plt.figure(figsize=(10, 6))
    #     plt.title(f"Decision Path for the Instance")
    #     # Visualization code for decision path can be added here.
    #
    #     plt.tight_layout(rect=(0, 0, 0.9, 1))  # Adjust the layout to prevent overlap with text
    #     plt.show()

    def global_explain(self, X_train ,y_train):
        """
            Display and plot the feature importances from a trained Decision Tree model.
            Args:
                X_train: The dataset used for the model's training.
            """
        feature_names = X_train.columns

        # Extract feature importances
        importances = self.model.feature_importances_

        # Combine feature names and their importances into a DataFrame
        feature_importances = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        # Sort features by importance in descending order
        feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

        # Select the top 20 features
        top_20_features = feature_importances.head(20)

        # Plot the feature importances as a horizontal bar chart
        plt.figure(figsize=(8, 9.5))  # Adjust this ratio as needed

        # Create horizontal bar plot with reversed order so the most important feature is at the top
        plt.barh(top_20_features['Feature'][::-1], top_20_features['Importance'][::-1], color='skyblue')

        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')

        # Ensure labels are horizontal for better readability
        plt.yticks(rotation=0)
        plt.tight_layout()  # Ensures the labels and titles fit in the plot

        plt.show()

        return top_20_features