import pandas as pd
from sklearn.preprocessing import StandardScaler

from explainability.src.Models.DecisionTreeModel import DecisionTreeModel
from explainability.src.Models.LogisticRegressionModel import LogisticRegressionModel
from explainability.src.Models.XGBoostModel import XGBoostModel
from explainability.src.ModelManager import ModelManager

# Load the data
file_path = "example_test_data.csv"
data = pd.read_csv(file_path)
print(f"Data loaded successfully from {file_path}")

# Separate features (X), target (y), and IDs
X = data.drop(columns=['hospital_death', 'patient_id'])
y = data['hospital_death']
ids = data['patient_id']

# Load the models
decisionTreeModel = ModelManager.load_model("DecisionTreeClassifier")
logisticRegressionModel = ModelManager.load_model("LogisticRegression")
xgboostModel = ModelManager.load_model("XGBClassifier")

def normalize_data(X):
    """
    Normalize the dataset using StandardScaler (mean=0, std=1).

    Args:
        X: The training features (pandas DataFrame).
        X_test: The test features (pandas DataFrame).

    Returns:
        X_train_normalized: The normalized training set (pandas DataFrame).
        X_test_normalized: The normalized test set (pandas DataFrame).
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform the train set
    X_train_normalized = scaler.fit_transform(X)

    # Return as DataFrame with the same column names
    return pd.DataFrame(X_train_normalized, columns=X.columns)

normalized_X = normalize_data(X)

# Prepare an empty DataFrame to store results
results = pd.DataFrame()
results['patient_id'] = ids

# # Predict probabilities for each model
# results['DecisionTree_Pred'] = [decisionTreeModel.predict_proba(normalized_X.iloc[[i]])[:, 1][0] for i in range(len(X))]
# results['LogisticRegression_Pred'] = [logisticRegressionModel.predict_proba(normalized_X.iloc[[i]])[:, 1][0] for i in range(len(X))]
# results['XGBoost_Pred'] = [xgboostModel.predict_proba(X.iloc[[i]])[:, 1][0] for i in range(len(X))]

# Save results to a CSV file
output_file = "model_predictions.csv"
results.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
