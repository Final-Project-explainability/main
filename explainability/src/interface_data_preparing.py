import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from explainability.src.ModelManager import ModelManager
from preprocessing import preprocess_data, feature_engineering
from sklearn.model_selection import train_test_split


# Load the data
file_path = "example_test_data.csv"
data = pd.read_csv(file_path)
print(f"Data loaded successfully from {file_path}")

# Separate features (X), target (y), and IDs
X_sample = data.drop(columns=['hospital_death', 'patient_id'])
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

    Returns:
        X_normalized: The normalized training set (pandas DataFrame).
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform the train set
    X_normalized = scaler.fit_transform(X)

    # Return as DataFrame with the same column names
    return pd.DataFrame(X_normalized, columns=X.columns)


normalized_X = normalize_data(X_sample)


# # Prepare an empty DataFrame to store results
# results = pd.DataFrame()
# results['patient_id'] = ids

# # Predict probabilities for each model
# results['DecisionTree_Pred'] = [decisionTreeModel.predict_proba(normalized_X.iloc[[i]])[:, 1][0] for i in range(len(X))]
# results['LogisticRegression_Pred'] = [logisticRegressionModel.predict_proba(normalized_X.iloc[[i]])[:, 1][0] for i in range(len(X))]
# results['XGBoost_Pred'] = [xgboostModel.predict_proba(X.iloc[[i]])[:, 1][0] for i in range(len(X))]

# # Save results to a CSV file
# output_file = "model_predictions.csv"
# results.to_csv(output_file, index=False)
# print(f"Predictions saved to {output_file}")

def get_model_data():
    """
    Main function to execute the model management workflow.
    """
    # Load and preprocess the dataset
    data = load_data()
    if data is None:
        print("Error loading the dataset")
        return

    print("Column names in the dataset:", data.columns)

    # Feature engineering and preprocessing
    data = feature_engineering(data)
    data = preprocess_data(data)

    X = data.drop(columns=['hospital_death'])
    y = data['hospital_death']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_model_data()
models = [xgboostModel, logisticRegressionModel, decisionTreeModel]

for i in range(len(X_sample)):
    for model in models:
        if model.get_type() == "XGBClassifier":
            X_sample_for_prediction = X_sample
        else:
            X_sample_for_prediction = normalized_X

        individual_data = X_sample_for_prediction.iloc[[i]]
        model.backend_local_shap(individual_data)
        model.backend_local_lime(X_sample_for_prediction, individual_data)





