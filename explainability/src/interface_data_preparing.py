import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from explainability.src.ModelManager import ModelManager
from preprocessing import preprocess_data, feature_engineering
from sklearn.model_selection import train_test_split
import json
import os

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


def normalize_contributions(df):
    """
    Normalize the contributions in a DataFrame so their sum equals 100%.

    Args:
        df (DataFrame): DataFrame with a 'Contribution' column.

    Returns:
        DataFrame: Normalized DataFrame.
    """
    total_contribution = np.abs(df['Contribution']).sum()
    if total_contribution != 0:
        df['Normalized Contribution'] = (df['Contribution'] / total_contribution) * 100
    else:
        df['Normalized Contribution'] = 0
    df = df.drop(columns=['Contribution'])
    return df


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
models = [decisionTreeModel, xgboostModel, logisticRegressionModel]

# Directory to save JSON files
output_dir = "patient_contributions"
os.makedirs(output_dir, exist_ok=True)


# Function to structure the output like the mock example
def format_model_output(model_name, shap_data, lime_data, inherent_data):
    return {
        model_name: {
            "SHAP": {row["Feature"]: row["Normalized Contribution"] for row in shap_data},
            "Lime": {row["Feature"]: row["Normalized Contribution"] for row in lime_data},
            "Inherent": {row["Feature"]: row["Normalized Contribution"] for row in inherent_data},
        }
    }


# Loop through each patient and generate structured JSON
for i in range(len(X_sample)):
    individual_id = ids.iloc[i]
    json_output = {}

    for model in models:
        model_name = model.backend_get_name()

        # Select appropriate data for normalization
        if model_name == "XGBOOST":
            X_sample_for_prediction = X_sample
            X_train_for_prediction = X_train
        else:
            X_sample_for_prediction = normalized_X
            X_train_for_prediction = normalize_data(X_train)

        individual_data = X_sample_for_prediction.iloc[[i]]

        # Generate contributions
        inherent_df = model.backend_inherent(individual_data)
        # Get backend_local_shap contributions
        if model.get_type() == "LogisticRegression":
            shap_df = decisionTreeModel.backend_local_shap(individual_data)
        else:
            shap_df = model.backend_local_shap(individual_data)
        lime_df = model.backend_local_lime(X_train_for_prediction, individual_data)

        # Normalize contributions
        inherent_df = normalize_contributions(inherent_df)
        shap_df = normalize_contributions(shap_df)
        lime_df = normalize_contributions(lime_df)

        # Format output
        formatted_output = format_model_output(
            model_name,
            shap_data=shap_df.to_dict(orient="records"),
            lime_data=lime_df.to_dict(orient="records"),
            inherent_data=inherent_df.to_dict(orient="records"),
        )

        json_output.update(formatted_output)

    # Save the JSON for the patient
    output_file = os.path.join(output_dir, f"patient_{individual_id}_explanation.json")
    with open(output_file, "w") as f:
        json.dump(json_output, f, indent=4)

    print(f"Contributions saved to {output_file}")
