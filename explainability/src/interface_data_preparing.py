import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from explainability.src.ModelManager import ModelManager
from preprocessing import preprocess_data, feature_engineering, preprocessing
from sklearn.model_selection import train_test_split
import json
import os

# Load the data
file_path = "example_test_data_new.csv"
data = pd.read_csv(file_path)
print(f"Data loaded successfully from {file_path}")


# Separate features (X), target (y), and IDs
X_sample = data.drop(columns=['readmitted', 'patient_nbr'])
y = data['readmitted']
ids = data['patient_nbr']

# data_before_preprocessing = load_data()
#
# # שלב 1: טען את הנתונים המקוריים
# data_before_preprocessing = load_data()
#
# # שלב 2: הגדר את קבוצת ה-IDs הרלוונטיים
# ids = set(data['patient_id'])
#
# # שלב 3: סינון הנתונים לפי patient_id
# filtered_data = data_before_preprocessing[data_before_preprocessing['patient_id'].isin(ids)]
#
# # שלב 4: הסרה של encounter_id אם קיימת
# if 'encounter_id' in filtered_data.columns:
#     filtered_data = filtered_data.drop(columns=['encounter_id'])
#
# # שלב 5: המרת כל הערכים למחרוזות (NaN יהפוך ל-"nan")
# filtered_data_str = filtered_data.astype(str)
#
# # שלב 6: המרה למבנה JSON
# records = filtered_data_str.to_dict(orient='records')
#
# # שלב 7: שמירה לקובץ JSON
# with open('filtered_patients.json', 'w', encoding='utf-8') as f:
#     json.dump(records, f, ensure_ascii=False, indent=4)

# Load the models
decisionTreeModel = ModelManager.load_model("DecisionTreeClassifier")
logisticRegressionModel = ModelManager.load_model("LogisticRegression")
xgboostModel = ModelManager.load_model("XGBClassifier")

# probs = decisionTreeModel.get_unique_leaf_probabilities()
# print("הסתברויות שונות למוות:", probs)
# print("סה״כ הסתברויות שונות:", len(probs))


# Initialize the StandardScaler
scaler = StandardScaler()


def normalize_data(X):
    """
    Normalize the dataset using StandardScaler (mean=0, std=1).

    Args:
        X: The training features (pandas DataFrame).

    Returns:
        X_normalized: The normalized training set (pandas DataFrame).
    """

    # Fit the scaler to the training data and transform the train set
    X_normalized = scaler.fit_transform(X)

    # Return as DataFrame with the same column names
    return pd.DataFrame(X_normalized, columns=X.columns)


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
    data = load_data()
    print("Column names in the dataset:", data.columns)

    newData = True

    if (newData):
        data = preprocessing(data)
        label = 'readmitted'

    else:
        # Feature engineering and preprocessing
        data = feature_engineering(data)
        data = preprocess_data(data)
        label = 'hospital_death'

    X = data.drop(columns=[label])
    y = data[label]
    # Load and preprocess the dataset


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_model_data()
models = [logisticRegressionModel, decisionTreeModel, xgboostModel]

# Directory to save JSON files
output_dir = "patient_contributions_new_data"
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


X_train_normalized = normalize_data(X_train)
normalized_X_sample = normalize_data(X_sample)


def build_contributions_matrix():
    # Loop through each patient and generate structured JSON
    for i in range(len(X_sample)):
        individual_id = ids.iloc[i]
        json_output = {}

        for model in models:
            model_name = model.backend_get_name()

            # Select appropriate data for normalization
            if model_name == "LogisticRegression":
                X_sample_for_prediction = normalized_X_sample
                X_train_for_prediction = X_train_normalized
            else:
                X_sample_for_prediction = X_sample
                X_train_for_prediction = X_train

            individual_data = X_sample_for_prediction.iloc[[i]]

            # Generate contributions
            inherent_df = model.backend_inherent(individual_data)
            # Get backend_local_shap contributions
            # if model.get_type() == "LogisticRegression":
            #     shap_df = decisionTreeModel.backend_local_shap(individual_data)
            shap_df = model.backend_local_shap(individual_data, X_train_for_prediction)
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


build_contributions_matrix()

X_train_sample = X_train.sample(frac=0.05, random_state=42)
X_train_sample_normalized = X_train_normalized.sample(frac=0.05, random_state=42)

def global_explain():
    for model in models:
        model_name = model.backend_get_name()
        if model_name == "LogisticRegression":
            X_sample_for_prediction = X_train_sample_normalized
            X_train_for_prediction = X_train_normalized
        else:
            X_sample_for_prediction = X_train_sample
            X_train_for_prediction = X_train

        shap_val = model.global_explain_with_shap(X_train_for_prediction)
        lime_val = model.global_explain_with_lime(X_train_for_prediction, X_sample_for_prediction)
        inherent_df = model.global_explain_inherent(X_train=X_train_for_prediction)

        shap_df = normalize_contributions(shap_val)
        lime_df = normalize_contributions(lime_val)
        inherent_df = normalize_contributions(inherent_df)

        # Format output
        formatted_output = format_model_output(
            model_name,
            shap_data=shap_df.to_dict(orient="records"),
            lime_data=lime_df.to_dict(orient="records"),
            inherent_data=inherent_df.to_dict(orient="records"),
        )

        json_output = {}
        json_output.update(formatted_output)
        # Save the JSON for the patient
        output_file = os.path.join(f"{model_name}_global.json")
        with open(output_file, "w") as f:
            json.dump(json_output, f, indent=4)

        print(f"Contributions saved to {output_file}")

#global_explain()