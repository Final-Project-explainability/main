from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np


def preprocess_data(data):
    """
    Preprocess the dataset by encoding categorical variables and handling missing values.
    Args:
        data (DataFrame): The original dataset.
    Returns:
        data (DataFrame): Preprocessed dataset with encoded categorical variables and no missing values.
    """
    # Identify columns with categorical data
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Encode categorical columns
    for col in categorical_columns:
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col].astype(str))

    # Handle missing values by filling them with the median
    data = data.fillna(data.median())

    return data


def balance_data(X_train, y_train, method="smote"):
    """
    Balance the training dataset using SMOTE (oversampling) or undersampling.
    Args:
        X_train (DataFrame): Features for training.
        y_train (Series): Target values for training.
        method (str): The method for balancing the data. Options are "smote" or "undersample".
    Returns:
        X_resampled, y_resampled: The balanced dataset.
    """
    if method == "smote":
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elif method == "undersample":
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    elif method is None:
        X_resampled, y_resampled = X_train, y_train
    else:
        raise ValueError("Method should be either 'smote' or 'undersample'.")

    return X_resampled, y_resampled


def feature_engineering(data):
    # Drop redundant columns
    data = data.drop(columns=['patient_id', 'encounter_id', 'hospital_id'], errors='ignore')

    # 1. Aggregated Features (Range of Vital Signs)
    data['d1_diasbp_range'] = data['d1_diasbp_max'] - data['d1_diasbp_min']
    data['d1_heartrate_range'] = data['d1_heartrate_max'] - data['d1_heartrate_min']
    data['d1_mbp_range'] = data['d1_mbp_max'] - data['d1_mbp_min']
    data['d1_resprate_range'] = data['d1_resprate_max'] - data['d1_resprate_min']
    data['d1_spo2_range'] = data['d1_spo2_max'] - data['d1_spo2_min']
    data['d1_sysbp_range'] = data['d1_sysbp_max'] - data['d1_sysbp_min']
    data['d1_temp_range'] = data['d1_temp_max'] - data['d1_temp_min']

    # 2. Relative Ratios
    data['bilirubin_to_creatinine'] = data['d1_bilirubin_max'] / (data['d1_creatinine_max'] + 0.1)
    data['bun_to_creatinine'] = data['d1_bun_max'] / (data['d1_creatinine_max'] + 0.1)
    data['pao2_fio2_ratio'] = data['pao2_apache'] / (data['fio2_apache'] + 0.1)

    # 3. Binary Flags for Critical Ranges
    data['high_bun_flag'] = np.where(data['d1_bun_max'] > 30, 1, 0)
    data['high_creatinine_flag'] = np.where(data['d1_creatinine_max'] > 1.5, 1, 0)
    data['low_albumin_flag'] = np.where(data['d1_albumin_min'] < 3.5, 1, 0)

    # 4. Temporal Features (Binning pre-ICU length of stay)
    data['pre_icu_los_days_bin'] = pd.cut(data['pre_icu_los_days'], bins=[-1, 1, 3, 7, 30],
                                          labels=['<1 day', '1-3 days', '3-7 days', '>7 days'])

    # 5. Interaction and Polynomial Features
    data['age_squared'] = data['age'] ** 2
    data['heart_rate_map_interaction'] = data['heart_rate_apache'] * data['map_apache']

    # 6. Medical Condition Flags - Severity Score
    data['severity_score'] = data[
        ['aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma',
         'solid_tumor_with_metastasis']].sum(axis=1)

    # 7. Encoding Categorical Variables
    data = pd.get_dummies(data, columns=['ethnicity', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem',
                                         'pre_icu_los_days_bin'], drop_first=True)

    return data
