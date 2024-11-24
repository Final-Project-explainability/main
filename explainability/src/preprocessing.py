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

    data = clean_data(data)

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
    data = data.drop(columns=['patient_id', 'encounter_id'], errors='ignore')

    # 1. Aggregated Features (Range of Vital Signs)
    data['d1_diasbp_range'] = data['d1_diasbp_max'] - data['d1_diasbp_min']
    data['d1_heartrate_range'] = data['d1_heartrate_max'] - data['d1_heartrate_min']
    data['d1_mbp_range'] = data['d1_mbp_max'] - data['d1_mbp_min']
    data['d1_resprate_range'] = data['d1_resprate_max'] - data['d1_resprate_min']
    data['d1_spo2_range'] = data['d1_spo2_max'] - data['d1_spo2_min']
    data['d1_sysbp_range'] = data['d1_sysbp_max'] - data['d1_sysbp_min']
    data['d1_temp_range'] = data['d1_temp_max'] - data['d1_temp_min']

    # # 2. Relative Ratios
    # data['bilirubin_to_creatinine'] = data['d1_bilirubin_max'] / (data['d1_creatinine_max'] + 0.1)
    # data['bun_to_creatinine'] = data['d1_bun_max'] / (data['d1_creatinine_max'] + 0.1)
    # data['pao2_fio2_ratio'] = data['pao2_apache'] / (data['fio2_apache'] + 0.1)

    # 3. Binary Flags for Critical Ranges
    data['high_bun_flag'] = np.where(data['d1_bun_max'] > 30, 1, 0)
    data['high_creatinine_flag'] = np.where(data['d1_creatinine_max'] > 1.5, 1, 0)
    data['low_albumin_flag'] = np.where(data['d1_albumin_min'] < 3.5, 1, 0)

    # # 4. Temporal Features (Binning pre-ICU length of stay)
    # data['pre_icu_los_days_bin'] = pd.cut(data['pre_icu_los_days'], bins=[-1, 1, 3, 7, 30],
    #                                       labels=['<1 day', '1-3 days', '3-7 days', '>7 days'])

    # # 5. Interaction and Polynomial Features
    # data['age_squared'] = data['age'] ** 2
    # data['heart_rate_map_interaction'] = data['heart_rate_apache'] * data['map_apache']

    # # 6. Medical Condition Flags - Severity Score
    # data['severity_score'] = data[
    #     ['aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma',
    #      'solid_tumor_with_metastasis']].sum(axis=1)

    # # 7. Encoding Categorical Variables
    # data = pd.get_dummies(data, columns=['ethnicity', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem',
    #                                      'pre_icu_los_days_bin'], drop_first=True)

    # 3. Sepsis Triad
    data['sepsis_triad_flag'] = (
            (data['d1_heartrate_max'] > 100) &  # Tachycardia
            (data['d1_temp_max'] > 38) &  # Fever
            (data['d1_resprate_max'] > 20)  # Increased respiratory rate
    ).astype(int)

    # 4. Renal Function Triad
    data['renal_function_triad_flag'] = (
            (data['d1_creatinine_max'] > 1.5) &  # Elevated creatinine
            (data['urineoutput_apache'] < 500) &  # Low urine output
            (data['d1_bun_max'] > 30)  # Elevated BUN
    ).astype(int)

    # 5. Multi-Organ Dysfunction Syndrome (MODS) Pentad
    data['mods_pentad_flag'] = (
            (data['d1_creatinine_max'] > 1.5) &  # Renal dysfunction
            (data['d1_bilirubin_max'] > 2) &  # Hepatic dysfunction
            (data['d1_sysbp_min'] < 90) &  # Cardiovascular dysfunction
            (data['d1_spo2_min'] < 90) &  # Respiratory dysfunction
            (data['d1_platelets_min'] < 150)  # Hematological dysfunction
    ).astype(int)

    return data


import pandas as pd
import numpy as np


def clean_data(data):
    """
    Handle missing values in the dataset by identifying and imputing them.
    Args:
        data (DataFrame): The input dataset.
    Returns:
        DataFrame: Dataset with missing values handled.
    """
    # # Define additional missing value representations
    # missing_values = ['NA', 'N/A', '', 'None', '?', 'nan', 'NaN']
    #
    # # Replace these representations with np.nan
    # data.replace(missing_values, np.nan, inplace=True)
    #
    # # Check and display missing values
    # missing_columns = data.columns[data.isnull().any()]
    # print(f"Columns with missing values: {list(missing_columns)}")
    #
    # # Handle missing values: Example strategies
    # for column in missing_columns:
    #     if data[column].dtype in ['float64', 'int64']:  # Numeric columns
    #         data[column].fillna(data[column].median(), inplace=True)  # Fill with median
    #     else:  # Non-numeric columns
    #         data[column].fillna(data[column].mode()[0], inplace=True)  # Fill with mode
    #
    # print("All missing values have been handled.")
    #
    # # Logical Checks
    # print("Checking and handling logically invalid values...")
    #
    # # גיל
    # invalid_age_count = data[(data['age'] < 0) | (data['age'] > 120)].shape[0]
    # if invalid_age_count > 0:
    #     print(f"Found {invalid_age_count} invalid ages. Capping values to range [0, 120].")
    #     data['age'] = data['age'].clip(lower=0, upper=120)
    #
    # # BMI
    # invalid_bmi_count = data[(data['bmi'] < 10) | (data['bmi'] > 80)].shape[0]
    # if invalid_bmi_count > 0:
    #     print(f"Found {invalid_bmi_count} invalid BMI values. Capping values to range [10, 80].")
    #     data['bmi'] = data['bmi'].clip(lower=10, upper=80)
    #
    # # גובה
    # invalid_height_count = data[(data['height'] < 50) | (data['height'] > 250)].shape[0]
    # if invalid_height_count > 0:
    #     print(f"Found {invalid_height_count} invalid heights. Capping values to range [50, 250].")
    #     data['height'] = data['height'].clip(lower=50, upper=250)
    #
    # # משקל
    # invalid_weight_count = data[(data['weight'] < 3) | (data['weight'] > 300)].shape[0]
    # if invalid_weight_count > 0:
    #     print(f"Found {invalid_weight_count} invalid weights. Capping values to range [3, 300].")
    #     data['weight'] = data['weight'].clip(lower=3, upper=300)
    #
    # # אלבומין
    # invalid_albumin_count = data[(data['albumin_apache'] < 1.0) | (data['albumin_apache'] > 5.0)].shape[0]
    # if invalid_albumin_count > 0:
    #     print(f"Found {invalid_albumin_count} invalid albumin values. Capping values to range [1.0, 5.0].")
    #     data['albumin_apache'] = data['albumin_apache'].clip(lower=1.0, upper=5.0)
    #
    # # המוגלובין
    # invalid_hematocrit_count = data[(data['hematocrit_apache'] < 10) | (data['hematocrit_apache'] > 100)].shape[0]
    # if invalid_hematocrit_count > 0:
    #     print(f"Found {invalid_hematocrit_count} invalid hematocrit values. Capping values to range [20, 60].")
    #     data['hematocrit_apache'] = data['hematocrit_apache'].clip(lower=20, upper=60)
    #
    # # רמת גלוקוז
    # invalid_glucose_count = data[(data['glucose_apache'] < 20) | (data['glucose_apache'] > 900)].shape[0]
    # if invalid_glucose_count > 0:
    #     print(f"Found {invalid_glucose_count} invalid glucose values. Capping values to range [20, 900].")
    #     data['glucose_apache'] = data['glucose_apache'].clip(lower=20, upper=900)
    #
    # # דופק
    # invalid_hr_count = data[(data['heart_rate_apache'] < 30) | (data['heart_rate_apache'] > 300)].shape[0]
    # if invalid_hr_count > 0:
    #     print(f"Found {invalid_hr_count} invalid heart rate values. Capping values to range [30, 300].")
    #     data['heart_rate_apache'] = data['heart_rate_apache'].clip(lower=30, upper=300)
    #
    # # לחץ דם ממוצע
    # invalid_map_count = data[(data['map_apache'] < 20) | (data['map_apache'] > 200)].shape[0]
    # if invalid_map_count > 0:
    #     print(f"Found {invalid_map_count} invalid mean arterial pressure values. Capping values to range [20, 150].")
    #     data['map_apache'] = data['map_apache'].clip(lower=20, upper=200)
    #
    # # טמפרטורה
    # invalid_temp_count = data[(data['temp_apache'] < 30) | (data['temp_apache'] > 43)].shape[0]
    # if invalid_temp_count > 0:
    #     print(f"Found {invalid_temp_count} invalid temperature values. Capping values to range [30, 43].")
    #     data['temp_apache'] = data['temp_apache'].clip(lower=30, upper=43)
    #
    # # ניקוי עמודת fio2_apache
    # invalid_fio2_count = data[(data['fio2_apache'] < 0.21) | (data['fio2_apache'] > 1.0)].shape[0]
    # if invalid_fio2_count > 0:
    #     print(f"Found {invalid_fio2_count} invalid FiO2 values. Capping values to range [0.21, 1.0].")
    #     data['fio2_apache'] = data['fio2_apache'].clip(lower=0.21, upper=1.0)
    #
    # # כמות שתן
    # invalid_urine_count = data[(data['urineoutput_apache'] < 0) | (data['urineoutput_apache'] > 10000)].shape[0]
    # if invalid_urine_count > 0:
    #     print(f"Found {invalid_urine_count} invalid urine output values. Capping values to range [0, 10000].")
    #     data['urineoutput_apache'] = data['urineoutput_apache'].clip(lower=0, upper=10000)
    #
    # print("Logical checks and corrections complete.")
    return data
