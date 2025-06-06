from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np


def normalize_data(X_train, X_test):
    """
    Normalize the dataset using StandardScaler (mean=0, std=1).

    Args:
        X_train: The training features (pandas DataFrame).
        X_test: The test features (pandas DataFrame).

    Returns:
        X_train_normalized: The normalized training set (pandas DataFrame).
        X_test_normalized: The normalized test set (pandas DataFrame).
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform the train set
    X_train_normalized = scaler.fit_transform(X_train)

    # Transform the test set using the same scaler (no fitting on the test set)
    X_test_normalized = scaler.transform(X_test)

    # Return as DataFrame with the same column names
    return pd.DataFrame(X_train_normalized, columns=X_train.columns), pd.DataFrame(X_test_normalized, columns=X_test.columns)


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
        print("Applying SMOTE for oversampling...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elif method == "undersample":
        print("Applying Random Undersampling...")
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    elif method is None:
        print("Skipping balancing...")
        X_resampled, y_resampled = X_train, y_train
    else:
        raise ValueError("Method should be either 'smote', 'undersample', or None.")

    return X_resampled, y_resampled



def feature_engineering(data):
    # Drop columns that are likely outputs of prediction models (e.g., probabilistic predictions)
    data = data.drop(columns=['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob'], errors='ignore') #xgboost

    # # Drop columns that represent complex calculations or derived scores (e.g., APACHE diagnoses)
    # data = data.drop(columns=['apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative',
    #                           'apache_3j_bodysystem', 'apache_2_bodysystem'], errors='ignore')
    #
    # # Drop columns containing derived clinical metrics related to "apache" (e.g., scores or physiological assessments)
    # data = data.drop(columns=[col for col in data.columns if 'apache' in col], errors='ignore')

    # Drop columns that are purely identifiers or irrelevant to modeling (e.g., patient or hospital IDs)
    data = data.drop(columns=['patient_id', 'encounter_id'], errors='ignore') #xgboost
    # # 1. Aggregated Features (Range of Vital Signs) # one tree
    # data['d1_diasbp_range'] = data['d1_diasbp_max'] - data['d1_diasbp_min']
    # data['d1_heartrate_range'] = data['d1_heartrate_max'] - data['d1_heartrate_min']
    # data['d1_mbp_range'] = data['d1_mbp_max'] - data['d1_mbp_min']
    # data['d1_resprate_range'] = data['d1_resprate_max'] - data['d1_resprate_min']
    # data['d1_spo2_range'] = data['d1_spo2_max'] - data['d1_spo2_min']
    # data['d1_sysbp_range'] = data['d1_sysbp_max'] - data['d1_sysbp_min']
    # data['d1_temp_range'] = data['d1_temp_max'] - data['d1_temp_min']
    #
    # # 2. Relative Ratios
    # data['bilirubin_to_creatinine'] = data['d1_bilirubin_max'] / (data['d1_creatinine_max'] + 0.1) # one tree
    # data['bun_to_creatinine'] = data['d1_bun_max'] / (data['d1_creatinine_max'] + 0.1)
    # data['pao2_fio2_ratio'] = data['pao2_apache'] / (data['fio2_apache'] + 0.1)

    # # 3. Binary Flags for Critical Ranges
    # data['high_bun_flag'] = np.where(data['d1_bun_max'] > 30, 1, 0)
    # data['high_creatinine_flag'] = np.where(data['d1_creatinine_max'] > 1.5, 1, 0)
    # data['low_albumin_flag'] = np.where(data['d1_albumin_min'] < 3.5, 1, 0)

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

    # # 8. Sepsis Triad
    # data['sepsis_triad_flag'] = (
    #         (data['d1_heartrate_max'] > 100) &  # Tachycardia
    #         (data['d1_temp_max'] > 38) &  # Fever
    #         (data['d1_resprate_max'] > 20)  # Increased respiratory rate
    # ).astype(int)
    #
    # 9. Renal Function Triad #xgboost
    data['renal_function_triad_flag'] = (
            (data['d1_creatinine_max'] > 1.5) &  # Elevated creatinine
            (data['urineoutput_apache'] < 500) &  # Low urine output
            (data['d1_bun_max'] > 30)  # Elevated BUN
    ).astype(int)

    # 10. Multi-Organ Dysfunction Syndrome (MODS) Pentad #xgboost # one tree
    data['mods_pentad_flag'] = (
            (data['d1_creatinine_max'] > 1.5) &  # Renal dysfunction
            (data['d1_bilirubin_max'] > 2) &  # Hepatic dysfunction
            (data['d1_sysbp_min'] < 90) &  # Cardiovascular dysfunction
            (data['d1_spo2_min'] < 90) &  # Respiratory dysfunction
            (data['d1_platelets_min'] < 150)  # Hematological dysfunction
    ).astype(int)

    return data



# ---------------------------------------------- new dataSet -----------------------------------------------------------


def preprocess_readmitted_label(df):
    """
    ממיר את העמודה 'readmitted' ל-label בינארי:
    1 = אושפז תוך פחות מ-30 יום (<30)
    0 = לא אושפז תוך פחות מ-30 יום (>30 או NO)
    """
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    return df


def preprocessing(df):
    df = preprocess_readmitted_label(df)

    # הסרת מזהים
    df = df.drop(['encounter_id', 'patient_nbr'], axis=1)

    # החלפת '?' ב-NA
    df = df.replace('?', pd.NA)

    # הסרת עמודות עם יותר מ-50% ערכים חסרים
    missing_percent = df.isna().mean()
    df = df.drop(missing_percent[missing_percent > 0.5].index, axis=1)

    # הסרת עמודות עם ערך אחד בלבד
    nunique = df.nunique()
    df = df.drop(nunique[nunique <= 1].index, axis=1)

    df = preprocess_data(df)

    # ניקוי שמות עמודות כך שיתאימו ל־XGBoost
    # df.columns = df.columns.str.replace(r'[\[\]<>/\\\-]', '_', regex=True)

    return df
