from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


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
