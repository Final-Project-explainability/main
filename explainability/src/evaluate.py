from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
import numpy as np


def top_percent_recall(model, X_test, y_test, percentage):
    """
    Calculate recall for the top X% of predicted probabilities.

    Args:
        model: The trained model.
        X_test: Test features.
        y_test: True labels for the test set.
        percentage: The top percentage of samples to calculate recall (e.g., 2 or 5).

    Returns:
        recall: The recall for the top X% predictions.
    """
    # Step 1: Predict probabilities for the test set
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (hospital death)

    # Step 2: Sort the predicted probabilities in descending order
    sorted_indices = np.argsort(probabilities)[::-1]

    # Step 3: Select the top X% of the data
    top_n = int(len(probabilities) * (percentage / 100))
    top_indices = sorted_indices[:top_n]

    # Step 4: Calculate recall on the top X% predictions
    top_y_true = y_test.iloc[top_indices]
    top_y_pred = np.ones(top_n)  # Assume top predictions are all '1'

    recall = recall_score(top_y_true, top_y_pred)

    return recall


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    Args:
        model: The trained model.
        X_test (DataFrame): Features for testing.
        y_test (Series): True target values for testing.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Calculate recall for top 2% and 5%
    recall_2_percent = top_percent_recall(model, X_test, y_test, 2)
    recall_5_percent = top_percent_recall(model, X_test, y_test, 5)

    # Print classification report for more details
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Recall for top 2%: {recall_2_percent:.4f}")
    print(f"Recall for top 5%: {recall_5_percent:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
