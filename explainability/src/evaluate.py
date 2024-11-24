import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)


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


# def evaluate_model(model, X_test, y_test):
#     """
#     Evaluate the trained model on the test data.
#     Args:
#         model: The trained model.
#         X_test (DataFrame): Features for testing.
#         y_test (Series): True target values for testing.
#     """
#     # Make predictions
#     y_pred = model.predict(X_test)
#
#     # Calculate accuracy, precision, recall
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#
#     # Calculate recall for top 2% and 5%
#     recall_2_percent = top_percent_recall(model, X_test, y_test, 2)
#     recall_5_percent = top_percent_recall(model, X_test, y_test, 5)
#
#     # Print classification report for more details
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"Recall for top 2%: {recall_2_percent:.4f}")
#     print(f"Recall for top 5%: {recall_5_percent:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))


# def evaluate_model(model, X_test, y_test):
#     # Model predictions
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]
#
#     # Calculate basic metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_proba)
#
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"ROC AUC: {roc_auc:.4f}")
#
#     # Classification report
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))
#
#     # Calculate metrics for Top 2% and 5% risk predictions
#     top_2_percent_threshold = np.percentile(y_proba, 98)
#     top_5_percent_threshold = np.percentile(y_proba, 95)
#
#     # Predictions for the highest risk categories
#     top_2_preds = (y_proba >= top_2_percent_threshold).astype(int)
#     top_5_preds = (y_proba >= top_5_percent_threshold).astype(int)
#
#     top_2_recall = recall_score(y_test, top_2_preds)
#     top_5_recall = recall_score(y_test, top_5_preds)
#
#     print(f"Recall for Top 2%: {top_2_recall:.4f}")
#     print(f"Recall for Top 5%: {top_5_recall:.4f}")


def evaluate_model(model, X_test, y_test, optimal_threshold=0.5):
    # Model predictions with probability
    y_proba = model.predict_proba(X_test)[:, 1]

    # Apply custom threshold for classification
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate metrics for Top 2%, 5%, and 10% risk predictions
    top_2_percent_threshold = np.percentile(y_proba, 98)
    top_5_percent_threshold = np.percentile(y_proba, 95)
    top_10_percent_threshold = np.percentile(y_proba, 90)

    # Predictions for the highest risk categories
    top_2_preds = (y_proba >= top_2_percent_threshold).astype(int)
    top_5_preds = (y_proba >= top_5_percent_threshold).astype(int)
    top_10_preds = (y_proba >= top_10_percent_threshold).astype(int)

    top_2_recall = recall_score(y_test, top_2_preds)
    top_5_recall = recall_score(y_test, top_5_preds)
    top_10_recall = recall_score(y_test, top_10_preds)

    print(f"Recall for Top 2%: {top_2_recall:.4f}")
    print(f"Recall for Top 5%: {top_5_recall:.4f}")
    print(f"Recall for Top 10%: {top_10_recall:.4f}")

    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()