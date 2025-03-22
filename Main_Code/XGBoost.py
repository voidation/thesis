import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    """
    Trains an XGBoost model with hyperparameter tuning using GridSearchCV.

    Parameters:
        X_train (np.array): Training feature set.
        y_train (np.array): Training labels.

    Returns:
        model (GridSearchCV): Trained model with the best hyperparameters found during grid search.
    """
    # Hyperparameter tuning grid    
    param_grid_xgb = {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.01, 0.1],
        'n_estimators': [100, 200],
    }

    model = GridSearchCV(xgb.XGBClassifier(random_state=25), param_grid_xgb, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)

    preds = model.predict(X_train)
    print("\nBest XGBoost Params:", model.best_params_)
    print("Training Accuracy:", accuracy_score(y_train, preds))
    print("Classification Report:\n", classification_report(y_train, preds))

    return model

def evaluate_model(model, X_test, y_test, output_dir="./Models_Performance/XGB/"):
    """
    Evaluates the model on a test set, generating accuracy, precision, recall, and a confusion matrix.

    Parameters:
        model (GridSearchCV): Trained XGBoost model.
        X_test (np.array): Test feature set.
        y_test (np.array): Test labels.
        output_dir (str): Directory to save evaluation report and confusion matrix plot.

    Outputs:
        Saves a text file with the evaluation report and a PNG file with the confusion matrix.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Precision and Recall for both classes (binary classification)
    precision_high = precision_score(y_test, y_pred)
    recall_high = recall_score(y_test, y_pred)
    precision_low = precision_score(y_test, y_pred, pos_label=0)
    recall_low = recall_score(y_test, y_pred, pos_label=0)

    # Classification Report and Confusion Matrix
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Print results to the console
    print(f"Accuracy: {accuracy}")
    print(f"Precision (Class 1): {precision_high}, Recall (Class 1): {recall_high}")
    print(f"Precision (Class 0): {precision_low}, Recall (Class 0): {recall_low}")
    print(f"Confusion Matrix:\n{cm}")

    # Save the classification report
    report_filepath = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_filepath, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision (Class 1): {precision_high}, Recall (Class 1): {recall_high}\n")
        f.write(f"Precision (Class 0): {precision_low}, Recall (Class 0): {recall_low}\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
    print(f"Classification report saved to: {report_filepath}")

    # Plot confusion matrix and save as PNG
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (XGBoost)")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix saved to: {output_dir}/confusion_matrix.png")

def save_model(model, directory, model_name="xgboost_model.pkl"):
    """
    Saves the trained XGBoost model to a specified directory.

    Parameters:
        model (GridSearchCV): The trained model to be saved.
        directory (str): Path to the directory where the model should be saved.
        model_name (str): Name for the saved model file.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Full path to save the model
    model_path = os.path.join(directory, model_name)
    
    # Save the model using joblib
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

def load_model(directory, model_name="xgboost_model.pkl"):
    """
    Loads a trained XGBoost model from a specified directory.

    Parameters:
        directory (str): Directory where the model is saved.
        model_name (str): Name of the saved model file.

    Returns:
        model (GridSearchCV): Loaded model.
    """
    # Full path to load the model
    model_path = os.path.join(directory, model_name)
    
    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at: {model_path}")
    
    # Load the model using joblib
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    return model