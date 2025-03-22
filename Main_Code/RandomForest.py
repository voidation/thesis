import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(X_train, y_train):
    """
    Trains a Random Forest model with hyperparameter tuning using GridSearchCV.

    Parameters:
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.

    Returns:
        model (GridSearchCV): Trained model with best found parameters.
    """
    # Hyper parameter tuning grid
    param_grid_rf = {
        'n_estimators': [200]
    }

    model = GridSearchCV(RandomForestClassifier(random_state=25), param_grid_rf, cv=5, scoring='f1_weighted')
    model.fit(X_train, y_train)

    preds = model.predict(X_train)
    print("Best Random Forest Params:", model.best_params_)
    print("Training Accuracy:", accuracy_score(y_train, preds))
    print("Classification Report:\n", classification_report(y_train, preds))

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data and prints accuracy, classification report, and confusion matrix.

    Parameters:
        model (GridSearchCV): Trained Random Forest model.
        X_test (numpy.ndarray): Test data features.
        y_test (numpy.ndarray): True labels for the test data.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def save_model(model, directory, model_name="random_forest_model.pkl"):
    """
    Saves the trained model to a specified directory.

    Parameters:
        model (GridSearchCV): Trained Random Forest model to be saved.
        directory (str): Directory where the model will be saved.
        model_name (str): Name of the model file. Defaults to 'random_forest_model.pkl'.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Full path to save the model
    model_path = os.path.join(directory, model_name)
    
    # Save the model using joblib
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

def load_model(directory, model_name="random_forest_model.pkl"):
    """
    Loads a pre-trained model from a specified directory.

    Args:
        directory (str): Directory from which the model will be loaded.
        model_name (str): Name of the model file. Defaults to 'random_forest_model.pkl'.

    Returns:
        model (GridSearchCV): Loaded model ready for evaluation or prediction.

    Raises:
        FileNotFoundError: If the specified model file does not exist in the directory.
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