from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns

def train_model(X_train, y_train):
    """
    Trains a K-Nearest Neighbors (KNN) model using PCA and grid search for hyperparameter tuning.
    
    Parameters:
        X_train (numpy array): Training feature data.
        y_train (numpy array): Labels for training data.

    Returns:
        model (GridSearchCV object): Trained KNN model with best hyperparameters.
        pca (PCA object): PCA transformer fitted on the training data.
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=3)
    X_train = pca.fit_transform(X_train)

    # Hyperparameter tuning grid for KNN
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['distance', 'uniform'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Initialize the KNN classifier and perform grid search
    model = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)

    # Evaluate the model on the training set
    preds = model.predict(X_train)
    print("Best KNN Params:", model.best_params_)
    print("Training Accuracy:", accuracy_score(y_train, preds))
    print("Classification Report:\n", classification_report(y_train, preds))

    return model, pca

def evaluate_model(model, pca, X_test, y_test, output_dir="./Models_Performance/KNN/"):
    """
    Evaluates the performance of a trained KNN model on test data, 
    saves evaluation results, and plots the confusion matrix.

    Parameters:
        model (GridSearchCV object): Trained KNN model.
        pca (PCA object): PCA transformer fitted on training data.
        X_test (numpy array): Test feature data.
        y_test (numpy array): Labels for test data.
        output_dir (str): Directory to save evaluation results.

    Outputs:
        Evaluation metrics printed to console, and confusion matrix saved as a PNG file.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Transform the test data using PCA
    X_test = pca.transform(X_test)

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

    # Plot and save confusion matrix as PNG
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (KNN)")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix saved to: {output_dir}/confusion_matrix.png")

def save_model(model, directory, model_name="knn_model.pkl"):
    """
    Saves the trained model to a specified directory using joblib.

    Parameters:
        model (GridSearchCV object): Trained model to be saved.
        directory (str): Directory to save the model file.
        model_name (str): Name of the model file.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Full path to save the model
    model_path = os.path.join(directory, model_name)
    
    # Save the model using joblib
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

def load_model(directory, model_name="knn_model.pkl"):
    """
    Loads a saved model from a specified directory.

    Parameters:
        directory (str): Directory containing the model file.
        model_name (str): Name of the model file.

    Returns:
        model (GridSearchCV object): Loaded model.
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

def visualize_knn_clusters_pca(model, pca, X_train, y_train, X_test=None, y_test=None):
    """ ** NOT USEFUL FOR 3-DIMENSION WHICH IS WHAT IS BEING USED **
    Visualizes KNN decision boundaries on 2D PCA-transformed data and plots train and test points.

    Args:
        model (GridSearchCV object): Trained KNN model.
        pca (PCA object): PCA transformer fitted on training data.
        X_train (numpy array): Training feature data (before PCA).
        y_train (numpy array): Labels for training data.
        X_test (numpy array, optional): Test feature data (before PCA).
        y_test (numpy array, optional): Labels for test data.

    Displays:
        Decision boundary plot with training and test data points.
    """
    # Reduce the training data to 2D
    X_train_pca = pca.transform(X_train)
    
    if X_test is not None:
        X_test_pca = pca.transform(X_test)

    # Create a mesh grid to plot the decision boundaries
    h = 0.5  # Increase the step size to reduce memory usage
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on the mesh grid using the PCA-transformed data
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)  # Direct prediction on the PCA-reduced space
    Z = Z.reshape(xx.shape)

    # Set up the color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    # Plot the decision boundary by assigning a color to each point in the mesh
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot the training points after PCA transformation
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=50)
    
    # Optionally plot the test points if provided
    if X_test is not None and y_test is not None:
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=100, marker='*')

    # Add a legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)

    # Set plot titles and labels
    plt.title("KNN Classification with PCA (2D) and Decision Boundaries")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()
