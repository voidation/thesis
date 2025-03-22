import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MLPModel:
    def __init__(self, input_shape):
        """
        Initializes the MLP model instance with the input shape.
        
        Parameters:
            input_shape (int): Number of features in the input data.
        """
        self.input_shape = input_shape
        self.model_instance = None  # For holding the trained model

    def build_model(self):
        """
        Builds a simple Multi-Layer Perceptron (MLP) model with hidden layers and dropout for regularization.
        
        Returns:
            model (tf.keras.Sequential): Compiled MLP model ready for training.
        """
        model = models.Sequential()

        # Input layer
        model.add(layers.Input(shape=(self.input_shape,)))

        # Hidden layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))  # Dropout for regularization
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(0.3))

        # Output layer for binary classification
        model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

        return model

    def train_model(self, X_train, y_train, model_dir, lr=0.001, batch_size=32, epochs=150, patience=5, validation_split=0.2):
        """
        Trains the MLP model with early stopping and saves it to the specified directory.
        
        Parameters:
            X_train (numpy array): Training feature data.
            y_train (numpy array): Labels for training data.
            model_dir (str): Directory to save the trained model.
            lr (float): Learning rate for the optimizer. Default is 0.001.
            batch_size (int): Number of samples per batch. Default is 32.
            epochs (int): Maximum number of training epochs. Default is 150.
            patience (int): Early stopping patience. Default is 5.
            validation_split (float): Fraction of training data for validation. Default is 0.2.

        Returns:
            history (tf.keras.callbacks.History): Training history containing loss and accuracy metrics.
        """
        
        # Build the model
        self.model_instance = self.build_model()
        print(self.model_instance.summary())

        # Compile the model
        self.model_instance.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',  # Binary classification
            metrics=['accuracy']
        )

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )

        # Model training with validation split
        history = self.model_instance.fit(
            x=X_train,
            y=y_train,
            validation_split=validation_split,  # Automatically use a fraction of data for validation
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=2
        )

        # Save trained model
        self.save_model(model_dir)

        return history

    def evaluate_model(self, X_test, y_test, output_dir="./Models_Performance/MLP/"):
        """
        Evaluates the trained MLP model on test data, saves performance metrics and confusion matrix.
        
        Parameters:
            X_test (numpy array): Test feature data.
            y_test (numpy array): Labels for test data.
            output_dir (str): Directory to save evaluation results and confusion matrix. Default is "./Models_Performance/MLP/".

        Returns:
            test_loss (float): Loss on test data.
            test_accuracy (float): Accuracy on test data.
        """
        # Helper function to plot and save confusion matrix
        def plot_confusion_matrix(y_true, y_pred, output_path):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Confusion Matrix (MLP)')
            plt.savefig(output_path)
            plt.close()

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not self.model_instance:
            print("Model not found. Train or load the model first.")
            return None

        # Evaluate the model
        test_loss, test_accuracy = self.model_instance.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Get model predictions
        predictions = self.model_instance.predict(X_test)
        predicted_labels = (predictions > 0.5).astype(int)  # Convert probabilities to 0/1 labels

        # Accuracy
        accuracy = accuracy_score(y_test, predicted_labels)

        # Precision and Recall for both classes (binary classification)
        precision_high = precision_score(y_test, predicted_labels)
        recall_high = recall_score(y_test, predicted_labels)
        precision_low = precision_score(y_test, predicted_labels, pos_label=0)
        recall_low = recall_score(y_test, predicted_labels, pos_label=0)

        # Classification Report and Confusion Matrix
        report = classification_report(y_test, predicted_labels)
        cm = confusion_matrix(y_test, predicted_labels)

        # Save the classification report to a file
        report_filepath = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_filepath, "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision (Class 1): {precision_high}, Recall (Class 1): {recall_high}\n")
            f.write(f"Precision (Class 0): {precision_low}, Recall (Class 0): {recall_low}\n\n")
            f.write(f"Classification Report:\n{report}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
        print(f"Classification report saved to: {report_filepath}")

        # Plot and save the confusion matrix
        plot_confusion_matrix(y_test, predicted_labels, os.path.join(output_dir, "confusion_matrix.png"))
        print(f"Confusion matrix saved to: {output_dir}/confusion_matrix.png")

        return test_loss, test_accuracy

    def save_model(self, model_filepath):
        """
        Saves the trained model to the specified file path.
        
        Parameters:
            model_filepath (str): Path to save the model.
        """
        if self.model_instance:
            self.model_instance.save(model_filepath)
            print(f"Model saved to: {model_filepath}")
        else:
            print("No model found to save.")

    def load_model(self, model_filepath):
        """
        Loads a pre-trained model from the specified file path.
        
        Parameters:
            model_filepath (str): Path to the model file.

        Returns:
            model_instance (tf.keras.Model): Loaded model instance.
        """
        self.model_instance = tf.keras.models.load_model(model_filepath)
        print(f"Model loaded from: {model_filepath}")
        return self.model_instance