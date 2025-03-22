"""
Adapted from: https://keras.io/examples/timeseries/timeseries_classification_transformer/ 
"""
import tensorflow as tf
import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class TransformerModel:
    """
    A Transformer-based model for time series classification, with EEG and NIRS data compatibility.
    Includes K-fold cross-validation training and evaluation.

    Attributes:
        input_shape (int): The number of input features.
        model_instance (tensorflow.keras.Model): Holds the built and compiled model instance.
    """
    def __init__(self, input_shape):
        """
        Initializes the TransformerModel with the input shape.
        
        Parameters:
            input_shape (int): The number of features in the input data.
        """

        self.input_shape = input_shape
        self.model_instance = None  # For holding the trained model

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        Defines a Transformer encoder block with multi-head self-attention and feed-forward layers.
        
        Parameters:
            inputs (tf.Tensor): Input tensor.
            head_size (int): Size of the attention heads.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimensionality of the feed-forward layer.
            dropout (float): Dropout rate for regularization.
        
        Returns:
            tf.Tensor: Output tensor after applying the encoder block.
        """
        # Attention and Normalisation
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    
    def build_model(self,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
    ):
        """
        Builds a Transformer-based model with multi-layer perceptron (MLP) classifier layers.
        
        Parameters:
            head_size (int): Size of the attention heads.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimensionality of the feed-forward layer.
            num_transformer_blocks (int): Number of Transformer encoder blocks.
            mlp_units (list of int): Number of units in each MLP layer.
            dropout (float): Dropout rate for the Transformer encoder.
            mlp_dropout (float): Dropout rate for the MLP layers.
        
        Returns:
            keras.Model: Compiled Transformer model.
        """

        # Set the input shape to (1, input_shape) for Conv1D compatibility
        print(f"Input shape for the model: (1, {self.input_shape})")
        inputs = keras.Input(shape=(1, self.input_shape))
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(2, activation="softmax")(x)
        return keras.Model(inputs, outputs)

    def train_model(self, X_train, y_train, model_dir, lr=1e-4, batch_size=64, epochs=150, patience=10, validation_split=0.2):
        """
        Trains the transformer model using cross-validation and early stopping.

        Parameters:
            X_train (numpy.ndarray): Training feature data.
            y_train (numpy.ndarray): Training labels.
            model_dir (str): Directory to save the trained model.
            lr (float): Learning rate.
            batch_size (int): Training batch size.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
            validation_split (float): Fraction of data for validation.

        Returns:
            history: Training history.
        """
        # Reshape X_train to (num_samples, 1, num_features)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        # Initialise KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_no = 1
        for train_index, val_index in kf.split(X_train):
            print(f"Training on fold {fold_no}...")
            # Splitting the data into training and validation sets for this fold
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            # Build the model
            self.model_instance = self.build_model(
                head_size=256,
                num_heads=4,
                ff_dim=4,
                num_transformer_blocks=4,
                mlp_units=[128],
                mlp_dropout=0.4,
                dropout=0.25,
            )
            print(self.model_instance.summary())

            # Compile the model
            self.model_instance.compile(
                optimizer=optimizers.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',  # Binary classification
                metrics=['sparse_categorical_accuracy']
            )

            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )

            # Learning rate scheduler
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5,  # Factor by which the learning rate will be reduced
                patience=3,  # Number of epochs with no improvement after which learning rate will be reduced
                min_lr=1e-6,  # Lower bound on the learning rate
                verbose=1
            )

            # Train the model
            history = self.model_instance.fit(
                x=X_fold_train,
                y=y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

            # Evaluate the model on the validation set for this fold
            val_loss, val_accuracy = self.model_instance.evaluate(X_fold_val, y_fold_val)
            print(f"Fold {fold_no} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            fold_no += 1

        # Save trained model
        self.save_model(model_dir)

        return history

    def evaluate_model(self, X_test, y_test, output_dir="./Models_Performance/Transformer/"):
        """
        Evaluates the trained model and saves metrics including accuracy, precision, recall, and confusion matrix.

        Parameters:
            X_test (np.array): Test features.
            y_test (np.array): Test labels.
            output_dir (str): Directory to save evaluation reports and plots.
        
        Returns:
            tuple: test_loss, test_accuracy
        """
        # Reshape X_train to (num_samples, 1, num_features)
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Helper function to plot and save confusion matrix
        def plot_confusion_matrix(y_true, y_pred, output_path):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Confusion Matrix (Transformer)')
            plt.savefig(output_path)
            plt.close()

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not self.model_instance:
            print("Model not found. Train or load the model first.")
            return None

        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model_instance.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Get model predictions (probabilities) and convert to class labels (using argmax for multi-class classification)
        predictions = self.model_instance.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)  # Get the index of the highest probability

        # Accuracy
        accuracy = accuracy_score(y_test, predicted_labels)

        # Precision and Recall for both classes (multi-class classification)
        precision_high = precision_score(y_test, predicted_labels, average='binary', pos_label=1)
        recall_high = recall_score(y_test, predicted_labels, average='binary', pos_label=1)
        precision_low = precision_score(y_test, predicted_labels, average='binary', pos_label=0)
        recall_low = recall_score(y_test, predicted_labels, average='binary', pos_label=0)

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
        """Saves the trained model to the specified file path."""
        if self.model_instance:
            self.model_instance.save(model_filepath)
            print(f"Model saved to: {model_filepath}")
        else:
            print("No model found to save.")

    def load_model(self, model_filepath):
        """Loads a pre-trained model from the specified file path."""
        self.model_instance = tf.keras.models.load_model(model_filepath)
        print(f"Model loaded from: {model_filepath}")
        return self.model_instance