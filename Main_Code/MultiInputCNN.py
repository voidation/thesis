import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
This is the MultiInputCNN model class, this is the model that was implemented that took windowed
raw EEG and NIRS samples separately. More details on the windowing and the model are present in
7.1 Isolating Data Samples and in Appendix A of the thesis for this code.
'''

class MultiInputCNN:

    def __init__(self, eeg_input_dims, nirs_input_dims):
        """
        Initializes the MultiInputCNN with dimensions for EEG and NIRS data inputs.
        
        Parameters:
            eeg_input_dims (tuple): Input dimensions for EEG data.
            nirs_input_dims (tuple): Input dimensions for NIRS data.
        """
        self.eeg_input_dims = eeg_input_dims
        self.nirs_input_dims = nirs_input_dims
        self.model_instance = None # For holding the trained model

    def model(self):
        """
        Builds the CNN model for multi-input (EEG and NIRS) data.

        Returns:
            model (tf.keras.Model): Compiled multi-input CNN model.
        """
        # EEG input
        eeg_input = layers.Input(shape=self.eeg_input_dims, name="eeg_input")

        # EEG branch
        h_eeg = layers.Conv2D(
            filters=64, kernel_size=(32, 1), activation="relu"
        )(eeg_input)
        h_eeg = layers.MaxPooling2D(
            pool_size=(10, 1), strides=(5, 1)
        )(h_eeg)
        h_eeg = layers.Flatten()(h_eeg)

        # NIRS input
        nirs_input = layers.Input(shape=self.nirs_input_dims, name="nirs_input")

        # NIRS branch
        h_nirs = layers.Conv2D(
            filters=128, kernel_size=(4, 1), activation="relu"
        )(nirs_input)
        h_nirs = layers.MaxPooling2D(
            pool_size=(4, 1), strides=(2, 1)
        )(h_nirs)
        h_nirs = layers.Flatten()(h_nirs)

        # Concatenate both branches
        concatenated = layers.Concatenate()([h_eeg, h_nirs])

        # Fully connected layer
        fc = layers.Dense(2, activation="relu")(concatenated)

        # Output layer with softmax for classification
        output = layers.Dense(2, activation="softmax", name="output")(fc)

        # Create model
        model = models.Model(inputs=[eeg_input, nirs_input], outputs=output)

        return model
    
    def train(self, train_data, val_data, model_dir, lr=0.001, batch_size=32, epochs=10, patience=10, gpu_count=1):
        """
        Trains the MultiInputCNN model with EEG and NIRS data.

        Parameters:
            train_data (tuple): Training data tuple (eeg_train_data, nirs_train_data, labels).
            val_data (tuple): Validation data tuple (eeg_val_data, nirs_val_data, labels).
            model_dir (str): Directory to save the trained model.
            lr (float): Learning rate for the optimizer. Default is 0.001.
            batch_size (int): Number of samples per batch. Default is 32.
            epochs (int): Number of training epochs. Default is 10.
            patience (int): Early stopping patience. Default is 10.
            gpu_count (int): Number of GPUs to use, if applicable.

        Returns:
            history (tf.keras.callbacks.History): Training history with loss and accuracy metrics.
        """
        # Set up learning rate scheduler
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            decay_rate=0.9
        )

        # Build the model
        self.model_instance = self.model()
        print(self.model_instance.summary())

        # Model preparation
        self.model_instance.compile(
            optimizer=optimizers.Adam(learning_rate=lr_schedule),
            loss="sparse_categorical_crossentropy",  # Binary classification
            metrics=["accuracy"]
        )

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # You can also monitor 'val_accuracy'
            patience=patience,  # Number of epochs to wait before stopping
            restore_best_weights=True  # Restore the weights of the best epoch
        )

        # Compute class weights
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(train_data[1]),  # unique classes in the dataset
            y=train_data[1]                    # actual labels
        )
        class_weights_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weights_dict}")

        # Model training
        history = self.model_instance.fit(
            x=train_data[0],  # EEG and NIRS data as a tuple (eeg_train_data, nirs_train_data)
            y=train_data[1],  # Labels for training
            validation_data=(val_data[0], val_data[1]),  # EEG, NIRS validation data and labels
            batch_size=batch_size,
            epochs=epochs,
            #callbacks=[early_stopping],
            class_weight=class_weights_dict
        )

        # Save trained model
        self.model_instance.save(model_dir)

        return history
    
    def evaluate(self, test_data):
        """
        Evaluates the trained model on test data and plots a confusion matrix.

        Parameters:
            test_data (tuple): Test data tuple (eeg_test, nirs_test, labels).

        Returns:
            None: Prints evaluation metrics and displays a confusion matrix.
        """
        # Plot confusion matrix helper function
        def plot_confusion_matrix(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Confusion Matrix')
            plt.show()

        if not self.model_instance:
            print("Model not found. Train the model first.")
            return None
        
        # Evaluate on the test set
        test_loss, test_accuracy = self.model_instance.evaluate(
            x=test_data[0],  # ([eeg_test, nirs_test])
            y=test_data[1]   # (labels_test)
        )

        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Get model predictions (probabilities)
        predictions = self.model_instance.predict(test_data[0])
        
        # Convert predictions to class labels (0 for low, 1 for high)
        predicted_labels = np.argmax(predictions, axis=1)

        # True labels
        true_labels = test_data[1]

        # Compute precision and recall for class "1" (highs) and class "0" (lows)
        precision_high = precision_score(true_labels, predicted_labels, pos_label=1)
        recall_high = recall_score(true_labels, predicted_labels, pos_label=1)

        precision_low = precision_score(true_labels, predicted_labels, pos_label=0)
        recall_low = recall_score(true_labels, predicted_labels, pos_label=0)

        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Precision (Highs): {precision_high}")
        print(f"Recall (Highs): {recall_high}")
        print(f"Precision (Lows): {precision_low}")
        print(f"Recall (Lows): {recall_low}")

        plot_confusion_matrix(test_data[1], predicted_labels)