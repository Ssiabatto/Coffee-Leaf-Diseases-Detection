# Import necessary libraries
import tensorflow as tf  # TensorFlow for building and training the model
from tensorflow.keras.models import load_model  # For loading pre-trained models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation and preprocessing
import numpy as np  # For numerical operations
import os  # For file and directory operations

# Function to evaluate the model
def evaluate_model(model_path, data_dir, batch_size=32):
    # Load the pre-trained model from the specified path
    model = load_model(model_path)

    # Create a data generator for the test dataset
    test_datagen = ImageDataGenerator(rescale=1.0/255)  # Rescale pixel values

    # Create a test data generator
    test_generator = test_datagen.flow_from_directory(
        data_dir,  # Directory containing the test dataset
        target_size=(224, 224),  # Resize images to the specified height and width
        batch_size=batch_size,  # Number of images to process in a batch
        class_mode='binary',  # Binary classification (healthy or infected)
        shuffle=False  # Do not shuffle the data
    )

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")  # Print the test loss
    print(f"Test Accuracy: {accuracy:.4f}")  # Print the test accuracy

# Main block to execute the evaluation
if __name__ == "__main__":
    model_path = os.path.join('path_to_your_model', 'model.h5')  # Update with your model path
    data_dir = os.path.join('..', 'data')  # Adjust path as necessary
    evaluate_model(model_path, data_dir)  # Call the evaluate_model function