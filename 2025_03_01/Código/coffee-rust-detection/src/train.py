import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # For data augmentation and preprocessing
from tensorflow.keras.utils import to_categorical  # For one-hot encoding the labels
from tensorflow.keras.models import load_model  # For loading pre-trained models
from src.ai import create_model  # Import the create_model function from ai.py
from src.utils.data_loader import (
    load_dataset,
    preprocess_data,
    split_dataset,
)  # Import data loading and preprocessing functions

# Set parameters
batch_size = 32  # Number of images to process in a batch
img_height = 224  # Height of the input images
img_width = 224  # Width of the input images
epochs = 20  # Number of epochs to train the model (increased from 10 to 20)
max_images = 500  # Maximum number of images to load from each class (increased from 100 to 500)

print("Loading and preprocessing the dataset...")

# Load and preprocess the dataset
healthy_folder = os.path.join(
    os.path.dirname(__file__), "..", "data", "healthy"
)  # Directory containing healthy images
infected_folder = os.path.join(
    os.path.dirname(__file__), "..", "data", "infected"
)  # Directory containing infected images
images, labels = load_dataset(
    healthy_folder, infected_folder, max_images=max_images
)  # Load the dataset
images, labels = preprocess_data(images, labels)  # Preprocess the dataset

# Print class distribution
print(f"Number of healthy images: {np.sum(labels == 0)}")
print(f"Number of infected images: {np.sum(labels == 1)}")

# One-hot encode the labels
labels = to_categorical(labels, num_classes=2)

print("Splitting the dataset into training, validation, and test sets...")

# Split the dataset into training, validation, and test sets
train_images, val_images, test_images, train_labels, val_labels, test_labels = (
    split_dataset(images, labels)
)  # Split the dataset

print("Creating data generators...")

# Create data generators
train_datagen = ImageDataGenerator()  # Create a training data generator
val_datagen = ImageDataGenerator()  # Create a validation data generator

train_generator = train_datagen.flow(
    train_images, train_labels, batch_size=batch_size
)  # Create a training data generator
val_generator = val_datagen.flow(
    val_images, val_labels, batch_size=batch_size
)  # Create a validation data generator

print("Loading and compiling the model...")

# Load the pre-trained model if it exists, otherwise create a new one
model_path = os.path.join(os.path.dirname(__file__), "coffee_rust_model.h5")
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
    # Optionally, freeze some layers
    for layer in model.layers[:-4]:  # Freeze all layers except the last 4
        layer.trainable = False
else:
    print("Creating a new model...")
    model = create_model(
        num_classes=2
    )  # Create the model with 2 output classes (healthy and infected)

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)  # Compile the model with Adam optimizer and binary cross-entropy loss

try:
    print("Training the model...")
    # Train the model
    history = model.fit(
        train_generator, validation_data=val_generator, epochs=epochs
    )  # Train the model

    print("Saving the model...")
    # Save the model
    model.save(model_path)  # Save the trained model to a file

    print("Model was successfully generated and saved.")

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

except Exception as e:
    print(f"An error occurred during model training or saving: {e}")
