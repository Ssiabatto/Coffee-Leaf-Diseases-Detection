import sys
import os

# Add the project directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import necessary libraries
from tensorflow.keras.models import load_model  # For loading pre-trained models
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)  # For data augmentation and preprocessing
from tensorflow.keras.utils import to_categorical  # For one-hot encoding the labels
import numpy as np  # For numerical operations
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)  # For evaluation metrics
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For heatmap visualization
from src.utils.data_loader import (
    load_dataset,
    preprocess_data,
    split_dataset,
)  # Import data loading and preprocessing functions
import random  # For random sampling
import tkinter as tk
from tkinter import filedialog, simpledialog


# Function to plot images with predictions and actual labels
def plot_images_with_predictions(images, y_true, y_pred, class_names, num_images=10):
    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}")
        plt.axis("off")
    plt.show()


# Function to plot individual images with predictions and actual labels
def plot_individual_images(images, y_true, y_pred, class_names, num_images=5):
    indices = random.sample(range(len(images)), num_images)
    for i in indices:
        plt.figure(figsize=(4, 4))
        plt.imshow(images[i])
        plt.title(f"True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}")
        plt.axis("off")
        plt.show()


# Function to evaluate the model on a random subset of images
def evaluate_random_subset(model, images, labels, num_images=200, batch_size=32):
    indices = random.sample(range(len(images)), num_images)
    subset_images = images[indices]
    subset_labels = labels[indices]

    # One-hot encode the labels
    subset_labels = to_categorical(subset_labels, num_classes=2)

    # Create data generator for the subset
    subset_datagen = ImageDataGenerator()
    subset_generator = subset_datagen.flow(
        subset_images, subset_labels, batch_size=batch_size, shuffle=False
    )

    print("Evaluating the model on the random subset...")
    # Evaluate the model on the subset data
    loss, accuracy = model.evaluate(subset_generator)
    print(f"Subset Test Loss: {loss:.4f}")
    print(f"Subset Test Accuracy: {accuracy:.4f}")

    # Predict the labels for the subset data
    y_pred = model.predict(subset_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(subset_labels, axis=1)

    # Print classification report
    print("Classification Report for Subset:")
    print(
        classification_report(
            y_true, y_pred_classes, target_names=["Healthy", "Infected"]
        )
    )

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Infected"],
        yticklabels=["Healthy", "Infected"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for Subset")
    plt.show()

    # Plot images with predictions and actual labels
    print("Plotting images with predictions and actual labels for subset...")
    plot_images_with_predictions(
        subset_images, y_true, y_pred_classes, ["Healthy", "Infected"], num_images=10
    )

    # Plot individual images with predictions and actual labels
    print("Plotting individual images with predictions and actual labels for subset...")
    plot_individual_images(
        subset_images, y_true, y_pred_classes, ["Healthy", "Infected"], num_images=5
    )


# Function to predict a single image
def predict_single_image(model, image_path, class_names):
    print(f"Predicting for image: {image_path}")
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    print(f"Predicted: {class_names[predicted_class]}")

    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis("off")
    plt.show()


# Function to evaluate the model
def evaluate_model(model_path, healthy_folder, infected_folder, batch_size=32):
    print("Loading the pre-trained model...")
    # Load the pre-trained model from the specified path
    model = load_model(model_path)

    print("Loading and preprocessing the dataset...")
    # Load and preprocess the dataset
    images, labels = load_dataset(healthy_folder, infected_folder)
    images, labels = preprocess_data(images, labels)

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=2)

    print("Splitting the dataset into training, validation, and test sets...")
    # Split the dataset into training, validation, and test sets
    train_images, val_images, test_images, train_labels, val_labels, test_labels = (
        split_dataset(images, labels)
    )

    print("Creating data generator for the validation dataset...")
    # Create a data generator for the validation dataset
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(
        val_images, val_labels, batch_size=batch_size, shuffle=False
    )

    print("Evaluating the model on the validation data...")
    # Evaluate the model on the validation data
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Predict the labels for the validation data
    y_pred = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(val_labels, axis=1)

    # Print classification report
    print("Classification Report for Validation Set:")
    print(
        classification_report(
            y_true, y_pred_classes, target_names=["Healthy", "Infected"]
        )
    )

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Infected"],
        yticklabels=["Healthy", "Infected"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for Validation Set")
    plt.show()

    # Plot images with predictions and actual labels
    print("Plotting images with predictions and actual labels for validation set...")
    plot_images_with_predictions(
        val_images, y_true, y_pred_classes, ["Healthy", "Infected"], num_images=10
    )


# Main block to execute the evaluation
if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "coffee_rust_model.h5")
    healthy_folder = os.path.join(os.path.dirname(__file__), "..", "data", "healthy")
    infected_folder = os.path.join(os.path.dirname(__file__), "..", "data", "infected")

    # Create a simple GUI for selecting the functionality
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    choice = simpledialog.askinteger(
        "Input", "Enter 1 to evaluate the model or 2 to predict a single image:"
    )

    if choice == 1:
        evaluate_model(model_path, healthy_folder, infected_folder)
    elif choice == 2:
        initial_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        single_image_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select an image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
        )
        if single_image_path:
            model = load_model(model_path)
            predict_single_image(model, single_image_path, ["Healthy", "Infected"])
        else:
            print("No image selected.")
    else:
        print("Invalid choice.")
