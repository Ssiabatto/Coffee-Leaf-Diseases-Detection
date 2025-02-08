# Import necessary libraries
import os  # For file and directory operations
import numpy as np  # For numerical operations
import tensorflow as tf  # TensorFlow for building and training the model
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # For loading and converting images
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []  # List to store images
    for filename in os.listdir(folder):  # Loop through each file in the folder
        img_path = os.path.join(folder, filename)  # Get the full path of the image
        if os.path.isfile(img_path):  # Check if the path is a file
            img = load_img(img_path, target_size=(224, 224))  # Load and resize the image to match MobileNetV2 input
            img_array = img_to_array(img)  # Convert the image to a NumPy array
            images.append(img_array)  # Add the image array to the list
    return np.array(images)  # Return the list of images as a NumPy array

# Function to load the dataset from healthy and infected folders
def load_dataset(healthy_folder, infected_folder):
    healthy_images = load_images_from_folder(healthy_folder)  # Load healthy images
    infected_images = load_images_from_folder(infected_folder)  # Load infected images

    # Create labels: 0 for healthy, 1 for infected
    healthy_labels = np.zeros(len(healthy_images))  # Labels for healthy images
    infected_labels = np.ones(len(infected_images))  # Labels for infected images

    # Combine the images and labels
    images = np.concatenate((healthy_images, infected_images), axis=0)  # Combine healthy and infected images
    labels = np.concatenate((healthy_labels, infected_labels), axis=0)  # Combine healthy and infected labels

    return images, labels  # Return the combined images and labels

# Function to preprocess the data
def preprocess_data(images, labels):
    # Normalize the images
    images = images.astype('float32') / 255.0  # Convert pixel values to float and normalize to [0, 1]
    return images, labels  # Return the normalized images and labels

# Function to split the dataset into training and testing sets
def split_dataset(images, labels, test_size=0.2, random_state=42):
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)  # Split the data