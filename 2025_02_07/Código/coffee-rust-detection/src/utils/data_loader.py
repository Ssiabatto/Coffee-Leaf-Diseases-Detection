import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = load_img(img_path, target_size=(224, 224))  # Resize to match MobileNetV2 input
            img_array = img_to_array(img)
            images.append(img_array)
    return np.array(images)

def load_dataset(healthy_folder, infected_folder):
    healthy_images = load_images_from_folder(healthy_folder)
    infected_images = load_images_from_folder(infected_folder)

    # Create labels: 0 for healthy, 1 for infected
    healthy_labels = np.zeros(len(healthy_images))
    infected_labels = np.ones(len(infected_images))

    # Combine the images and labels
    images = np.concatenate((healthy_images, infected_images), axis=0)
    labels = np.concatenate((healthy_labels, infected_labels), axis=0)

    return images, labels

def preprocess_data(images, labels):
    # Normalize the images
    images = images.astype('float32') / 255.0
    return images, labels

def split_dataset(images, labels, test_size=0.2, random_state=42):
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)