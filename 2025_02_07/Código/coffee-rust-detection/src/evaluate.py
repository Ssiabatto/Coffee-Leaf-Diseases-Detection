import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def evaluate_model(model_path, data_dir, batch_size=32):
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    model_path = os.path.join('path_to_your_model', 'model.h5')  # Update with your model path
    data_dir = os.path.join('..', 'data')  # Adjust path as necessary
    evaluate_model(model_path, data_dir)