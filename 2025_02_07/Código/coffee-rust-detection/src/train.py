import tensorflow as tf  # TensorFlow for building and training the model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation and preprocessing
from tensorflow.keras.models import load_model  # For loading pre-trained models
from src.ai import create_model  # Import the create_model function from ai.py

# Set parameters
batch_size = 32  # Number of images to process in a batch
img_height = 224  # Height of the input images
img_width = 224  # Width of the input images
epochs = 10  # Number of epochs to train the model

# Load the dataset
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Rescale pixel values and set validation split

# Create a training data generator
train_generator = train_datagen.flow_from_directory(
    'data/',  # Directory containing the dataset
    target_size=(img_height, img_width),  # Resize images to the specified height and width
    batch_size=batch_size,  # Number of images to process in a batch
    class_mode='binary',  # Binary classification (healthy or infected)
    subset='training'  # Use the training subset
)

# Create a validation data generator
validation_generator = train_datagen.flow_from_directory(
    'data/',  # Directory containing the dataset
    target_size=(img_height, img_width),  # Resize images to the specified height and width
    batch_size=batch_size,  # Number of images to process in a batch
    class_mode='binary',  # Binary classification (healthy or infected)
    subset='validation'  # Use the validation subset
)

# Create and compile the model
model = create_model(num_classes=2)  # Create the model with 2 output classes (healthy and infected)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model with Adam optimizer and binary cross-entropy loss

# Train the model
model.fit(
    train_generator,  # Training data generator
    validation_data=validation_generator,  # Validation data generator
    epochs=epochs  # Number of epochs to train the model
)

# Save the model
model.save('coffee_rust_model.h5')  # Save the trained model to a file