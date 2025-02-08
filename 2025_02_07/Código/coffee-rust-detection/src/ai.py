# Import necessary libraries
import tensorflow as tf  # TensorFlow for building and training the model
from tensorflow.keras.applications import MobileNetV2  # Pre-trained MobileNetV2 model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Layers for building the model
from tensorflow.keras.models import Model  # Model class for creating the model

# Function to create the model
def create_model(num_classes):
    # Load the MobileNetV2 model pre-trained on ImageNet, excluding the top layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add a global average pooling layer to reduce the spatial dimensions
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add a dense layer with 1024 units and ReLU activation
    x = Dense(1024, activation='relu')(x)
    
    # Add the output layer with 'num_classes' units and softmax activation
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model by specifying the inputs and outputs
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model  # Return the created model

# Function to compile the model
def compile_model(model):
    # Compile the model with Adam optimizer, categorical cross-entropy loss, and accuracy metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Main function to create, compile, and print the model summary
def main():
    num_classes = 2  # Number of classes: Healthy and Infected
    model = create_model(num_classes)  # Create the model
    compile_model(model)  # Compile the model
    print(model.summary())  # Print the model summary

# If the script is run directly, execute the main function
if __name__ == "__main__":
    main()