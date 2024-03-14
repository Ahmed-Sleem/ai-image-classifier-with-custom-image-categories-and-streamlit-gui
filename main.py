import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import numpy as np
import os
import uuid

# SessionState class definition
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def get(**kwargs):
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return SessionState(**kwargs)
    
# Function to build the model
def build_model(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Dynamic number of classes
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, train_generator, epochs):
    history = model.fit(train_generator, epochs=epochs)
    return history

# Function to classify image using the trained model
def classify_image(image, model, class_names):
    if not class_names:
        return "No classes", 0.0
    
    image = image.resize((150, 150))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255
    image_array = image_array[:, :, :, :3]  # Keep only the first 3 channels
    prediction = model.predict(image_array)
    
    # Get the class index with the highest probability
    predicted_class_index = np.argmax(prediction)
    
    # Get the class name
    predicted_class_name = class_names[predicted_class_index]
    
    # Get the percentage probability
    percentage_probability = prediction[0][predicted_class_index] * 100
    
    return predicted_class_name, percentage_probability

# Function to reset the classifier
def reset_classifier(inference_placeholder):
    # Clear uploaded images
    session_state.uploaded_image = None
    # Clear classification results
    session_state.prediction_result = None
    # Clear the content of inference_placeholder
    inference_placeholder.write("")
    # Reset the state of file uploader by changing its key
    st.file_uploader(label="Upload an image for classification", key=uuid.uuid4())
    
# Main function
def main():
    st.title("Image Classifier")
    
    # Placeholder for inference tab
    inference_placeholder = st.empty()
    
    # Sidebar for selecting number of classes
    num_classes = st.sidebar.number_input("Select number of classes:", min_value=2, max_value=10, value=2)
    
    # Build the model
    model = build_model(num_classes)
    
    # Training section
    st.sidebar.title("Train Model")
    
    # Upload images for each class
    st.sidebar.write(f"Upload images for {num_classes} classes:")
    class_data = {}
    for i in range(num_classes):
        class_name = st.sidebar.text_input(f"Class {i+1} Name:")
        uploaded_files = st.sidebar.file_uploader(f"Upload images for {class_name}:", key=f"file_uploader_{i}", accept_multiple_files=True)
        class_data[class_name] = uploaded_files
    
    if st.sidebar.button("Train Model"):
        # Train the model
        st.sidebar.write("Training the model...")
        # Data preprocessing
        # Load images using ImageDataGenerator
        train_datagen = ImageDataGenerator(rescale=1.0/255)
        train_generator = train_datagen.flow_from_directory(
            'data/',
            target_size=(150, 150),
            batch_size=20,
            class_mode='categorical'
        )
        # Train the model
        history = train_model(model, train_generator, epochs=25)
        st.sidebar.write("Training complete!")
        
        # Display training history
        st.sidebar.subheader("Training History")
        st.sidebar.line_chart(history.history['accuracy'])
        
        # Show inference tab
        inference_placeholder.text("")
        inference_placeholder.write("## Inference")

    # Inference section
    image_file = inference_placeholder.file_uploader("Upload an image for classification:", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        session_state.uploaded_image = Image.open(image_file)
        inference_placeholder.image(session_state.uploaded_image, caption='Uploaded Image', use_column_width=True)
        
        if inference_placeholder.button("Classify"):
            # Classify the image
            class_names = list(class_data.keys())  # Get the list of class names
            predicted_class_name, percentage_probability = classify_image(session_state.uploaded_image, model, class_names)
            session_state.prediction_result = (predicted_class_name, percentage_probability)
            # Display prediction
            inference_placeholder.write(f"Prediction: {predicted_class_name} (Confidence: {percentage_probability:.2f}%)")

    # Load button
    if st.sidebar.button("Load"):
        model_file = st.sidebar.file_uploader("Upload a trained model file:", type=["h5"])
        if model_file is not None:
            model = tf.keras.models.load_model(model_file)

    # Reset button
    if st.button("Reset"):
        reset_classifier(inference_placeholder)
        st.write("Classifier has been reset.")
            

# Entry point of the script
if __name__ == "__main__":
    session_state = SessionState.get(uploaded_image=None, prediction_result=None)
    main()
