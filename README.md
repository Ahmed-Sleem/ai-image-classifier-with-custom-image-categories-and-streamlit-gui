# ai-image-classifier-with-custom-image-categories-and-streamlit-gui
Image Classifier using Streamlit and TensorFlow: An interactive web application for image classification, powered by Streamlit and TensorFlow. Upload images, train the model with custom classes, and perform real-time classification with ease.




This project is a user-friendly image classification application built using Streamlit and TensorFlow. The application allows users to classify images into custom categories by uploading them and training the classifier with the provided dataset.

## How It Works:
- **Upload Images**: Users can upload images directly through the application's interface.
- **Training Model**: After uploading images, users can select the number of classes and provide class names. The application then trains a convolutional neural network (CNN) model using TensorFlow.
- **Real-time Classification**: Once the model is trained, users can upload an image for real-time classification. The application displays the predicted class along with the confidence level.

## Features:
- **User-friendly Interface**: The application provides an intuitive interface with sidebar controls for easy navigation.
- **Customizable Classes**: Users can define custom classes for image classification.
- **Interactive Training History**: The application visualizes the training history of the model with accuracy charts.
- **Model Loading**: Users can load pre-trained models for classification tasks.
- **Reset Functionality**: Users can reset the classifier to upload new images for classification.

## How to Use:
1. Clone the repository to your local machine.
2. Install the required dependencies specified in the `requirements.txt` file.
3. Run the Streamlit application using the command `streamlit run main.py`.
4. Upload images for each class, specify the number of classes, and train the model.
5. Once the model is trained, upload an image for real-time classification and view the prediction results.

This project serves as a practical example of building machine learning applications with Streamlit and TensorFlow, making it accessible for both beginners and experienced developers interested in image classification tasks.
