import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:\\RDD2022_Japan\\Japan\\trained_model.h5')

# Load and preprocess the image you want to classify
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Classify the image
def classify_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    print(prediction)
    if prediction < 0.7:
        return "Not Damaged"
    else:
        return "Damaged"

# Example usage:
image_path = 'D:\\RDD2022_Japan\\Japan\\test\\images\\Japan_000228.jpg'
classification = classify_image(image_path)
print("Classification:", classification)
