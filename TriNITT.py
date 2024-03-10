import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import save_model

import cv2

# Data preprocessing
def load_dataset(images_folder, annotations_folder, target_size=(224, 224)):
    image_files = os.listdir(images_folder)
    annotation_files = os.listdir(annotations_folder)
    
    dataset = []
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        annotation_file = os.path.splitext(img_file)[0] + '.xml'  # Assuming annotations are XML files with the same name as images
        annotation_path = os.path.join(annotations_folder, annotation_file)
        
        if annotation_file in annotation_files and os.path.isfile(annotation_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)  # Resize the image to a fixed size
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            # Extract annotations
            annotations = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                annotations.append({'name': name})  # Only extracting the 'name' for classification
            
            # Determine classification label based on annotations
            # Assuming 'D00' represents undamaged and other values represent damaged
            classification_label = 0 if 'D00' in [ann['name'] for ann in annotations] else 1
            
            dataset.append({'image': img, 'classification_label': classification_label})
    
    return dataset


# Example usage:
images_folder = 'D:\\RDD2022_Japan\\Japan\\train\\images'
annotations_folder = 'D:\\RDD2022_Japan\\Japan\\train\\annotations\\xmls'
dataset = load_dataset(images_folder, annotations_folder)

# Convert dataset to numpy arrays
X = np.array([data['image'] for data in dataset])
y = np.array([data['classification_label'] for data in dataset])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), steps_per_epoch=5)


#saving the model
model.save('D:\\RDD2022_Japan\\Japan\\trained_model.h5')