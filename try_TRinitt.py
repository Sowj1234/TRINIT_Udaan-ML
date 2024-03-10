import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np

#data preprocessing
def load_dataset(images_folder, annotations_folder):
    image_files = os.listdir(images_folder)
    annotation_files = os.listdir(annotations_folder)
    
    dataset = []
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        annotation_file = os.path.splitext(img_file)[0] + '.xml'  # Assuming annotations are XML files with the same name as images
        annotation_path = os.path.join(annotations_folder, annotation_file)
        
        #print("Image Path:", img_path)
        #print("Annotation Path:", annotation_path)
        
        if annotation_file in annotation_files and os.path.isfile(annotation_path):
            img = cv2.imread(img_path)
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            # Extract annotations
            annotations = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                xmin = int(obj.find('bndbox').find('xmin').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymax = int(obj.find('bndbox').find('ymax').text)
                annotations.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
            
            dataset.append({'image': img, 'annotations': annotations})
    
    return dataset

# Example usage:
images_folder = 'D:\\RDD2022_Japan\\Japan\\train\\images'
annotations_folder = 'D:\\RDD2022_Japan\\Japan\\train\\annotations\\xmls'
dataset = load_dataset(images_folder, annotations_folder)

if not dataset:
    print("The dataset is empty")
else:
    print("The dataset is not empty")
for data in dataset:
    image = data['image']
    annotations = data['annotations']
    # Draw bounding boxes on the image
    for annotation in annotations:
        bbox = annotation['bbox']
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw green bounding box

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)