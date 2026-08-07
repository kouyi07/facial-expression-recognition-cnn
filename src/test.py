# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:35:34 2024

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Reduce TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# def plot_confusion_matrix(cm, class_names):
#     """Plot the confusion matrix."""
#     # Calculate percentage
#     cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
#     cm_percentage = np.round(cm_percentage, 1)  # Round to 1 decimal place
    
#     # Create formatted annotations with percentage symbol
#     annotations = np.core.defchararray.add(cm_percentage.astype(str), '%')
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm_percentage, annot=annotations, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.title('Confusion Matrix',fontsize=14)
#     plt.ylabel('Actual', fontsize=13)
#     plt.xlabel('Predicted', fontsize=13)
#     plt.tight_layout()
#     plt.savefig('confusion_matrix32-400.png')
#     plt.show()

def plot_confusion_matrix(cm, class_names):
    """Plot the confusion matrix with both raw values and percentage annotations."""
    # Calculate percentage
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percentage = np.round(cm_percentage, 1)  # Round to 1 decimal place

    # Create formatted annotations with both value and percentage
    annotations = []
    for i in range(cm.shape[0]):
        row_annotations = []
        for j in range(cm.shape[1]):
            value = cm[i, j]
            percentage = cm_percentage[i, j]
            annotation = f"{value}\n({percentage}%)"
            row_annotations.append(annotation)
        annotations.append(row_annotations)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=annotations, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    # Titles and axis labels with larger font size
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Predicted', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix32-400.png')
    plt.show()


def evaluate_model():
    """Load and evaluate the trained model."""
    test_dir = 'data/test'
    batch_size = 64

    ''' Custom '''
    # Prepare test data generator 
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)
    
    # '''VGG16'''
    # # Prepare test data generator
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_generator = test_datagen.flow_from_directory(
    # test_dir,
    # target_size=(224, 224),  # Resize to 224x224
    # batch_size=batch_size,
    # color_mode="rgb",  # Use RGB images
    # class_mode='categorical',
    # shuffle=False)

    # Load the trained model
    model = load_model('emotion_recognition_model32-400.keras')
    print("Model loaded.")

    # Make predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Confusion matrix and classification report
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(true_classes, predicted_classes, target_names=class_labels))

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_labels)

if __name__ == "__main__":
    evaluate_model()


