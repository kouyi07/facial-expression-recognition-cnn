# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:29:10 2024

@author: User
"""
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def plot_model_history(model_history, plot_name):
    """Plot accuracy and loss curves."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    
    fig.savefig(plot_name)
    plt.show()

def create_model(lr=0.0001):
    """Define and compile the model."""
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    return model

def train_model(batch_size = 32, num_epoch = 400, train_dir = 'data/train', val_dir = 'data/test',
                num_train = 28709, num_val = 7178):
    """Train the model with the given hyperparameters."""
    
    # Auto-generated, accurate filenames
    model_name = f'emotion_recognition_model{batch_size}-{num_epoch}.keras'
    plot_name = f'emotion{batch_size}-{num_epoch}.png'
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,         # ±10° rotations
        width_shift_range=0.1,     # ±10% horizontal shifting
        height_shift_range=0.1,    # ±10% vertical shifting
        zoom_range=0.1,            # ±10% zoom
        horizontal_flip=True       # Horizontal mirroring
    )
    # train_datagen = ImageDataGenerator(rescale=1. / 255)

    
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    model = create_model()

    start_time = time.time()
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    elapsed_time = (time.time() - start_time) / 60
    print(f"Training completed in: {elapsed_time:.2f} minutes")

    plot_model_history(model_info, plot_name)
    model.save(model_name)
    print(f"Model saved to {model_name}")

if __name__ == "__main__":
    train_model()

