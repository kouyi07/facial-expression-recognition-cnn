# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:19:09 2024

@author: User
"""

import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt
import sys

# Initialize the main application window
class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FER")
        self.root.geometry("600x550")

        # Preload the model
        self.model = self.load_pretrained_model()
        self.img_path = None
        self.prediction_results = None  # Store prediction results

        # UI Elements
        self.label = Label(root, text="Facial Expression Recognition", font=("Arial", 16))
        self.label.pack(pady=10)

        self.upload_image_button = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_image_button.pack(pady=10)

        # Image display area
        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        # Result label (for predicted emotion)
        self.result_label = Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Time display area
        self.time_label = Label(root, text="", font=("Arial", 12))
        self.time_label.pack(pady=10)

        self.show_percentage_button = Button(root, text="Show percentages", command=self.show_bar_graph)
        self.show_percentage_button.pack(pady=10)
        self.show_percentage_button.pack_forget()  # Hide the button initially

        # Exit button at the bottom left corner
        self.exit_button = Button(root, text="Exit", command=self.exit_app, width=5)
        self.exit_button.place(x=530, y=510)  # Position at bottom left corner

    def load_pretrained_model(self):
        try:
            # Load the pre-trained model here
            model_path = "emotion_recognition_model32-400.keras"  # Replace with the correct model file path
            model = load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit()

    def upload_image(self):
        if self.model is None:
            self.result_label.config(text="Model not loaded correctly!", fg="red")
            return

        self.img_path = filedialog.askopenfilename(title="Select Image", filetypes=(("Image files", "*.jpg *.png *.jpeg"),))
        if self.img_path:
            self.display_image(self.img_path)
            self.predict_emotion(self.img_path)

    def display_image(self, img_path):
        # Read image with OpenCV
        img = cv2.imread(img_path)

        # Convert image to RGB (Tkinter uses RGB format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL format
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((200, 200))  # Resize for consistent display

        # Convert to ImageTk format
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update the label
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep reference to avoid garbage collection

    # def predict_emotion(self, img_path):
    #     # Define emotion dictionary
    #     emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

    #     # Preprocess the image
    #     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #     img_resized = cv2.resize(img, (48, 48))
    #     img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
    #     img_resized = np.expand_dims(img_resized, axis=0)   # Add batch dimension
    #     img_resized = img_resized / 255.0  # Normalize pixel values

    #     # Start time measurement for prediction
    #     start_time = time.time()

    #     # Predict emotion
    #     predictions = self.model.predict(img_resized)
    #     maxindex = int(np.argmax(predictions))
    #     emotion = emotion_dict[maxindex]

    #     # Store prediction results
    #     self.prediction_results = predictions[0]

    #     # End time measurement for prediction
    #     end_time = time.time()
    #     prediction_time = end_time - start_time  # Calculate the time taken

    #     # Update result label with predicted emotion in black color
    #     self.result_label.config(text=f"Predicted Emotion: {emotion}", fg="black")

    #     # Display the time taken for prediction
    #     self.time_label.config(text=f"Time taken for prediction: {prediction_time:.4f} seconds")

    #     # Show the bar graph button after prediction
    #     self.show_percentage_button.pack()  # Make the button visible
    
    def predict_emotion(self, img_path):
        # Define emotion dictionary
        emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
    
        # Preprocess the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (48, 48))
        img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
        img_resized = np.expand_dims(img_resized, axis=0)   # Add batch dimension
        img_resized = img_resized / 255.0  # Normalize pixel values
    
        # Start time measurement for prediction
        start_time = time.time()
    
        # Predict emotion
        predictions = self.model.predict(img_resized)
        maxindex = int(np.argmax(predictions))
        emotion = emotion_dict[maxindex]
        emotion_percentage = round(predictions[0][maxindex] * 100, 1)  # Get the percentage of the predicted emotion
    
        # Store prediction results
        self.prediction_results = predictions[0]
    
        # End time measurement for prediction
        end_time = time.time()
        prediction_time = end_time - start_time  # Calculate the time taken
    
        # Update result label with predicted emotion and its percentage
        self.result_label.config(text=f"Predicted Emotion: {emotion} ({emotion_percentage}%)", fg="black")
    
        # Display the time taken for prediction
        self.time_label.config(text=f"Time taken for prediction: {prediction_time:.4f} seconds")
    
        # Show the bar graph button after prediction
        self.show_percentage_button.pack()  # Make the button visible


    def show_bar_graph(self):
        if self.prediction_results is None:
            return
        
        plt.close('all')

        # Emotion labels
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        # Calculate the maximum value and its index
        max_index = np.argmax(self.prediction_results)

        # Assign colors, with the highest value in a distinct color (e.g., red)
        bar_colors = ['lightblue' for _ in self.prediction_results]
        bar_colors[max_index] = 'lightgreen'  # Highlight the maximum bar in red

        # Plot bar graph
        plt.figure(figsize=(7, 5))
        bars = plt.bar(emotion_labels, self.prediction_results * 100, color=bar_colors)  # Set colors
        plt.xlabel("Expressions",fontsize = 14)
        plt.title("Expression Prediction Percentages", fontsize=15)

        # Hide the y-axis
        plt.gca().get_yaxis().set_visible(False)

        # Remove the plot's bounding box (spines)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        # Add percentage values on top of each bar
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f"{yval:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold', color='lightgreen' if i == max_index else 'black')

        plt.show()

    def exit_app(self):
        # Exit the application
        plt.close('all')
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()

