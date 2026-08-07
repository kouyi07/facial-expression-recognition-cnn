# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:02:00 2024

@author: User
"""
import matplotlib.pyplot as plt

# Data from the table
batch_sizes = [16, 32, 64, 128]
accuracies = [0.25, 0.62, 0.6, 0.59]
training_times = [26.76, 21.3, 19.24, 18.67]

# Create the first plot: Batch Size vs. Accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(batch_sizes, accuracies, marker='o', color='blue', linewidth=1, markersize=5)  # Smaller dots (markersize=5)
plt.title("Batch Size vs. Accuracy")
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.grid(True, axis='y')  # Horizontal grid lines only

# Add value labels above the dots with some vertical offset
for i, txt in enumerate(accuracies):
    # Move the labels for the first two points slightly to the right
    if i < 1:
        plt.text(batch_sizes[i] + 5, accuracies[i] + 0.003, f'{txt:.2f}', ha='center', va='bottom')  # Shift to the right for the first two points
    else:
        plt.text(batch_sizes[i], accuracies[i] + 0.003, f'{txt:.2f}', ha='center', va='bottom')

# Create the second plot: Batch Size vs. Training Time
plt.subplot(1, 2, 2)
plt.plot(batch_sizes, training_times, marker='o', color='red', linewidth=1, markersize=5)  # Smaller dots (markersize=5)
plt.title("Batch Size vs. Training Time")
plt.xlabel("Batch Size")
plt.ylabel("Training Time (min)")
plt.grid(True, axis='y')  # Horizontal grid lines only

# Add value labels above the dots with some vertical offset
for i, txt in enumerate(training_times):
    # Round the training time to one decimal place
    rounded_time = round(txt, 1)
    plt.text(batch_sizes[i], training_times[i]+0.1, f'{rounded_time:.1f}', ha='center', va='bottom')

# Show the plots
plt.tight_layout()
plt.show()




