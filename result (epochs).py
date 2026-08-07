# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 20:13:53 2024

@author: User
"""
import matplotlib.pyplot as plt
plt.close()
# Data from the table
epochs = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
accuracy = [0.49, 0.61, 0.64, 0.65, 0.67, 0.67, 0.68, 0.68, 0.69, 0.69, 0.69]
training_time = [4.42, 21.97, 44.74, 70.7, 104.61, 122.34, 139.03, 153.42, 187.21, 201.66, 251.22]

# Create the first plot: Epoch vs. Accuracy
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, marker='o', color='blue', linewidth=1, markersize=4)  # Smaller dots (markersize=5)
plt.title("Number of epochs vs. Accuracy",fontsize = 14)
plt.xlabel("Number of epochs", fontsize = 12)
plt.ylabel("Accuracy", fontsize = 12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, axis='y')  # Horizontal grid lines only

# Add value labels above the dots with some vertical offset
for i, txt in enumerate(accuracy):
    # Move the labels for the first two points slightly to the right
    if i < 2:
        plt.text(epochs[i] + 26, accuracy[i] + 0.001, f'{txt:.2f}', ha='center', va='bottom')  
    elif i == 3:  # Corrected 'elseif' to 'elif'
        plt.text(epochs[i], accuracy[i] + 0.003, f'{txt:.2f}', ha='center', va='bottom')  
    else:
        plt.text(epochs[i], accuracy[i] + 0.001, f'{txt:.2f}', ha='center', va='bottom')

# Create the second plot: Epoch vs. Training Time
plt.subplot(1, 2, 2)
plt.plot(epochs, training_time, marker='o', color='red', linewidth=1, markersize=4)  # Smaller dots (markersize=5)
plt.title("Number of epochs vs. Training Time",fontsize=14)
plt.xlabel("Number of epochs",fontsize=12)
plt.ylabel("Training Time (min)",fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, axis='y')  # Horizontal grid lines only

# Add value labels above the dots with some vertical offset
for i, txt in enumerate(training_time):
    # Round the training time to one decimal place
    rounded_time = round(txt, 1)
    plt.text(epochs[i], training_time[i] + 3, f'{rounded_time:.1f}', ha='center', va='bottom')

# Show the plots
plt.tight_layout()
plt.show()


