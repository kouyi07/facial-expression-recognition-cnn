import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the entire model (architecture and weights)
model = load_model('emotion_recognition_model32-400.keras')

# Prevents OpenCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary that assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Create a named window and set its size
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL to allow resizing
cv2.resizeWindow('Video', 800, 600)  # Set the desired width and height

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)  # 1 means flipping around the y-axis

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)

        # Extract the region of interest (ROI)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Make prediction
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))

        # Display the emotion label
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)

    # Show the resulting frame
    cv2.imshow('Video', cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC))

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
