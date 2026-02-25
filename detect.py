import os
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('hand_sign_language_model.h5')

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Desired image size
class_labels = sorted(os.listdir('D:\\PythonProjects\\Hand Sign Language Recognition\\asl_dataset'))  # Get class labels from dataset folder

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Camera not opened.")
    exit()

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to match the model input shape
    roi = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize the image and prepare it for prediction
    roi = roi.astype('float32') / 255.0  # Normalize to [0, 1]
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(roi)
    predicted_class = class_labels[np.argmax(prediction)]  # Get the predicted label

    # Display the prediction on the frame
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Sign Language Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
