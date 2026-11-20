import cv2
import os
import time
import pickle

# Set parameters
DATA_DIR = './data'
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # Update this list with your labels
NUM_IMAGES = 100  # Number of images to capture for each label

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Create folders for each label
for label in LABELS:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture images
for label in LABELS:
    print(f"Collecting images for {label}. Press 'c' to capture and 'q' to quit.")
    time.sleep(2)  # Wait for camera to warm up

    count = 0
    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Image Collection', frame)

        key = cv2.waitKey(1)
        if key == ord('c'):
            img_name = os.path.join(DATA_DIR, label, f"{label}_{count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Captured {img_name}")
            count += 1
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save collected data for training
data_dict = {'data': [], 'labels': []}
for label in LABELS:
    label_path = os.path.join(DATA_DIR, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        data_dict['data'].append(img_path)
        data_dict['labels'].append(label)

with open('sign_data.pickle', 'wb') as f:
    pickle.dump(data_dict, f)

print("Data collection completed and saved.")
