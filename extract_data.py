import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import EarlyStopping

# Constants
DATASET_PATH = 'D:/PythonProjects/Hand Sign Language Recognition/asl_dataset'  # Update this to your dataset path
IMG_WIDTH, IMG_HEIGHT = 224, 224  # MobileNetV2 input size

# Load images and labels
data = []
labels = []

# Debug: Print to ensure the dataset path is correct
print(f"Dataset path: {DATASET_PATH}")

for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)
    if os.path.isdir(folder_path):
        print(f"Loading images for label: {label}")  # Debug: Print label being loaded
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize to MobileNetV2 input size
                data.append(img)
                labels.append(label)
            else:
                print(f"Failed to load image: {img_path}")  # Log any failed images

# Convert to numpy arrays
data = np.array(data, dtype='float32') / 255.0  # Normalize the images
labels = np.array(labels)

# Map string labels to integers
unique_labels = np.unique(labels)
label_map = {label: idx for idx, label in enumerate(unique_labels)}

# Debug: Print label mapping to ensure it's correct
print(f"Label mapping: {label_map}")

y = np.array([label_map[label] for label in labels])

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=len(unique_labels))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freeze the base model layers
base_model.trainable = False

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Convert the 2D output to 1D for Dense layers
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(len(unique_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, callbacks=[early_stopping])

# Save the trained model
model.save('hand_sign_language_mobilenet_model.h5')

# Debug: Check the accuracy of the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")
