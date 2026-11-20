import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import layers, models

# Define parameters
IMAGE_SIZE = (224, 224)  # MobileNetV2 expects images of this size
BATCH_SIZE = 16
EPOCHS = 10


# Function to load data
def load_data(data_dir):
    X = []
    y = []

    # Iterate through each folder in the dataset
    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    img = cv2.resize(img, IMAGE_SIZE)  # Resize image
                    X.append(img)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)
    X = X.astype('float32') / 255.0  # Normalize images

    return X, y


# Load data
data_dir = './data'  # Update this path to your dataset directory
X, y = load_data(data_dir)

# Encode labels
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Load the MobileNetV2 model pre-trained on ImageNet
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False,
                                               input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(lb.classes_), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save the model
model.save('hand_sign_mobilenet_model.h5')
print("Model saved as 'hand_sign_mobilenet_model.h5'")
