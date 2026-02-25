# Hand Sign Language Recognition

A deep learning-based **Hand Sign Language Recognition** project that trains a model on sign image datasets and performs **real-time sign prediction from webcam input**.

---

## Project Overview

This project includes:

1. **Model Training**
   - Loads sign language image dataset from class folders
   - Preprocesses and resizes images
   - Trains a **MobileNetV2-based classifier** using transfer learning
   - Saves the trained model

2. **Real-Time Recognition**
   - Captures webcam frames
   - Resizes and normalizes frames
   - Predicts the hand sign using the trained model
   - Displays predicted sign label on the live video feed

---

## Features

- Hand sign image classification
- Transfer learning with MobileNetV2
- Webcam-based real-time recognition
- Automatic class label loading from dataset folders
- Early stopping during training
- Keras model saving for inference

---

## Tech Stack

- Python
- OpenCV
- NumPy
- TensorFlow / Keras
- Scikit-learn

---
