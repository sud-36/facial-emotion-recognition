# facial-emotion-recognition
CNN based emotion classifier
**Language / Frameworks:** Python, TensorFlow, OpenCV  

**Personal Project | Jul 2025**

---

## Project Overview

This project is a real-time facial emotion recognition system using a Convolutional Neural Network (CNN) trained on the FER2013 dataset. The system can detect 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. Model was trained on an ipynb in Colab using Google Cloud GPU for free and fast tranining, while also saving storage on local disk. 

---

## Features

- CNN-based emotion classifier achieving ~66% accuracy on FER2013.
- Data augmentation and class weights to address class imbalance, improving minority class recall.
- Real-time webcam emotion detection using OpenCV, displaying probabilities on the video feed.
