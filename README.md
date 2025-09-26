ASL Sign Language Detector
This project implements a real-time American Sign Language (ASL) finger-spelling detection system using Python, OpenCV, MediaPipe, and TensorFlow/Keras. The goal is to bridge communication gaps by enabling automatic recognition of ASL signs from a webcam feed.

Features
- Dataset capture from webcam for custom ASL gesture collection.
- Deep learning model (CNN) trained on captured gestures.
- Real-time prediction of ASL gestures via webcam.
- Accuracy and loss visualization during training.
- Support for detecting signs from two individuals simultaneously.
- Streamlit-based web interface and a simple GUI application.

Repository Structure:
ASL_Sign_Language_Detector/
│── step1_capturedataset.py       # Capture ASL gesture images
│── step2_train_model.py          # Train CNN model on gestures
│── step3_predict_from_webcam.py  # Real-time prediction via webcam
│── step4_dualpersonality.py      # Detect signs from two people
│── step4.5_generategraph.py      # Plot accuracy/loss graphs
│── step5_gui.py                  # Simple GUI for sign detection
│── step6_streamlit_2ppl.py       # Streamlit app for 2-person detection
│── finger_spelling_modellll_finall.h5   # Trained model
│── label_encoderrrr_finall.pkl          # Label encoder
│── Report - Real Time Sign Language Detection.pdf
│── Final Real-time Sign Language Gesture Detection.pdf


Installation & Setup

1. Clone the repository:
git clone https://github.com/Sana874/ASL_Sign_Language_Detector.git
cd ASL_Sign_Language_Detector

2. Install dependencies:
pip install -r opencv-python mediapipe tensorflow numpy pickle-mixin streamlit

Run desired scripts:
- Capture dataset → python step1_capturedataset.py
- Train model → python step2_train_model.py
- Predict live → python step3_predict_from_webcam.py
- Streamlit app → streamlit run step6_streamlit_2ppl.py

Applications
- Assistive technology for hearing-impaired communication.
- Educational tool for ASL learning.
- Prototype for integration into smart devices and AR/VR platforms.
