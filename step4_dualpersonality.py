import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Load model and label encoder
model = tf.keras.models.load_model(r'C:\Users\abkal\Desktop\fds 4.0\finger_spelling_modellll_palak.h5')

with open(r'C:\Users\abkal\Desktop\fds 4.0\label_encoderrrr_palak.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Setup Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open Webcam
cap = cv2.VideoCapture(1)

print("Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    person1_prediction = ""
    person2_prediction = ""

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                landmark_array = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmark_array)
                predicted_class = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]

                if idx == 0:
                    person1_prediction = predicted_label
                elif idx == 1:
                    person2_prediction = predicted_label

    # Display predictions
    cv2.putText(frame, f"Person 1: {person1_prediction}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Person 2: {person2_prediction}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Finger Spelling Prediction - Two Persons', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
