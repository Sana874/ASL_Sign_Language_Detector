import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# --- Load your trained model and label encoder ---
model = tf.keras.models.load_model(r'C:\Users\abkal\Desktop\fds 4.0\finger_spelling_modellll_palak.h5')

with open(r'C:\Users\abkal\Desktop\fds 4.0\label_encoderrrr_palak.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# --- Setup Mediapipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Open Webcam ---
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

    prediction_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                landmark_array = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmark_array)
                predicted_class = np.argmax(prediction)
                prediction_text = label_encoder.inverse_transform([predicted_class])[0]

    # Display the prediction
    cv2.putText(frame, f"Predicted: {prediction_text}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    cv2.imshow('Finger Spelling Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
