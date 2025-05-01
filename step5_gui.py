import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import time
import os
from collections import deque, Counter

# --- Load model and label encoder ---
model_path = r'C:\Users\abkal\Desktop\fds 4.0\finger_spelling_modellll_palak_wspacewbs.h5'
label_path = r'C:\Users\abkal\Desktop\fds 4.0\label_encoderrrr_palak_wspacewbs.pkl'

if not os.path.exists(model_path) or not os.path.exists(label_path):
    st.error("Model or label encoder file not found.")
    st.stop()

model = tf.keras.models.load_model(model_path)
with open(label_path, 'rb') as f:
    label_encoder = pickle.load(f)

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Streamlit UI ---
st.set_page_config(page_title="ASL Sentence Builder", layout="centered")
st.title("ASL Finger Spelling Sentence Builder")
st.markdown("Hold gestures steady - Make finger signs —> we turn it into text!")

# --- Session state setup ---
for key in ['run', 'sentence', 'last_pred', 'last_update_time', 'frame_queue']:
    if key not in st.session_state:
        if key == 'frame_queue':
            st.session_state[key] = deque(maxlen=7)
        elif key == 'run':
            st.session_state[key] = False
        elif key in ['sentence', 'last_pred']:
            st.session_state[key] = ""
        else:
            st.session_state[key] = time.time()

# --- UI buttons ---
col1, col2, col3 = st.columns(3)
with col1:
    start = st.button("Start")
with col2:
    stop = st.button("Stop")
with col3:
    clear = st.button("Clear Sentence")

# --- Display areas ---
FRAME_WINDOW = st.image([])
sentence_placeholder = st.empty()

def capture_and_predict():
    cap = cv2.VideoCapture(1)  # Use 0 if needed depending on your webcam

    while st.session_state.run:
        success, frame = cap.read()
        if not success:
            st.warning("Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_prediction = ""

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction = model.predict(np.array(landmarks).reshape(1, -1), verbose=0)
                predicted_class = np.argmax(prediction)
                current_prediction = label_encoder.inverse_transform([predicted_class])[0]

        # Stability logic
        now = time.time()
        if current_prediction:
            st.session_state.frame_queue.append(current_prediction)
            most_common, count = Counter(st.session_state.frame_queue).most_common(1)[0]

            if (
                most_common == current_prediction and
                count >= 6 and
                most_common != st.session_state.last_pred and
                now - st.session_state.last_update_time > 5
            ):
                if current_prediction == "SPACE":
                    st.session_state.sentence += " "
                elif current_prediction == "BACKSPACE":
                    st.session_state.sentence = st.session_state.sentence[:-1]
                else:
                    st.session_state.sentence += current_prediction

                st.session_state.last_pred = current_prediction
                st.session_state.last_update_time = now
                st.session_state.frame_queue.clear()

        # Overlay + Display
        cv2.putText(frame, f"Prediction: {current_prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        FRAME_WINDOW.image(frame, channels="BGR")
        sentence_placeholder.markdown(f"### Sentence: `{st.session_state.sentence}`")

    cap.release()
    FRAME_WINDOW.empty()
    sentence_placeholder.markdown(f"### Sentence: `{st.session_state.sentence}`")

# --- Button logic ---
if start:
    st.session_state.run = True
    capture_and_predict()

if stop:
    st.session_state.run = False

if clear:
    st.session_state.sentence = ""
    st.session_state.last_pred = ""
    st.session_state.last_update_time = time.time()
    st.session_state.frame_queue.clear()
    sentence_placeholder.markdown("### Sentence: `...waiting for signs...`")
