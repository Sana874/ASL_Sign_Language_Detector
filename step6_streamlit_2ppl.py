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
model_path = r'C:\Users\abkal\Desktop\fds 4.0\finger_spelling_modellll_palak_wspace.h5'
label_path = r'C:\Users\abkal\Desktop\fds 4.0\label_encoderrrr_palak_wspace.pkl'

if not os.path.exists(model_path) or not os.path.exists(label_path):
    st.error("Model or label encoder file not found.")
    st.stop()

model = tf.keras.models.load_model(model_path)
with open(label_path, 'rb') as f:
    label_encoder = pickle.load(f)

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Streamlit UI ---
st.set_page_config(page_title="ASL Dual-Person Sentence Builder", layout="centered")
st.title("ASL Finger Spelling - Dual Person Sentence Builder")
st.markdown("Supports two simultaneous users — each with their own real-time sentence.")

# --- Initialize session state ---
for person in ['1', '2']:
    for key in [f'sentence_{person}', f'last_pred_{person}', f'last_time_{person}', f'queue_{person}']:
        if key not in st.session_state:
            if 'queue' in key:
                st.session_state[key] = deque(maxlen=7)
            elif 'last_time' in key:
                st.session_state[key] = time.time()
            else:
                st.session_state[key] = ""

if 'run' not in st.session_state:
    st.session_state.run = False

# --- UI buttons ---
col1, col2, col3 = st.columns(3)
with col1:
    start = st.button("Start")
with col2:
    stop = st.button("Stop")
with col3:
    clear = st.button("Clear Sentences")

# --- Display areas ---
FRAME_WINDOW = st.image([])
sentence_box = st.empty()

def capture_and_predict_dual():
    cap = cv2.VideoCapture(1)  # Use 0 if needed

    while st.session_state.run:
        success, frame = cap.read()
        if not success:
            st.warning("Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predictions = ["", ""]

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    prediction = model.predict(np.array(landmarks).reshape(1, -1), verbose=0)
                    predicted_class = np.argmax(prediction)
                    predictions[idx] = label_encoder.inverse_transform([predicted_class])[0]

        now = time.time()

        for i in [0, 1]:
            person = str(i + 1)
            pred = predictions[i]
            if pred:
                st.session_state[f'queue_{person}'].append(pred)
                most_common, count = Counter(st.session_state[f'queue_{person}']).most_common(1)[0]

                if (
                    most_common == pred and
                    count >= 6 and
                    most_common != st.session_state[f'last_pred_{person}'] and
                    now - st.session_state[f'last_time_{person}'] > 2.5
                ):
                    if pred == "SPACE":
                        st.session_state[f'sentence_{person}'] += " "
                    else:
                        st.session_state[f'sentence_{person}'] += pred

                    st.session_state[f'last_pred_{person}'] = pred
                    st.session_state[f'last_time_{person}'] = now
                    st.session_state[f'queue_{person}'].clear()

        # Overlay predictions
        cv2.putText(frame, f"Person 1: {predictions[0]}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
        cv2.putText(frame, f"Person 2: {predictions[1]}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 150), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

        # Show both sentences
        sentence_box.markdown(f"### 👤 Person 1 Sentence: `{st.session_state['sentence_1']}`")
        sentence_box.markdown(f"### 👤 Person 2 Sentence: `{st.session_state['sentence_2']}`")

    cap.release()
    FRAME_WINDOW.empty()

# --- Button logic ---
if start:
    st.session_state.run = True
    capture_and_predict_dual()

if stop:
    st.session_state.run = False

if clear:
    for p in ['1', '2']:
        st.session_state[f'sentence_{p}'] = ""
        st.session_state[f'last_pred_{p}'] = ""
        st.session_state[f'last_time_{p}'] = time.time()
        st.session_state[f'queue_{p}'].clear()
    sentence_box.markdown("### 👤 Sentences cleared.")
