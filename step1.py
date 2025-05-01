import cv2
import mediapipe as mp
import pandas as pd
import os

# --- Setup Mediapipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Create/Open Dataset File ---
if not os.path.exists('landmarks_dataset.csv'):
    # If file doesn't exist, create with header
    columns = []
    for i in range(21):
        columns += [f'x{i}', f'y{i}', f'z{i}']
    columns.append('label')
    df = pd.DataFrame(columns=columns)
    df.to_csv('landmarks_dataset.csv', index=False)
else:
    # If file exists, load it
    df = pd.read_csv('landmarks_dataset.csv')
# --- Open Webcam ---
cap = cv2.VideoCapture(1)
print("Starting webcam... Press 's' to save landmarks, 'q' to quit.")
current_label = input("Enter the label for this recording session (e.g., A, B, C, etc.): ").upper()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display Label
    cv2.putText(frame, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Landmark Recorder', frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                landmarks.append(current_label)
                df.loc[len(df)] = landmarks
                print(f"Saved sample {len(df)} for label '{current_label}'.")
            else:
                print("Incomplete landmarks. Skipping capture.")
        else:
            print("No hand detected. Try again!")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save dataset
df.to_csv('landmarks_dataset.csv', index=False)
print("Dataset saved as 'landmarks_dataset.csv'.")
