# Define the actions/words for data collection
# actions = np.array(['predicting...'])
# actions = np.array(['good', 'thin', 'more', 'same', 'small'])
# actions = np.array(['come', 'sleep', 'eat', 'yes', 'no'])
# actions = np.array(['flower', 'sun', 'star', 'moon', 'road'])
# actions = np.array(['predicting...', 'good', 'thin', 'more', 'same', 'small', 'come', 'sleep', 'eat', 'yes', 'no', 'flower', 'sun', 'star', 'moon', 'road'])


import cv2
import numpy as np
import os
import time
import mediapipe as mp

# Create a directory to save the data
DATA_PATH = "BSL_Word_Level_Data"
os.makedirs(DATA_PATH, exist_ok=True)

# Define the actions/words for data collection
actions = np.array(['predicting...','come', 'sleep', 'eat', 'yes', 'no'])
# actions = ["predicting...", "good", "thin", "more", "same", "small", "come", "sleep", "eat", "yes", "no", "flower", "sun", "star", "moon", "road"]

# Number of sequences (videos) to collect for each action
no_sequences = 30

# Number of frames in each sequence
sequence_length = 30

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create a function to extract keypoints from head, shoulders, arms, and palms
def extract_keypoints(results):
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    return np.array(keypoints)

cap = cv2.VideoCapture(1)  # Use the appropriate camera index

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    for action in actions:
        for sequence in range(no_sequences):
            # Create a directory for the current action and sequence
            action_dir = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(action_dir, exist_ok=True)

            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                # Make detections using MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract keypoints for head, shoulders, arms, and palms
                keypoints = extract_keypoints(results)

                # Save keypoints as a NumPy array
                npy_path = os.path.join(action_dir, str(frame_num))
                np.save(npy_path, keypoints)

                # Overlay landmarks on the camera feed
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                cv2.putText(image, f"Collecting data for {action} - Sequence {sequence}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (6, 6, 255), 1, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
