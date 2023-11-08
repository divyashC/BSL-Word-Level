import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import mediapipe as mp

# Load the trained model
model = keras.models.load_model('bsl_word_level_2.h5')

# Define the actions corresponding to model output
actions = np.array(['predicting...', 'come', 'sleep', 'eat', 'yes', 'no'])

# Set up OpenCV to capture video from your camera
cap = cv2.VideoCapture(1)  # Use the appropriate camera index

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic

st.title("BSL Word Level Recognition Model")

# Create a Streamlit placeholder to display video
video_placeholder = st.empty()

# Create a Streamlit placeholder for the prediction
prediction_placeholder = st.empty()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()

        # Check if the frame is not empty
        if frame is not None:
            # Convert the frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the video feed in the Streamlit app
            video_placeholder.image(rgb_frame, channels="RGB")

            # Make detections using Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract arm and hand keypoints
            keypoints = []
            if results.pose_landmarks is not None:
                for landmark in results.pose_landmarks.landmark[11:21]:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])

            # Reshape the keypoints to match the model's expected input shape
            keypoints = np.array(keypoints)
            keypoints = keypoints.reshape(-1, 30, 99)

            # Make a prediction using the model
            prediction = model.predict(keypoints)
            action = actions[np.argmax(prediction)]

            # Update the prediction placeholder with the current prediction
            prediction_placeholder.text("Recognized Action: " + action)

        # Stop the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
