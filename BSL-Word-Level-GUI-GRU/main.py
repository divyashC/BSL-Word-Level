import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Function to release the camera and close OpenCV windows
def release_camera():
    cap.release()
    cv2.destroyAllWindows()

# Load your model
model_path = "./bsl_word_level_gru.h5"
model = tf.keras.models.load_model(model_path)

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

sign_classes = {
    0: ("predicting...", ""),
    1: ("come", "ཤོག།"),
    2: ("sleep", "ཉལ་ནི།"),
    3: ("eat", "བཟའ།"),
    4: ("yes", "ཨིན།"),
    5: ("no", "མེན།"),
    6: ("good", "ལེགས་ཤོམ།"),
    7: ("thin", "ཕྱ་སི་སི།"),
    8: ("more", "མངམ།"),
    9: ("same", "ཅོག་འཐདཔ།"),
    10: ("small", "ཆུང་ཀུ།"),
    11: ("flower", "མེ་ཏོག།"),
    12: ("sun", "ཉིམ།"),
    13: ("star", "སྐར་མ།"),
    14: ("moon", "ཟླཝ།"),
    15: ("road", "ལམ།")
}

# CSS for custom styling
css = """
<style>
.title-box {
    background-color: #d3d3d3;
    color: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    width:80%;
    margin: 0 auto;
    margin-bottom: 50px;
    text-align: center;
}

.camera-box {
    border: 1px solid #ccc;
    width: 640px;
    height: 480px;
    margin: 50px auto;
    text-align: center;
    padding: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-bottom: 50px;
}

.output-box {
    border: 0.2px solid #ccc;
    width: 400px;
    padding: 2px 20px;
    margin: 0 auto 50px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 20px;
    font-size: 30px;
}

.text-box {
    border: 0.2px solid #ccc;
    width: 400px;
    padding: 2px 20px;
    margin: 0 auto 50px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 20px;
    font-size: 30px;
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)
st.markdown('<div class="title-box"><h1>Bhutanese Sign Language <br/>Word Level Recognition</h1></div>', unsafe_allow_html=True)

# Initialize the webcam
cap = cv2.VideoCapture(1)  # Use camera 1

# Define frame for camera view
frame_placeholder = st.empty()
frame_placeholder.markdown('<div class="camera-box">', unsafe_allow_html=True)
frame_placeholder.markdown("Camera Feed")
frame_placeholder.markdown("</div>", unsafe_allow_html=True)

# Define frame for predicted output
output_placeholder = st.empty()
sequence = []
sentence = []
predictions = []
threshold = 0.5

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    predicted_class = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Unable to retrieve frames from the camera.")
            st.stop()

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(frame_rgb)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if np.max(res) > threshold:
                current_class = np.argmax(res)

                if current_class != predicted_class:
                    eng_word, dzongkha_word = sign_classes[current_class]
                    output_placeholder.markdown(
                        f'<div class="output-box">English: {eng_word}<br/>Dzongkha: {dzongkha_word}</div>',
                        unsafe_allow_html=True,
                    )
                    predicted_class = current_class

        # Display camera frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the camera
release_camera()
