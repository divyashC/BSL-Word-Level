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
model_path = "./bsl_accrray_92_date_02_11_2023.h5"
model = tf.keras.models.load_model(model_path)

mp_hands = mp.solutions.hands

# Increase min_detection_confidence for better hand landmark detection
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

# Map sign class to English and Dzongkha translations
sign_classes = {
    0: ("come", "ཤོག།"),
    1: ("sleep", "ཉལ་ནི།"),
    2: ("eat", "བཟའ།"),
    3: ("yes", "ཨིན།"),
    4: ("no", "མེན།"),
    5: ("good", "ལེགས་ཤོམ།"),
    6: ("thin", "ཕྱ་སི་སི།"),
    7: ("more", "མངམ།"),
    8: ("same", "ཅོག་འཐདཔ།"),
    9: ("small", "ཆུང་ཀུ།"),
    10: ("flower", "མེ་ཏོག།"),
    11: ("sun", "ཉིམ།"),
    12: ("star", "སྐར་མ།"),
    13: ("moon", "ཟླཝ།"),
    14: ("road", "ལམ།")
}

# CSS for custom styling
css = """
<style>
.title-box {
    background-color: #1E90FF;
    color: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    width: 60%;
    margin: 50px auto;
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
}

.output-box {
    border: 0.2px solid #ccc;
    width: 400px;
    padding: 2px 20px;
    margin: 0 auto 50px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 20px;
    font-size: 1.5em;
}

.text-box {
    border: 0.2px solid #ccc;
    width: 400px;
    padding: 2px 20px;
    margin: 0 auto 50px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 20px;
    font-size: 1.3em;
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.title("Bhutanese Sign Language - Word Level Recognition")

# Initialize the webcam
cap = cv2.VideoCapture(1)  # Use camera 1

# Define frame for camera view
frame_placeholder = st.empty()
frame_placeholder.markdown('<div class="camera-box">', unsafe_allow_html=True)
frame_placeholder.markdown("Camera Feed")
frame_placeholder.markdown("</div>", unsafe_allow_html=True)

# Define frame for predicted output
output_placeholder = st.empty()
output_placeholder.markdown('<div class="output-box">', unsafe_allow_html=True)
output_placeholder.markdown("Output")
output_placeholder.markdown("</div>", unsafe_allow_html=True)
output_message = st.empty()
output_message.markdown('<div class="text-box"></div>', unsafe_allow_html=True)

sequence = []
sentence = []
threshold = 0.5

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    predicted_class = -1
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Unable to retrieve frames from the camera.")
            st.stop()

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        data_aux = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x - min(x_), landmark.y - min(y_)])

                data_aux.extend(hand_data)

        while len(data_aux) < 1662:
            data_aux.extend([0, 0])

        sequence.append(data_aux)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if np.max(res) > threshold:
                current_class = np.argmax(res)

                if current_class != predicted_class:
                    eng_word, dzongkha_word = sign_classes[current_class]
                    output_message.markdown(
                        f'<div class="text-box"><p>English: {eng_word}</p><p>Dzongkha: {dzongkha_word}</p></div>',
                        unsafe_allow_html=True,
                    )
                    predicted_class = current_class

        # Display camera frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the camera
release_camera()

# Display supported sign classes at the end
# st.write("Supported Sign Classes:")
# st.write("1. Come - ཤོག།")
# st.write("2. Sleep - ཉལ་ནི།")
# st.write("3. Eat - བཟའ།")
# st.write("4. Yes - ཨིན།")
# st.write("5. No - མེན།")
# st.write("6. Good - ལེགས་ཤོམ།")
# st.write("7. Thin - ཕྱ་སི་སི།")
# st.write("8. More - མངམ།")
# st.write("9. Same - ཅོག་འཐདཔ།")
# st.write("10. Small - ཆུང་ཀུ།")
# st.write("11. Flower - མེ་ཏོག།")
# st.write("12. Sun - ཉིམ།")
# st.write("13. Star - སྐར་མ།")
# st.write("14. Moon - ཟླཝ།")
# st.write("15. Road - ལམ།")
