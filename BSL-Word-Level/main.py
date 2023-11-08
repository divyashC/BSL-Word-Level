import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your TensorFlow/Keras model using a cached function


@st.cache_resource
def load_keras_model():
    model = load_model('bsl_accrray_92_date_02_11_2023.h5')
    return model


# Initialize the webcam
cap = cv2.VideoCapture(1)  # Changed camera index to 1

# Function to release the camera and close OpenCV windows


def release_camera():
    cap.release()
    cv2.destroyAllWindows()


# Specify the classes and their corresponding Dzongkha and English text
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
# Create Streamlit web app
st.title("BSL Sign Language Recognition")

# Function to predict the sign


def predict_sign(frame, model):
    # Preprocess the frame: Resize it to (30, 1662)
    frame = cv2.resize(frame, (1662, 30))

    # Normalize the frame if needed
    # frame = frame / 255.0

    # Assuming your data shape is (30, 1662), reshape it to (1, 30, 1662)
    data = frame.reshape(1, 30, 1662)

    # Make predictions using the loaded model
    prediction = model.predict(data)

    return prediction


# Load the model using the cached function
model = load_keras_model()

# Main Streamlit loop
while True:
    ret, frame = cap.read()

    if not ret:
        st.error("Error: Unable to retrieve frames from the camera.")
        break

    # You can process the frame here or directly pass it to predict_sign function
    prediction = predict_sign(frame, model)

    # Get the predicted sign
    predicted_class = np.argmax(prediction)

    # Display the camera feed
    st.image(frame, channels="BGR",
             use_column_width=True, caption="Camera Feed")

    # Display the predicted sign text (both Dzongkha and English)
    st.header("Predicted Sign:")
    dzongkha_text, english_text = sign_classes.get(
        predicted_class, ("Unknown", "Unknown"))
    st.write(f"Dzongkha: {dzongkha_text}")
    st.write(f"English: {english_text}")

# Release the camera when the app is closed
release_camera()
