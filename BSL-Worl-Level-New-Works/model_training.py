import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Load the collected data from the "Custom_Data" folder
DATA_PATH = "Custom_Data"

# Define the actions/words you collected data for
# actions = ["come", "flower", "more"]
actions = ['predicting...', 'good', 'thin', 'more', 'same', 'small', 'come', 'sleep', 'eat', 'yes', 'no', 'flower', 'sun', 'star', 'moon', 'road']


# Number of sequences (videos) collected for each action
no_sequences = 30

# Number of frames in each sequence
sequence_length = 30

# Initialize variables to store data and labels
sequences, labels = [], []

# Load the data and labels
for action in actions:
    for sequence in range(no_sequences):
        action_dir = os.path.join(DATA_PATH, action, str(sequence))
        for frame_num in range(sequence_length):
            keypoints = np.load(os.path.join(action_dir, str(frame_num) + ".npy"))
            sequences.append(keypoints)
            labels.append(actions.index(action))

# Convert sequences and labels to NumPy arrays
X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# Reshape the training and testing data to be 3D
X_train = X_train.reshape(X_train.shape[0], sequence_length, -1)
X_test = X_test.reshape(X_test.shape[0], sequence_length, -1)


# Define and compile the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X_train.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model using confusion matrix and accuracy
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
accuracy = accuracy_score(ytrue, yhat)

print("Confusion Matrix:")
print(confusion_matrix)
print("Accuracy:", accuracy)

# Save the trained model
model.save('bsl_word_level_1.h5')