import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical

# Define your data directory
DATA_PATH = "BSL_Word_Level_Data"

# Load and preprocess data
X = []
y = []
actions = os.listdir(DATA_PATH)

for action in actions:
    sequences = os.listdir(os.path.join(DATA_PATH, action))
    for sequence in sequences:
        frames = os.listdir(os.path.join(DATA_PATH, action, sequence))
        sequence_data = []
        for frame in frames:
            key_points = np.load(os.path.join(DATA_PATH, action, sequence, frame))
            sequence_data.append(key_points)
        X.append(sequence_data)
        y.append(action)

X = np.array(X)
y = np.array(y)

# Encode the actions as numerical labels
action_to_label = {action: i for i, action in enumerate(actions)}
y = [action_to_label[action] for action in y]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(actions))
y_test = to_categorical(y_test, num_classes=len(actions))

# Ensure that the labels match the sequence length
sequence_length, num_features = X.shape[1], X.shape[2]
y_train = np.repeat(y_train[:, np.newaxis, :], sequence_length, axis=1)
y_test = np.repeat(y_test[:, np.newaxis, :], sequence_length, axis=1)

# Add dropout layers to prevent overfitting
model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, num_features), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(Dense(len(actions), activation='softmax'))

# Implement learning rate scheduling
initial_learning_rate = 0.001
def lr_scheduler(epoch):
    return initial_learning_rate * 0.95 ** epoch

optimizer = Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Create a learning rate scheduler
lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model with learning rate scheduling
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, callbacks=[lr_callback])

# Evaluate the model on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=2)
y_test_classes = np.argmax(y_test, axis=2)

# Calculate and print confusion matrix
conf_matrix = confusion_matrix(y_test_classes.ravel(), y_pred_classes.ravel())
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and print accuracy
accuracy = accuracy_score(y_test_classes.ravel(), y_pred_classes.ravel())
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
