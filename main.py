import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import pygame
import time

# Load the emotion detection model
with open("model_fer.json", "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("model_fer.h5")

# Emotion labels as per model's training order
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Map emotions to music file paths
emotion_to_music = {
    'Angry': "music/angry.mp3",
    'Happy': "music/happy.mp3",
    'Sad': "music/sad.mp3",
    'Neutral': "music/neutral.mp3"
    # Add more if you have them
}

# Initialize pygame mixer
pygame.mixer.init()

# Play music function
def play_music(emotion):
    file_path = emotion_to_music.get(emotion, "music/neutral.mp3")
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing music for {emotion}: {e}")

# Load OpenCV face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))

            prediction = model.predict(roi, verbose=0)[0]
            emotion = emotion_labels[np.argmax(prediction)]

            # Draw label and rectangle
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # If a new emotion is detected, play corresponding song
            if emotion != last_emotion and emotion in emotion_to_music:
                pygame.mixer.music.stop()
                play_music(emotion)
                last_emotion = emotion
        else:
            cv2.putText(frame, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Emotion-Based Music Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
