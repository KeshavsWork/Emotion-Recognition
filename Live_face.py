from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import sys

# Ensure UTF-8 encoding is used
sys.stdout.reconfigure(encoding='utf-8')

# Custom img_to_array function
def img_to_array(img):
    return np.array(img, dtype='float32')

# Load pre-trained emotion detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model("D:\\PROJECT\\folder\\emotion_model.h5")

# Define emotion classes
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels = []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Prepare image for prediction
        roi = roi_gray.astype('float') / 255.0  # Scale
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  # Expand dims to (1, 48, 48, 1)

        preds = emotion_model.predict(roi)[0]  # Get one-hot encoded result for 7 classes
        label = class_labels[preds.argmax()]  # Find the label
        label_position = (x, y)
        
        # Print the label for debugging (in case of encoding issues)
        print(label)

        # Ensure correct encoding for cv2.putText (if required)
        cv2.putText(frame, label.encode('utf-8').decode('utf-8'), label_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    # Show the emotion-detected frame
    cv2.imshow('Emotion Detector', frame)
    
    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
