import cv2
import numpy as np
from tensorflow.keras.models import load_model

#Saved model
model = load_model("/Users/sudheernarasimha/facial-emotion-recognition/models/best_model.h5")
#Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        cropped = cv2.resize(roi_gray, (48, 48))
        cropped = cropped.astype("float32") / 255.0
        cropped = np.expand_dims(cropped, axis=0)
        cropped = np.expand_dims(cropped, axis=-1)

        preds = model.predict(cropped)[0]
        emotion_prob = np.max(preds)
        emotion_label = emotion_labels[np.argmax(preds)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion_label} ({emotion_prob*100:.1f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()