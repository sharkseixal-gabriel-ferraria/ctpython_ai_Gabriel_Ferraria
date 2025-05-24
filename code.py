import cv2
import numpy as np
from deepface import DeepFace
import time

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera.")
else:
    frame_count = 0
    emotion = "neutral"

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_color = frame[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Run emotion detection every 30 frames
            if frame_count % 30 == 0:
                try:
                    result = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                except Exception as e:
                    print("Emotion detection error:", e)

            # Display emotion
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Trigger actions
            if emotion == 'happy':
                auto.press("win")
                auto.typewrite("chrome")
                auto.press("enter")
                time.sleep(1)
                auto.press

            elif emotion == 'sad':
                auto.press("win")
                auto.typewrite("get out")

        frame_count += 1
        cv2.imshow('Emotion-Aware Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
