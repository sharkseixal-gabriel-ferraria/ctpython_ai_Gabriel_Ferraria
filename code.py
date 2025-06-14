import cv2
import numpy as np
from deepface import DeepFace
import tensorflow as tf
import keras
import pyautogui as auto
import time
from playsound import playsound
import threading

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture
cam = cv2.VideoCapture(0)

def play_sound():
    playsound('C:\\Users\\CTPAI - GABRIEL F\\Sad.wav')

def play_chill():
    playsound('C:\\Users\\CTPAI - GABRIEL F\\Chill.wav')



stop_detection = False
if not cam.isOpened():
    print("Error: Could not open camera.")
else:
    frame_count = 0
    emotion = "Vai ganhar o 50/50"

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

            # Run emotion detection every 120 frames
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
                print("Estás feliz? mas quem é que te deixou estar feliz,toma")
                auto.press("win")
                auto.typewrite("opera")
                auto.press("enter")
                time.sleep(2)
                auto.typewrite("League of Legends download")
                auto.press("enter")


                cam.release()
                cv2.destroyAllWindows()
                exit()

            elif emotion == 'sad':
                print("Estás triste")
                # Call the function in a new thread
                threading.Thread(target=play_sound, daemon=True).start()


            elif emotion == 'angry':
                print("Estás chatiado")
                threading.Thread(target=play_chill, daemon=True).start()

            elif emotion == 'fear':
                print("Medricas")
                auto.press("win")
                auto.typewrite("opera")
                auto.press("enter")
                time.sleep(2)
                auto.typewrite("https://i5.walmartimages.com/seo/10-Freddy-Large-Size-Five-Nights-at-Freddy-s-FNAF-Brown-Bear-Plush-Doll-Toy_abda3d68-ec3b-4547-a468-ff4ff90c4e08.f9e7514a15171130fdf69679ee662712.jpeg")
                auto.press("enter")

                cam.release()
                cv2.destroyAllWindows()
                exit()

        frame_count += 1
        cv2.imshow('Emotion-Aware Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
