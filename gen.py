import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import mediapipe as mp

# Load gender model (make sure path is correct)
model_path = r"C:\Users\Asma\Downloads\face parts\gender_model.hdf5"
gender_model = load_model(model_path, compile=False)

# Corrected label order based on testing
gender_labels = ['Female', 'Male']  # Swapped: index 0 = Female, 1 = Male

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get bounding box
            x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
            x1, y1 = max(min(x_coords), 0), max(min(y_coords), 0)
            x2, y2 = min(max(x_coords), w), min(max(y_coords), h)

            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                try:
                    face_resized = cv2.resize(face_img, (64, 64))
                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    face_gray = face_gray.astype("float") / 255.0
                    face_gray = img_to_array(face_gray)
                    face_gray = np.expand_dims(face_gray, axis=0)

                    gender_pred = gender_model.predict(face_gray)[0]
                    print("Raw prediction:", gender_pred)  # Debugging

                    gender = gender_labels[np.argmax(gender_pred)]
                    confidence = np.max(gender_pred)

                    cv2.putText(frame, f"Gender: {gender} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception as e:
                    print("Prediction error:", e)

            # === Face Part Labels (smaller font size) ===
            part_indices = {
                "Nose": 1,
                "Mouth": 13,
                "Left Eye": 33,
                "Right Eye": 263,
                "Left Brow": 70,
                "Right Brow": 300,
                "Chin": 152,
                "Left Ear": 234,
                "Right Ear": 454,
                "Hair": 10
            }

            for name, idx in part_indices.items():
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 255), 1)  # Smaller font and thinner stroke

    cv2.imshow("Gender & Face Parts", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
