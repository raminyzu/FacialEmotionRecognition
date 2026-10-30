import cv2
import numpy as np
import joblib
from preprocessing import preprocess_face_image
from feature_extraction import combine_features
from face_detector import HaarFaceDetector

emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

MODEL_PATH = "models/svm_emotion_improved.pkl"
SCALER_PATH = "models/scaler_improved.pkl"

svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
face_detector = HaarFaceDetector()

cap = cv2.VideoCapture(0)
print("Improved webcam predictor running. Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = face_detector.detect_faces(gray)
    face_list = preprocess_face_image(frame)

    for box, face in zip(boxes, face_list):
        x, y, w, h = box
        feat_vec = combine_features(face)
        # MUST scale feature before prediction
        feat_scaled = scaler.transform([feat_vec])
        prediction = svm_model.predict(feat_scaled)[0]
        label_text = emotion_labels[prediction]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Improved Emotion Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
