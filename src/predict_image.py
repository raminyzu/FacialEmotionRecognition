import cv2
import numpy as np
from face_detector import HaarFaceDetector
from preprocessing import preprocess_face_image
from feature_extraction import combine_features
import joblib

# Path settings
TEST_IMG_PATH = "test_images/happy.jpg"
MODEL_PATH = "models/svm_emotion_improved.pkl"
SCALER_PATH = "models/scaler_improved.pkl"

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Load trained model and scaler
svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
face_detector = HaarFaceDetector()


def predict_emotion_from_image(img_path):
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        print(f"Cannot read image: {img_path}")
        return

    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_detector.detect_faces(gray)

    # Filter out small false positive detections in background
    min_face_size = 65
    valid_faces = []
    for box in face_boxes:
        x, y, w, h = box
        if w >= min_face_size and h >= min_face_size:
            valid_faces.append(box)

    for (x, y, w, h) in valid_faces:
        face_list = preprocess_face_image(raw_img, single_roi=(x, y, w, h))
        preprocessed_face = face_list[0]
        feat = combine_features(preprocessed_face)
        feat_scaled = scaler.transform([feat])

        pred_idx = svm_model.predict(feat_scaled)[0]
        pred_emotion = emotion_labels[int(pred_idx)]

        cv2.rectangle(raw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(raw_img, f"{pred_emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Resizable OpenCV window
    cv2.namedWindow("Image Prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Prediction", 700, 700)
    cv2.imshow("Image Prediction", raw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_emotion_from_image(TEST_IMG_PATH)
