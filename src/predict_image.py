import cv2
import numpy as np
from face_detector import HaarFaceDetector
from preprocessing import preprocess_face_image
from feature_extraction import extract_lbp_hog_features
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

    # Preprocess, detect faces
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_detector.detect_faces(gray)

    for (x, y, w, h) in face_boxes:
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (48, 48))

        # Extract feature and scale
        feat = extract_lbp_hog_features(face_resized)
        feat_scaled = scaler.transform([feat])

        # Predict emotion
        pred_idx = svm_model.predict(feat_scaled)[0]
        pred_emotion = emotion_labels[int(pred_idx)]

        # Draw bounding box and text
        cv2.rectangle(raw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(raw_img, f"{pred_emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ========== This part controls window size ==========
    # Resize whole output image for smaller display window
    display_scale = 0.7
    display_img = cv2.resize(raw_img, (0,0), fx=display_scale, fy=display_scale)
    cv2.imshow("Image Prediction", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_emotion_from_image(TEST_IMG_PATH)
