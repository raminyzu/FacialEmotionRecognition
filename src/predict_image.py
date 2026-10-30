import cv2
import numpy as np
import joblib
from pathlib import Path
from face_detector import HaarFaceDetector
from preprocessing import preprocess_face_image
from feature_extraction import extract_lbp_features, extract_hog_features

# Base project root (E:\FacialEmotionRecognition)
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "svm_emotion_improved.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler_improved.pkl"
TEST_IMG_PATH = BASE_DIR / "test_images" / "happy.jpg"

face_detector = HaarFaceDetector()
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Load your saved improved model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def extract_combined_features(face_gray):
    lbp_feat = extract_lbp_features(face_gray)
    hog_feat = extract_hog_features(face_gray)
    combined = np.concatenate([lbp_feat, hog_feat])
    return combined

def predict_emotion_from_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print("Cannot read image file, check the path!")
        return

    faces = preprocess_face_image(img)
    for face in faces:
        feat = extract_combined_features(face)
        feat_scaled = scaler.transform([feat])
        pred_idx = model.predict(feat_scaled)[0]
        pred_emotion = emotion_labels[pred_idx]

        cv2.putText(img, f"{pred_emotion}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    cv2.imshow("Image Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_emotion_from_image(TEST_IMG_PATH)
