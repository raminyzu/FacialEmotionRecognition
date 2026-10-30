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


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    a1 = [x1, y1, x1+w1, y1+h1]
    a2 = [x2, y2, x2+w2, y2+h2]
    inter_x1 = max(a1[0], a2[0])
    inter_y1 = max(a1[1], a2[1])
    inter_x2 = min(a1[2], a2[2])
    inter_y2 = min(a1[3], a2[3])
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / union


def nms(boxes, iou_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        new_boxes = []
        for b in boxes:
            if iou(current, b) < iou_thresh:
                new_boxes.append(b)
        boxes = new_boxes
    return keep


def predict_emotion_from_image(img_path):
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        print(f"Cannot read image: {img_path}")
        return

    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_detector.detect_faces(gray)

    valid_faces = []
    min_face_size = 80
    for box in face_boxes:
        x, y, w, h = box
        if w >= min_face_size and h >= min_face_size:
            ratio = w / h
            if 0.75 < ratio < 1.25:
                valid_faces.append(box)

    # Run non-max suppression
    valid_faces = nms(valid_faces, iou_thresh=0.3)

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
import cv2
import numpy as np
from face_detector import HaarFaceDetector
from preprocessing import preprocess_face_image
from feature_extraction import combine_features
import joblib

# Path settings
TEST_IMG_PATH = "test_images/surprise.jpg"
MODEL_PATH = "models/svm_emotion_improved.pkl"
SCALER_PATH = "models/scaler_improved.pkl"

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Load trained model and scaler
svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
face_detector = HaarFaceDetector()


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    a1 = [x1, y1, x1+w1, y1+h1]
    a2 = [x2, y2, x2+w2, y2+h2]
    inter_x1 = max(a1[0], a2[0])
    inter_y1 = max(a1[1], a2[1])
    inter_x2 = min(a1[2], a2[2])
    inter_y2 = min(a1[3], a2[3])
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / union


def nms(boxes, iou_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        new_boxes = []
        for b in boxes:
            if iou(current, b) < iou_thresh:
                new_boxes.append(b)
        boxes = new_boxes
    return keep


def predict_emotion_from_image(img_path):
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        print(f"Cannot read image: {img_path}")
        return

    img_h, img_w = raw_img.shape[:2]
    img_center_x = img_w / 2
    img_center_y = img_h / 2

    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_detector.detect_faces(gray)

    valid_faces = []
    min_face_size = 80
    for box in face_boxes:
        x, y, w, h = box
        if w >= min_face_size and h >= min_face_size:
            ratio = w / h
            if 0.75 < ratio < 1.25:
                # calculate center of this candidate box
                box_cx = x + w/2
                box_cy = y + h/2
                distance = np.sqrt((box_cx - img_center_x)**2 + (box_cy - img_center_y)**2)
                # only keep boxes within threshold distance from image center
                max_allowed_dist = min(img_w, img_h) * 0.45
                if distance < max_allowed_dist:
                    valid_faces.append(box)

    valid_faces = nms(valid_faces, iou_thresh=0.3)

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
