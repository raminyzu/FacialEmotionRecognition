import cv2
import numpy as np
import joblib
from face_detector import HaarFaceDetector
from preprocessing import preprocess_face_image
from feature_extraction import combine_features

MODEL_PATH = "models/svm_emotion_improved.pkl"
SCALER_PATH = "models/scaler_improved.pkl"
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

face_detector = HaarFaceDetector()
svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    a1 = [x1, y1, x1 + w1, y1 + h1]
    a2 = [x2, y2, x2 + w2, y2 + h2]
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
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
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


def webcam_emotion():
    cap = cv2.VideoCapture(0)
    print("Webcam emotion predictor running, press q to close window")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        img_cx = w / 2
        img_cy = h / 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_boxes = face_detector.detect_faces(gray)
        valid_faces = []
        min_face_size = 80

        for box in face_boxes:
            x, y, bw, bh = box
            if bw >= min_face_size and bh >= min_face_size:
                ratio = bw / bh
                if 0.75 < ratio < 1.25:
                    box_cx = x + bw / 2
                    box_cy = y + bh / 2
                    dist = np.sqrt((box_cx - img_cx) ** 2 + (box_cy - img_cy) ** 2)
                    max_dist = min(w, h) * 0.45
                    if dist < max_dist:
                        valid_faces.append(box)

        valid_faces = nms(valid_faces, 0.3)

        for (x, y, bw, bh) in valid_faces:
            face_list = preprocess_face_image(frame, single_roi=(x, y, bw, bh))
            face = face_list[0]
            feat = combine_features(face)
            feat_scaled = scaler.transform([feat])
            pred = svm_model.predict(feat_scaled)[0]
            label = emotion_labels[int(pred)]

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Webcam Emotion Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam_emotion()
