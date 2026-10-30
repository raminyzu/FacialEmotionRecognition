import cv2
import numpy as np
from face_detector import HaarFaceDetector

# Initialize our face detector once
face_detector = HaarFaceDetector()

def preprocess_face_image(raw_bgr_img, target_size=(48, 48)):
    """
    Full preprocessing pipeline for a single input image
    :param raw_bgr_img: Input image read by OpenCV (BGR format)
    :param target_size: Fixed size we use for FER-2013 (48,48)
    :return: List of preprocessed grayscale face images (one per detected face)
    """
    gray_full = cv2.cvtColor(raw_bgr_img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_detector.detect_faces(gray_full)
    processed_faces = []

    for (x, y, w, h) in face_boxes:
        # Crop the face region
        face_crop = gray_full[y:y + h, x:x + w]
        face_crop = cv2.equalizeHist(face_crop)

        # Resize to fixed 48x48
        face_resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
        # Normalize pixel values from [0,255] → [0,1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        processed_faces.append(face_normalized)

    return processed_faces


# ---------------- TEST CODE ----------------
if __name__ == "__main__":
    # Test preprocessing with webcam feed
    cap = cv2.VideoCapture(0)
    print("Preprocessing test running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam read error")
            break

        face_list = preprocess_face_image(frame)
        for idx, face in enumerate(face_list):
            # Convert normalized [0,1] back to 0-255 for display
            display_face = (face * 255).astype(np.uint8)
            cv2.imshow(f"Cropped Face {idx}", display_face)

        cv2.imshow("Original Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Preprocessing test completed.")
