import cv2
import os

class HaarFaceDetector:
    def __init__(self):
        # Load OpenCV's built-in frontal face Haar cascade
        # OpenCV stores this xml inside its installation files
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError("Haar cascade file was not found!")
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, gray_image):
        """
        Detect faces from a grayscale image
        :param gray_image: single channel grayscale numpy array (OpenCV format)
        :return: list of (x, y, w, h) bounding boxes
        """
        faces = self.detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces


# ---------------- TEST CODE ----------------
if __name__ == "__main__":
    detector = HaarFaceDetector()

    # Use webcam to test face detection
    cap = cv2.VideoCapture(0)
    print("Webcam test running. Press 'q' on the OpenCV window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read webcam frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_boxes = detector.detect_faces(gray)

        # Draw bounding boxes on detected faces
        for (x, y, w, h) in face_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Haar Face Detector Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test finished.")
