import cv2

class HaarFaceDetector:
    def __init__(self):
        # Haar cascade for frontal face
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_faces(self, gray_img):
        faces = self.cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.15,
            minNeighbors=8,
            minSize=(80, 80)
        )
        return faces
