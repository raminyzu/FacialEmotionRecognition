import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_lbp_features(gray_face, radius=1, n_points=8):
    """
    Extract Local Binary Pattern (LBP) histogram features
    :param gray_face: normalized 48x48 grayscale face array (0~1)
    :param radius: LBP radius
    :param n_points: number of surrounding sampling points
    :return: 1D LBP feature vector
    """
    # Scale back from [0,1] to 0-255 integer for LBP
    img_uint8 = (gray_face * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, n_points, radius, method="uniform")
    # Compute histogram of LBP values
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    # Normalize histogram
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist

def extract_hog_features(gray_face):
    """
    Extract HOG (Histogram of Oriented Gradients) features
    :param gray_face: normalized 48x48 grayscale face array (0~1)
    :return: 1D HOG feature vector
    """
    hog_feat = hog(
        gray_face,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        block_norm="L2-Hys"
    )
    return hog_feat

def combine_features(gray_face):
    """Combine LBP + HOG into one single feature vector"""
    lbp_feat = extract_lbp_features(gray_face)
    hog_feat = extract_hog_features(gray_face)
    combined = np.concatenate([lbp_feat, hog_feat])
    return combined


# ---------------- TEST CODE ----------------
if __name__ == "__main__":
    import cv2
    from preprocessing import preprocess_face_image

    cap = cv2.VideoCapture(0)
    print("Feature extraction test. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_list = preprocess_face_image(frame)
        for face in face_list:
            feat = combine_features(face)
            print(f"Feature vector shape: {feat.shape}")
            # We just print feature length to confirm extraction works
        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Feature extraction test done")
