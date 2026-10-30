import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import combine_features

emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

DATA_PATH = os.path.join("data", "fer2013.csv")
MODEL_SAVE_PATH = os.path.join("models", "svm_emotion_improved.pkl")
SCALER_SAVE_PATH = os.path.join("models", "scaler_improved.pkl")
CONFUSION_PLOT_PATH = os.path.join("results", "confusion_matrix_improved.png")

def load_fer2013(csv_path):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    expected_feat_len = None
    for idx, row in df.iterrows():
        try:
            pixels = np.array([int(p) for p in row["pixels"].split(" ")])
            face_img = pixels.reshape((48,48)).astype(np.float32)/255.0
            feat_vec = combine_features(face_img)

            if expected_feat_len is None:
                expected_feat_len = len(feat_vec)
            if len(feat_vec) == expected_feat_len:
                X.append(feat_vec)
                y.append(row["emotion"])
        except Exception as e:
            continue
    return np.array(X), np.array(y)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading and extracting features from FER2013 dataset, this will take some time...")
    X, y = load_fer2013(DATA_PATH)
    print(f"Total valid samples: {X.shape[0]}, feature dimension: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training improved SVM...")
    # gamma="scale" (safe default), C=1.5, balanced weights
    svm_clf = SVC(kernel="rbf", C=1.5, gamma="scale", class_weight="balanced")
    svm_clf.fit(X_train_scaled, y_train)

    y_pred = svm_clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(emotion_labels.values())))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=emotion_labels.values(),
                yticklabels=emotion_labels.values())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (LBP+HOG + Scaled + Balanced SVM)")
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_PATH, dpi=300)
    plt.close()

    joblib.dump(svm_clf, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\nModel saved: {MODEL_SAVE_PATH}")
    print(f"Scaler saved: {SCALER_SAVE_PATH}")
import joblib
joblib.dump(y_test, "outputs/y_test.pkl")
joblib.dump(y_pred, "outputs/y_pred.pkl")
