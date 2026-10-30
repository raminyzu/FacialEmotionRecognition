import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

if __name__ == "__main__":
    y_test = joblib.load("outputs/y_test.pkl")
    y_pred = joblib.load("outputs/y_pred.pkl")

    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Test Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrix plot
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.title(f"Confusion Matrix (Accuracy = {acc:.4f})")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=300)
    plt.show()

    # Per class accuracy
    class_acc = []
    for cls in range(7):
        mask = (y_test == cls)
        if np.sum(mask) == 0:
            ca = 0
        else:
            ca = np.mean(y_pred[mask] == cls)
        class_acc.append(ca)

    plt.figure(figsize=(8,5))
    bars = plt.bar(emotion_labels, class_acc, color="#4472C4")
    plt.ylim(0, 1)
    plt.ylabel("Per-class Accuracy")
    plt.title("Per-Class Emotion Recognition Accuracy")
    for bar, val in zip(bars, class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f"{val:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig("outputs/per_class_accuracy.png", dpi=300)
    plt.show()
