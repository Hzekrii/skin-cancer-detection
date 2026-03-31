"""
Model evaluation: confusion matrix, classification report,
per-class metrics, and ROC curves.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_datasets


def load_best_model():
    """Charge le meilleur modèle sauvegardé."""
    if not os.path.exists(config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No model found at {config.MODEL_CHECKPOINT_PATH}\n"
            f"Please train the model first: python main.py --mode train"
        )
    model = tf.keras.models.load_model(config.MODEL_CHECKPOINT_PATH)
    print(f"[INFO] Model loaded from {config.MODEL_CHECKPOINT_PATH}")
    return model


def predict_on_test(model, test_ds):
    """Génère les prédictions sur le test set."""
    y_true = []
    y_pred_proba = []

    print("[INFO] Generating predictions on test set...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_proba.append(preds)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred_proba = np.concatenate(y_pred_proba, axis=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)

    print(f"[INFO] Total test samples: {len(y_true_labels)}")
    return y_true_labels, y_pred, y_pred_proba


def plot_confusion_matrix(y_true, y_pred, save=True):
    """Trace la matrice de confusion (brute + normalisée)."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Matrice brute (counts)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASSES, yticklabels=config.CLASSES,
        ax=axes[0], cbar_kws={"shrink": 0.8}
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)

    # Matrice normalisée (pourcentages)
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=config.CLASSES, yticklabels=config.CLASSES,
        ax=axes[1], cbar_kws={"shrink": 0.8}
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)

    plt.tight_layout()
    
    if save:
        path = os.path.join(config.FIGURES_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {path}")
    plt.close()


def plot_roc_curves(y_true, y_pred_proba, save=True):
    """Trace les courbes ROC One-vs-Rest pour chaque classe."""
    y_true_bin = label_binarize(y_true, classes=range(config.NUM_CLASSES))

    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, config.NUM_CLASSES))

    for i, (cls, color) in enumerate(zip(config.CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color=color, linewidth=2,
            label=f"{config.CLASS_LABELS[cls]} (AUC = {roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — One vs Rest", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR, "roc_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {path}")
    plt.close()


def plot_per_class_metrics(y_true, y_pred, save=True):
    """Visualise precision, recall, f1 par classe."""
    report = classification_report(
        y_true, y_pred,
        target_names=config.CLASSES,
        output_dict=True
    )

    classes = config.CLASSES
    precision = [report[c]["precision"] for c in classes]
    recall = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#2196F3")
    bars2 = ax.bar(x, recall, width, label="Recall", color="#FF9800")
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#4CAF50")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Valeurs sur les barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7
            )

    plt.tight_layout()
    
    if save:
        path = os.path.join(config.FIGURES_DIR, "per_class_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {path}")
    plt.close()


def print_classification_report(y_true, y_pred):
    """Affiche et sauvegarde le rapport de classification."""
    report = classification_report(
        y_true, y_pred,
        target_names=[config.CLASS_LABELS[c] for c in config.CLASSES],
        digits=4
    )

    print("\n" + "=" * 70)
    print("  CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)

    # Sauvegarder
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("SKIN CANCER CLASSIFICATION — TEST SET RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"[INFO] Saved: {report_path}")

    return report


def evaluate():
    """Pipeline d'évaluation complet."""
    
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    # Charger le modèle
    model = load_best_model()
    
    # Charger les données
    _, _, test_ds, _, test_df = get_datasets()

    # Prédictions
    y_true, y_pred, y_pred_proba = predict_on_test(model, test_ds)

    # Métriques
    print_classification_report(y_true, y_pred)
    
    # Visualisations
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true, y_pred_proba)
    plot_per_class_metrics(y_true, y_pred)

    # Accuracy globale
    accuracy = np.mean(y_true == y_pred)
    print(f"\n{'='*50}")
    print(f"  TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")

    return y_true, y_pred, y_pred_proba


# ─── Point d'entrée ──────────────────────────────────────
if __name__ == "__main__":
    evaluate()