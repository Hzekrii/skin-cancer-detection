"""
Training pipeline: Phase 1 (Feature Extraction) + Phase 2 (Fine-Tuning).
Optimized for CPU training.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_datasets
from src.model import build_model, get_callbacks


def plot_training_history(history, phase="initial", save=True):
    """Trace les courbes d'entraînement."""

    available_metrics = [
        k for k in history.history.keys() if not k.startswith("val_")
    ]

    n_metrics = len(available_metrics)
    cols = 2
    rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    fig.suptitle(
        f"Training History — {phase.replace('_', ' ').title()}",
        fontsize=16, fontweight="bold"
    )

    if rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    for idx, metric in enumerate(available_metrics):
        ax = axes_flat[idx]

        train_values = history.history[metric]
        val_key = f"val_{metric}"

        ax.plot(
            train_values,
            label="Train",
            linewidth=2,
            color="#2196F3"
        )

        if val_key in history.history:
            val_values = history.history[val_key]
            ax.plot(
                val_values,
                label="Validation",
                linewidth=2,
                linestyle="--",
                color="#FF5722"
            )

        ax.set_title(metric.capitalize(), fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    if save:
        path = os.path.join(
            config.FIGURES_DIR,
            f"training_history_{phase}.png"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {path}")
    plt.close()


def train():
    """Pipeline d'entraînement complet en 2 phases."""

    # ═══════════════════════════════════════════════════════
    # PHASE 1 : Chargement des données
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 1: DATA LOADING")
    print("=" * 60)

    train_ds, val_ds, test_ds, class_weights, test_df = get_datasets()

    # ═══════════════════════════════════════════════════════
    # PHASE 2 : Feature Extraction (backbone gelé)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 2: FEATURE EXTRACTION (Backbone Frozen)")
    print("=" * 60)

    model = build_model(backbone=config.BACKBONE, fine_tune=False)
    callbacks = get_callbacks()

    print(f"\n[TRAIN] Starting Phase 2...")
    print(f"  Epochs:     {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Image size: {config.IMAGE_SIZE}")
    print(f"  Backbone:   {config.BACKBONE}")
    print(f"  ⏱️  This may take 30-60 minutes on CPU...\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    plot_training_history(history, phase="feature_extraction")

    # Résultats Phase 2
    print("\n[EVAL] Phase 2 — Validation results:")
    val_results = model.evaluate(val_ds, verbose=0)
    for name, value in zip(model.metrics_names, val_results):
        print(f"  {name:>12s}: {value:.4f}")

    # ═══════════════════════════════════════════════════════
    # PHASE 3 : Fine-Tuning
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 3: FINE-TUNING (Top Layers Unfrozen)")
    print("=" * 60)

    # Charger le meilleur modèle
    print(f"[INFO] Loading best model from Phase 2...")
    model = tf.keras.models.load_model(config.MODEL_CHECKPOINT_PATH)

    # Dégeler les dernières couches du backbone
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is not None:
        base_model.trainable = True
        for layer in base_model.layers[:config.FINE_TUNE_AT_LAYER]:
            layer.trainable = False

        trainable = sum(1 for l in base_model.layers if l.trainable)
        frozen = sum(1 for l in base_model.layers if not l.trainable)
        print(f"[INFO] Backbone: {frozen} frozen, {trainable} trainable")
    else:
        print("[WARNING] Could not find base model")

    # Recompiler avec learning rate plus petit
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.FINE_TUNE_LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ]
    )

    fine_tune_epochs = config.FINE_TUNE_EPOCHS
    print(f"\n[TRAIN] Starting Phase 3...")
    print(f"  Epochs: {fine_tune_epochs}")
    print(f"  LR:    {config.FINE_TUNE_LEARNING_RATE}")
    print(f"  ⏱️  This may take 20-40 minutes on CPU...\n")

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )

    plot_training_history(history_fine, phase="fine_tuning")

    # Résultats finaux
    print("\n[EVAL] Final — Validation results:")
    val_results = model.evaluate(val_ds, verbose=0)
    for name, value in zip(model.metrics_names, val_results):
        print(f"  {name:>12s}: {value:.4f}")

    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print(f"  Model saved: {config.MODEL_CHECKPOINT_PATH}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    model = train()