"""
Model architecture using Transfer Learning.
Supports ResNet50 and EfficientNetB3 as backbone.
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, EfficientNetB3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_model(backbone=config.BACKBONE, fine_tune=False):
    """
    Construit le modèle de classification avec Transfer Learning.

    Args:
        backbone: "resnet50" ou "efficientnet"
        fine_tune: Si True, dégèle les dernières couches du backbone

    Returns:
        Compiled tf.keras.Model
    """

    # ─── 1. Input Layer ──────────────────────────────────
    inputs = layers.Input(
        shape=(*config.IMAGE_SIZE, 3), 
        name="input_image"
    )

    # ─── 2. Feature Extractor (Pretrained Backbone) ──────
    if backbone == "resnet50":
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=inputs
        )
    elif backbone == "efficientnet":
        base_model = EfficientNetB3(
            weights="imagenet",
            include_top=False,
            input_tensor=inputs
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # ─── 3. Freeze / Unfreeze Strategy ──────────────────
    if fine_tune:
        # Dégeler les dernières couches pour le fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:config.FINE_TUNE_AT_LAYER]:
            layer.trainable = False
        
        trainable_layers = sum(
            1 for layer in base_model.layers if layer.trainable
        )
        frozen_layers = len(base_model.layers) - trainable_layers
        print(f"[MODEL] Fine-tuning mode:")
        print(f"  Frozen layers:    {frozen_layers}")
        print(f"  Trainable layers: {trainable_layers}")
    else:
        # Geler tout le backbone (feature extraction seulement)
        base_model.trainable = False
        print(f"[MODEL] Feature extraction mode (backbone frozen)")

    # ─── 4. Classification Head ──────────────────────────
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(config.DROPOUT_RATE / 2, name="dropout_2")(x)
    outputs = layers.Dense(
        config.NUM_CLASSES, 
        activation="softmax", 
        name="predictions"
    )(x)

    # ─── 5. Build Model ─────────────────────────────────
    model = Model(
        inputs=inputs, 
        outputs=outputs, 
        name=f"SkinCancer_{backbone}"
    )

    # ─── 6. Compile ──────────────────────────────────────
    lr = config.FINE_TUNE_LEARNING_RATE if fine_tune else config.LEARNING_RATE

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ]
    )

    # ─── 7. Summary ─────────────────────────────────────
    trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    total_count = trainable_count + non_trainable_count

    print(f"\n[MODEL] {model.name}")
    print(f"  Backbone:           {backbone}")
    print(f"  Total params:       {total_count:>12,}")
    print(f"  Trainable params:   {trainable_count:>12,}")
    print(f"  Non-trainable:      {non_trainable_count:>12,}")
    print(f"  Learning rate:      {lr}")
    print(f"  Dropout rate:       {config.DROPOUT_RATE}")

    return model


def get_callbacks():
    """Retourne les callbacks d'entraînement."""
    
    log_dir = os.path.join(config.RESULTS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        # Sauvegarde du meilleur modèle
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_CHECKPOINT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        # Arrêt anticipé si pas d'amélioration
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Réduction du learning rate sur plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    
    print(f"\n[INFO] Callbacks configured:")
    print(f"  - ModelCheckpoint (monitor: val_accuracy)")
    print(f"  - EarlyStopping (patience: 7)")
    print(f"  - ReduceLROnPlateau (patience: 3, factor: 0.5)")
    
    return callbacks


# ─── Test rapide ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Building model (feature extraction mode)...")
    print("=" * 60)
    model = build_model(backbone="efficientnet", fine_tune=False)
    model.summary(show_trainable=True)
    
    print("\n" + "=" * 60)
    print("  Building model (fine-tuning mode)...")
    print("=" * 60)
    model_ft = build_model(backbone="efficientnet", fine_tune=True)