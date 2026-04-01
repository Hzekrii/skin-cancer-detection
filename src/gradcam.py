"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
for Explainable AI in skin cancer classification.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations
from Deep Networks via Gradient-based Localization", ICCV 2017.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GradCAM:
    """
    Grad-CAM implementation for visualizing which regions
    of the image contribute most to the model's prediction.
    """

    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        self._build_grad_model()
        print(f"[Grad-CAM] Target layer: {self.layer_name}")

    def _find_target_layer(self):
        """Trouve la dernière couche Conv2D dans le modèle."""
        # Chercher dans les sous-modèles (backbone)
        for layer in self.model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                for sub_layer in reversed(layer.layers):
                    if 'conv' in sub_layer.name.lower() and len(sub_layer.output.shape) == 4:
                        return sub_layer.name

        # Chercher directement dans le modèle
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower() and len(layer.output.shape) == 4:
                return layer.name

        raise ValueError("No convolutional layer found!")

    def _get_target_layer_output(self):
        """Récupère le tensor de sortie de la couche cible."""
        # Chercher dans les sous-modèles d'abord
        for layer in self.model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                try:
                    target = layer.get_layer(self.layer_name)
                    return target.output
                except ValueError:
                    continue

        # Chercher dans le modèle principal
        try:
            return self.model.get_layer(self.layer_name).output
        except ValueError:
            raise ValueError(f"Layer '{self.layer_name}' not found!")

    def _build_grad_model(self):
        """Construit le modèle pour Grad-CAM."""
        target_output = self._get_target_layer_output()
        self.grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[target_output, self.model.output]
        )

    def compute_heatmap(self, image, pred_index=None, eps=1e-8):
        """
        Calcule la heatmap Grad-CAM.

        Args:
            image: Image (1, H, W, 3) normalisée
            pred_index: Classe à expliquer (None = classe prédite)
            eps: Stabilité numérique

        Returns:
            heatmap: np.array normalisé [0, 1]
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Gradients
        grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            print("[WARNING] Gradients are None, returning empty heatmap")
            return np.zeros((7, 7))

        # Global Average Pooling des gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Pondération
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalisation
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / (max_val + eps)

        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4, colormap="jet"):
        """Superpose la heatmap sur l'image originale."""
        h, w = original_image.shape[:2]
        
        # Redimensionner
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_resized = np.array(
            Image.fromarray(heatmap_uint8).resize((w, h), Image.BILINEAR)
        )

        # Colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized / 255.0)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)

        # Superposition
        original = original_image.astype(np.float32)
        overlay = heatmap_colored.astype(np.float32)
        superimposed = np.uint8(overlay * alpha + original * (1 - alpha))

        return superimposed, heatmap_colored


def preprocess_single_image(image_path):
    """Charge et prétraite une image pour l'inférence."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img_display = tf.cast(img, tf.uint8).numpy()
    img_normalized = tf.cast(img, tf.float32) / 255.0
    img_batch = tf.expand_dims(img_normalized, axis=0)
    return img_batch, img_display


def visualize_gradcam_single(model, image_path, true_label=None, 
                              gradcam_obj=None, save_path=None):
    """Visualise Grad-CAM pour une seule image."""
    img_batch, img_display = preprocess_single_image(image_path)

    # Prédiction
    preds = model.predict(img_batch, verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]
    pred_label = config.CLASSES[pred_class]
    pred_full = config.CLASS_LABELS[pred_label]

    # Grad-CAM
    if gradcam_obj is None:
        gradcam_obj = GradCAM(model)
    
    heatmap = gradcam_obj.compute_heatmap(img_batch, pred_index=pred_class)
    superimposed, heatmap_colored = gradcam_obj.overlay_heatmap(
        heatmap, img_display
    )

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", aspect="auto")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    # Titre
    title = f"Predicted: {pred_full} ({confidence:.1%})"
    if true_label is not None:
        true_full = config.CLASS_LABELS.get(true_label, true_label)
        correct = true_label == pred_label
        color = "green" if correct else "red"
        symbol = "✓" if correct else "✗"
        title += f"\nTrue: {true_full} [{symbol}]"
        fig.suptitle(title, fontsize=14, color=color, fontweight="bold")
    else:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    plt.close()

    return gradcam_obj  # Réutiliser pour éviter de recréer


def visualize_gradcam_grid(model, test_df, n_samples=12, save=True):
    """Grille de Grad-CAM pour plusieurs échantillons."""
    
    # Échantillonner de chaque classe
    samples_per_class = max(1, n_samples // config.NUM_CLASSES)
    sampled_dfs = []

    for cls in config.CLASSES:
        cls_df = test_df[test_df["dx"] == cls]
        n = min(samples_per_class, len(cls_df))
        if n > 0:
            sampled_dfs.append(
                cls_df.sample(n=n, random_state=config.RANDOM_STATE)
            )

    samples = pd.concat(sampled_dfs).head(n_samples).reset_index(drop=True)

    # Créer le GradCAM UNE SEULE FOIS
    print("[INFO] Initializing Grad-CAM...")
    gradcam = GradCAM(model)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.5 * rows))
    axes = axes.flatten()

    print(f"[INFO] Generating Grad-CAM for {len(samples)} samples...")
    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= n_samples:
            break

        img_batch, img_display = preprocess_single_image(row["image_path"])
        preds = model.predict(img_batch, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = preds[0][pred_class]

        heatmap = gradcam.compute_heatmap(img_batch, pred_index=pred_class)
        superimposed, _ = gradcam.overlay_heatmap(heatmap, img_display)

        axes[idx].imshow(superimposed)

        pred_name = config.CLASSES[pred_class]
        true_name = row["dx"]
        correct = pred_name == true_name
        color = "green" if correct else "red"
        symbol = "✓" if correct else "✗"

        axes[idx].set_title(
            f"True: {true_name} | Pred: {pred_name}\n"
            f"Conf: {confidence:.0%} [{symbol}]",
            fontsize=10, color=color, fontweight="bold"
        )
        axes[idx].axis("off")

        if (idx + 1) % 4 == 0:
            print(f"  Processed {idx + 1}/{n_samples}")

    for idx in range(len(samples), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Grad-CAM Explanations — Test Samples",
        fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR, "gradcam_grid.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {path}")
    plt.close()


if __name__ == "__main__":
    from src.dataset import get_datasets

    model = tf.keras.models.load_model(config.MODEL_CHECKPOINT_PATH)
    _, _, _, _, test_df = get_datasets()

    visualize_gradcam_grid(model, test_df, n_samples=12)