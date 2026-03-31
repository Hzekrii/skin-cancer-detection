"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
for Explainable AI in skin cancer classification.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations
from Deep Networks via Gradient-based Localization", ICCV 2017.
"""

import os
import sys
import numpy as np
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
        """
        Args:
            model: Trained Keras model
            layer_name: Name of the target conv layer.
                        If None, auto-detects the last conv layer.
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        print(f"[Grad-CAM] Target layer: {self.layer_name}")

    def _find_target_layer(self):
        """Trouve automatiquement la dernière couche convolutive."""
        # Chercher dans le modèle complet
        for layer in reversed(self.model.layers):
            # Vérifier si c'est une couche Conv2D
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            # Pour les modèles imbriqués (EfficientNet, ResNet)
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        return sub_layer.name
        
        raise ValueError("No convolutional layer found in the model!")

    def _get_target_layer(self):
        """Récupère l'objet layer par son nom."""
        # Chercher d'abord dans le modèle principal
        try:
            return self.model.get_layer(self.layer_name)
        except ValueError:
            pass
        
        # Chercher dans les sous-modèles
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    return layer.get_layer(self.layer_name)
                except ValueError:
                    continue
        
        raise ValueError(f"Layer '{self.layer_name}' not found!")

    def compute_heatmap(self, image, pred_index=None, eps=1e-8):
        """
        Calcule la heatmap Grad-CAM.

        Args:
            image: Image prétraitée de shape (1, 224, 224, 3)
            pred_index: Index de la classe à expliquer.
                        Si None, utilise la classe prédite.
            eps: Valeur epsilon pour stabilité numérique.

        Returns:
            heatmap: np.array normalisé [0, 1]
        """
        target_layer = self._get_target_layer()
        
        # Modèle qui sort les activations de la couche cible + prédictions
        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output]
        )

        # Forward pass + calcul des gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Gradients de la sortie par rapport aux feature maps
        grads = tape.gradient(class_channel, conv_outputs)

        # Global Average Pooling des gradients → importance de chaque filtre
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Pondérer chaque feature map par son importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU (on ne veut que les activations positives)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalisation [0, 1]
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + eps)

        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4, colormap="jet"):
        """
        Superpose la heatmap Grad-CAM sur l'image originale.

        Args:
            heatmap: Heatmap normalisée [0, 1]
            original_image: Image originale (H, W, 3) uint8
            alpha: Transparence
            colormap: Colormap matplotlib

        Returns:
            superimposed_img, heatmap_colored
        """
        # Redimensionner la heatmap
        h, w = original_image.shape[:2]
        heatmap_resized = np.uint8(255 * heatmap)
        heatmap_resized = np.array(
            Image.fromarray(heatmap_resized).resize((w, h), Image.BILINEAR)
        )

        # Appliquer la colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized / 255.0)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)

        # Superposer
        superimposed = np.uint8(
            heatmap_colored * alpha + original_image * (1 - alpha)
        )

        return superimposed, heatmap_colored


def preprocess_single_image(image_path):
    """Charge et prétraite une seule image pour l'inférence."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img_display = tf.cast(img, tf.uint8).numpy()  # Pour affichage
    img_normalized = tf.cast(img, tf.float32) / 255.0
    img_batch = tf.expand_dims(img_normalized, axis=0)
    return img_batch, img_display


def visualize_gradcam_single(model, image_path, true_label=None, save_path=None):
    """
    Visualise Grad-CAM pour une seule image.
    Affiche: Original | Heatmap | Superposition
    """
    # Préparer l'image
    img_batch, img_display = preprocess_single_image(image_path)

    # Prédiction
    preds = model.predict(img_batch, verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]
    pred_label = config.CLASSES[pred_class]
    pred_full = config.CLASS_LABELS[pred_label]

    # Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam.compute_heatmap(img_batch, pred_index=pred_class)
    superimposed, heatmap_colored = gradcam.overlay_heatmap(heatmap, img_display)

    # ─── Visualisation ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
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


def visualize_gradcam_grid(model, test_df, n_samples=12, save=True):
    """
    Grille de Grad-CAM pour plusieurs échantillons du test set.
    Sélectionne des images de différentes classes.
    """
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
    
    samples = pd.concat(sampled_dfs).head(n_samples)

    # Grad-CAM
    gradcam = GradCAM(model)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.5 * rows))
    axes = axes.flatten() if n_samples > cols else [axes] if cols == 1 else axes.flatten()

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

    # Masquer les axes inutilisés
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


# On a besoin de pandas pour visualize_gradcam_grid
import pandas as pd


# ─── Test ────────────────────────────────────────────────
if __name__ == "__main__":
    from src.dataset import get_datasets

    model = tf.keras.models.load_model(config.MODEL_CHECKPOINT_PATH)
    _, _, _, _, test_df = get_datasets()

    # Une seule image
    sample = test_df.sample(1, random_state=42).iloc[0]
    visualize_gradcam_single(
        model,
        sample["image_path"],
        true_label=sample["dx"],
        save_path=os.path.join(config.FIGURES_DIR, "gradcam_single.png")
    )

    # Grille
    visualize_gradcam_grid(model, test_df, n_samples=12)