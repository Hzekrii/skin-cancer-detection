"""
Main entry point for the Skin Cancer Classification project.

Usage:
    python main.py --mode all
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode gradcam
"""

import argparse
import os
import sys
import tensorflow as tf

# ─── System Info ─────────────────────────────────────────
print("=" * 60)
print("  SKIN CANCER CLASSIFICATION")
print("  Transfer Learning + Explainable AI (Grad-CAM)")
print("=" * 60)
print(f"  Python:     {sys.version.split()[0]}")
print(f"  TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPU:        {gpus if gpus else 'None (using CPU)'}")
print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Skin Cancer Classification"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "train", "evaluate", "gradcam"],
        help="Execution mode"
    )
    args = parser.parse_args()

    # ─── TRAIN ───────────────────────────────────────────
    if args.mode in ["all", "train"]:
        print("\n" + "🔥 " * 20)
        print("  STARTING TRAINING PIPELINE")
        print("🔥 " * 20)
        from src.train import train
        train()

    # ─── EVALUATE ────────────────────────────────────────
    if args.mode in ["all", "evaluate"]:
        print("\n" + "📊 " * 20)
        print("  STARTING EVALUATION")
        print("📊 " * 20)
        from src.evaluate import evaluate
        evaluate()

    # ─── GRAD-CAM ────────────────────────────────────────
    if args.mode in ["all", "gradcam"]:
        print("\n" + "🔍 " * 20)
        print("  GENERATING GRAD-CAM EXPLANATIONS")
        print("🔍 " * 20)
        import config
        from src.dataset import get_datasets
        from src.gradcam import (
            visualize_gradcam_grid,
            visualize_gradcam_single,
            GradCAM
        )

        model = tf.keras.models.load_model(config.MODEL_CHECKPOINT_PATH)
        _, _, _, _, test_df = get_datasets()

        # Grille Grad-CAM
        visualize_gradcam_grid(model, test_df, n_samples=12)

        # Exemples individuels (réutiliser le même objet GradCAM)
        gradcam_obj = None
        for i in range(3):
            sample = test_df.sample(1, random_state=i * 10).iloc[0]
            gradcam_obj = visualize_gradcam_single(
                model,
                sample["image_path"],
                true_label=sample["dx"],
                gradcam_obj=gradcam_obj,
                save_path=os.path.join(
                    config.FIGURES_DIR, f"gradcam_example_{i+1}.png"
                )
            )

    print("\n" + "=" * 60)
    print("  ✅ ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()