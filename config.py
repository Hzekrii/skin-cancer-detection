"""
Centralized configuration for the Skin Cancer Detection project.
Optimized for CPU training (no GPU).
"""

import os

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "HAM10000")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Créer les répertoires s'ils n'existent pas
for d in [MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────
IMAGE_SIZE = (128, 128)    # ⚡ Réduit (au lieu de 224x224) → 3x plus rapide
BATCH_SIZE = 16            # ⚡ Réduit (au lieu de 32) → moins de RAM
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
RANDOM_STATE = 42

# ─── Les 7 catégories du HAM10000 ────────────────────────
CLASSES = [
    "akiec",   # Actinic Keratoses
    "bcc",     # Basal Cell Carcinoma
    "bkl",     # Benign Keratosis
    "df",      # Dermatofibroma
    "mel",     # Melanoma
    "nv",      # Melanocytic Nevi
    "vasc"     # Vascular Lesions
]

CLASS_LABELS = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions"
}

NUM_CLASSES = len(CLASSES)

# ─── Training (optimisé CPU) ─────────────────────────────
EPOCHS = 15                    # ⚡ Réduit (au lieu de 30)
FINE_TUNE_EPOCHS = 10          # ⚡ Epochs pour le fine-tuning
LEARNING_RATE = 1e-3           # ⚡ Plus élevé (converge plus vite)
FINE_TUNE_LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.4
FINE_TUNE_AT_LAYER = 80       # ⚡ Ajusté pour MobileNetV2

# ─── Model ───────────────────────────────────────────────
BACKBONE = "mobilenet"         # ⚡ MobileNetV2 (10x plus léger que EfficientNetB3)
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_model.keras")