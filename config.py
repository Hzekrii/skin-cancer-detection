"""
Centralized configuration for the Skin Cancer Detection project.
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
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
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

# ─── Training ────────────────────────────────────────────
EPOCHS = 30
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.4
FINE_TUNE_AT_LAYER = 100  # Dégeler à partir de cette couche

# ─── Model ───────────────────────────────────────────────
BACKBONE = "efficientnet"  # "resnet50" ou "efficientnet"
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_model.keras")