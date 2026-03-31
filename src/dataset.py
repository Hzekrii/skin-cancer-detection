"""
Dataset loading, preprocessing, augmentation, and class balancing
for the HAM10000 skin lesion dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter le dossier racine au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_metadata():
    """
    Charge le fichier CSV HAM10000_metadata.csv et construit
    le chemin complet vers chaque image.
    """
    csv_path = os.path.join(config.DATA_DIR, "HAM10000_metadata.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata file not found at {csv_path}\n"
            f"Please download HAM10000 dataset and place it in {config.DATA_DIR}"
        )
    
    df = pd.read_csv(csv_path)

    # Construire un dictionnaire image_id → chemin
    # Les images peuvent être dans différents sous-dossiers
    image_id_to_path = {}
    
    # Chercher dans tous les sous-dossiers de DATA_DIR
    for root, dirs, files in os.walk(config.DATA_DIR):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_id = os.path.splitext(fname)[0]
                image_id_to_path[img_id] = os.path.join(root, fname)

    df["image_path"] = df["image_id"].map(image_id_to_path)
    
    # Supprimer les lignes sans image trouvée
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"[WARNING] {missing} images not found, dropping them.")
    df = df.dropna(subset=["image_path"])

    # Encoder les labels en entiers
    df["label"] = df["dx"].map({c: i for i, c in enumerate(config.CLASSES)})

    print(f"\n{'='*50}")
    print(f"[INFO] Dataset loaded successfully!")
    print(f"[INFO] Total samples: {len(df)}")
    print(f"[INFO] Class distribution:")
    print(df["dx"].value_counts().to_string())
    print(f"{'='*50}\n")

    return df


def plot_class_distribution(df, save=True):
    """Visualise la distribution des classes."""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        data=df, x="dx", order=config.CLASSES, 
        palette="viridis", hue="dx", legend=False
    )
    plt.title("Class Distribution — HAM10000 Dataset", fontsize=14, fontweight="bold")
    plt.xlabel("Lesion Category", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)

    # Ajouter les valeurs sur les barres
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    plt.tight_layout()
    if save:
        path = os.path.join(config.FIGURES_DIR, "class_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {path}")
    plt.close()


def plot_sample_images(df, n_per_class=3, save=True):
    """Affiche des échantillons d'images de chaque classe."""
    fig, axes = plt.subplots(
        n_per_class, config.NUM_CLASSES, 
        figsize=(3 * config.NUM_CLASSES, 3 * n_per_class)
    )
    
    for col, cls in enumerate(config.CLASSES):
        samples = df[df["dx"] == cls].sample(
            n=min(n_per_class, len(df[df["dx"] == cls])),
            random_state=config.RANDOM_STATE
        )
        for row, (_, sample) in enumerate(samples.iterrows()):
            img = plt.imread(sample["image_path"])
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(
                    config.CLASS_LABELS[cls], fontsize=9, fontweight="bold"
                )

    plt.suptitle("Sample Images per Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save:
        path = os.path.join(config.FIGURES_DIR, "sample_images.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {path}")
    plt.close()


def split_data(df):
    """Stratified train/validation/test split."""
    # Premier split : séparer le test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SPLIT,
        stratify=df["label"],
        random_state=config.RANDOM_STATE
    )

    # Deuxième split : séparer validation du train
    relative_val_size = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=config.RANDOM_STATE
    )

    print(f"[INFO] Split sizes:")
    print(f"  Train:      {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def compute_weights(train_df):
    """Calcule les poids de classe pour gérer le déséquilibre."""
    labels = train_df["label"].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n[INFO] Class weights (to handle imbalance):")
    for idx, cls in enumerate(config.CLASSES):
        print(f"  {cls:>6s} ({config.CLASS_LABELS[cls]:>25s}): {class_weight_dict[idx]:.3f}")
    
    return class_weight_dict


def create_data_augmentation():
    """Pipeline d'augmentation de données."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


def _parse_image(image_path, label):
    """Charge et prétraite une seule image."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0  # Normalisation [0, 1]
    return img, label


def create_dataset(df, augment=False, shuffle=True):
    """Crée un tf.data.Dataset à partir d'un DataFrame."""
    paths = df["image_path"].values
    labels = tf.keras.utils.to_categorical(
        df["label"].values, num_classes=config.NUM_CLASSES
    )

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df), 
            seed=config.RANDOM_STATE
        )

    dataset = dataset.map(
        _parse_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Appliquer l'augmentation sur le dataset d'entraînement
    if augment:
        augmentation_layer = create_data_augmentation()
        dataset = dataset.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def get_datasets():
    """
    Pipeline complet : charge les données, split, crée les datasets.
    
    Returns:
        train_ds, val_ds, test_ds, class_weights, test_df
    """
    print("\n" + "=" * 60)
    print("  DATA PREPARATION")
    print("=" * 60)
    
    # Charger les métadonnées
    df = load_metadata()
    
    # Visualisations
    plot_class_distribution(df)
    plot_sample_images(df)
    
    # Split
    train_df, val_df, test_df = split_data(df)
    
    # Poids de classe
    class_weights = compute_weights(train_df)
    
    # Créer les datasets TF
    print("\n[INFO] Creating TensorFlow datasets...")
    train_ds = create_dataset(train_df, augment=True, shuffle=True)
    val_ds = create_dataset(val_df, augment=False, shuffle=False)
    test_ds = create_dataset(test_df, augment=False, shuffle=False)
    
    # Vérification
    for images, labels in train_ds.take(1):
        print(f"\n[INFO] Batch verification:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Pixel range:  [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
    
    print("\n[✓] Data preparation complete!")
    return train_ds, val_ds, test_ds, class_weights, test_df


# ─── Test rapide ──────────────────────────────────────────
if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_weights, test_df = get_datasets()