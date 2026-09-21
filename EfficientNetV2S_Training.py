# ============================================================
# FRESH EFFICIENTNETV2S TRAINING — V2 DATASET
#
# Training design:
#
#   Mendeley images:
#       70% training
#       15% validation
#       15% internal test
#
#   Added External images:
#       100% training only
#
#   Untouched Kaggle test images:
#       external evaluation only
#
# Important:
#   - No previous MRI model is loaded.
#   - EfficientNetV2S starts from ImageNet weights.
#   - Mendeley split is performed by case ID.
#   - Related/augmented Mendeley images stay in one split.
# ============================================================


# ============================================================
# 1. MOUNT GOOGLE DRIVE
# ============================================================

from google.colab import drive
drive.mount("/content/drive")


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import re
import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import (
    layers,
    models,
    optimizers,
    mixed_precision
)

from tensorflow.keras.applications import EfficientNetV2S

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

V2_DATASET_PATH = (
    "/content/drive/MyDrive/Final Project/"
    "V2_MRI_images_balanced_1400"
)

EXTERNAL_TEST_ROOT = (
    "/content/drive/MyDrive/Final Project/"
    "external_MRI_test/retraining"
)

OUTPUT_FOLDER = (
    "/content/drive/MyDrive/Final Project/"
    "Fresh_EfficientNetV2S_V2"
)

FINAL_MODEL_PATH = os.path.join(
    OUTPUT_FOLDER,
    "fresh_efficientnetv2s_V2_final.keras"
)

PHASE1_MODEL_PATH = os.path.join(
    OUTPUT_FOLDER,
    "phase1_frozen_backbone.keras"
)

PHASE2_MODEL_PATH = os.path.join(
    OUTPUT_FOLDER,
    "phase2_last30_layers.keras"
)

PHASE3_MODEL_PATH = os.path.join(
    OUTPUT_FOLDER,
    "phase3_half_backbone.keras"
)

SPLIT_MANIFEST_PATH = os.path.join(
    OUTPUT_FOLDER,
    "V2_split_manifest.csv"
)

RESULTS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "evaluation_results.json"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Training epochs for each phase.
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 8
PHASE3_EPOCHS = 8

# Learning rates.
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
PHASE3_LR = 5e-7

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 4. REPRODUCIBILITY AND GPU SETTINGS
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

mixed_precision.set_global_policy(
    "mixed_float16"
)

print("TensorFlow version:", tf.__version__)
print(
    "Mixed precision policy:",
    mixed_precision.global_policy()
)

print(
    "Available GPUs:",
    tf.config.list_physical_devices("GPU")
)


# ============================================================
# 5. CHECK PATHS
# ============================================================

if not os.path.isdir(V2_DATASET_PATH):
    raise FileNotFoundError(
        f"V2 dataset was not found:\n{V2_DATASET_PATH}"
    )

if not os.path.isdir(EXTERNAL_TEST_ROOT):
    raise FileNotFoundError(
        "External test root was not found:\n"
        f"{EXTERNAL_TEST_ROOT}"
    )

for class_name in CLASS_NAMES:

    class_folder = os.path.join(
        V2_DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_folder):
        raise FileNotFoundError(
            f"Missing V2 class folder:\n{class_folder}"
        )

print("\nAll required folders were found.")


# ============================================================
# 6. IDENTIFY MENDELEY AND EXTERNAL IMAGES FILENAMES
# ============================================================

def identify_image_source(filename, class_name):
    """
    Identify whether an image follows the original Mendeley
    naming pattern or the added images/random naming pattern.

    Mendeley examples:
        G_6_BR.jpg
        G_6_BR_fl.jpg
        M_120_HF_rotate.jpg

    Mendeley rule:
        letters + underscore + number + underscore

    Returns
    -------
    source : str
        'mendeley' or 'kaggle'

    case_id : str
        Group ID used for leak-free splitting.
    """

    stem = Path(filename).stem

    match = re.match(
        r"^([A-Za-z]+)_(\d+)_",
        stem
    )

    if match:

        class_code = match.group(1).upper()
        case_number = match.group(2)

        # Include the folder class to avoid collisions between
        # equal numbers belonging to different classes.
        case_id = (
            f"MENDELEY_{class_name}_"
            f"{class_code}_{case_number}"
        )

        return "mendeley", case_id

    # Each Kaggle image is treated as an independent image.
    # All Kaggle images will be placed in training.
    case_id = (
        f"KAGGLE_{class_name}_{stem.lower()}"
    )

    return "kaggle", case_id


# ============================================================
# 7. SCAN THE V2 DATASET
# ============================================================

records = []

for class_name in CLASS_NAMES:

    class_folder = os.path.join(
        V2_DATASET_PATH,
        class_name
    )

    for filename in sorted(
        os.listdir(class_folder)
    ):

        full_path = os.path.join(
            class_folder,
            filename
        )

        if not os.path.isfile(full_path):
            continue

        extension = Path(filename).suffix.lower()

        if extension not in VALID_EXTENSIONS:
            continue

        source, case_id = identify_image_source(
            filename,
            class_name
        )

        records.append({
            "path": full_path,
            "filename": filename,
            "class_name": class_name,
            "label": CLASS_TO_INDEX[class_name],
            "source": source,
            "case_id": case_id
        })


dataset_df = pd.DataFrame(records)

if dataset_df.empty:
    raise ValueError(
        "No images were detected in the V2 dataset."
    )

print("\n" + "=" * 75)
print("V2 DATASET SUMMARY")
print("=" * 75)

summary_table = pd.crosstab(
    dataset_df["class_name"],
    dataset_df["source"]
)

summary_table["total"] = summary_table.sum(
    axis=1
)

print(summary_table)

print(
    "\nTotal images:",
    len(dataset_df)
)


# ============================================================
# 8. CHECK CASE IDS
# ============================================================

case_class_counts = (
    dataset_df
    .groupby("case_id")["class_name"]
    .nunique()
)

cross_class_cases = case_class_counts[
    case_class_counts > 1
]

if len(cross_class_cases) > 0:
    raise ValueError(
        "Some generated case IDs appear in more than one class."
    )

print("\nCase-to-class check: PASSED")


# ============================================================
# 9. SPLIT MENDELEY CASES BY CLASS
#
# This creates approximately:
#   70% training
#   15% validation
#   15% internal test
#
# The split is done independently for each tumor class.
# ============================================================

mendeley_df = dataset_df[
    dataset_df["source"] == "mendeley"
].copy()

kaggle_df = dataset_df[
    dataset_df["source"] == "kaggle"
].copy()

mendeley_split_frames = []

for class_name in CLASS_NAMES:

    class_df = mendeley_df[
        mendeley_df["class_name"] == class_name
    ].copy()

    unique_cases = np.array(
        sorted(class_df["case_id"].unique())
    )

    if len(unique_cases) < 3:
        raise ValueError(
            f"Not enough Mendeley cases in {class_name} "
            "to create a 70/15/15 split."
        )

    # First split:
    # 70% train, 30% temporary.
    train_cases, temporary_cases = train_test_split(
        unique_cases,
        test_size=0.30,
        random_state=SEED,
        shuffle=True
    )

    # Second split:
    # 15% validation, 15% test.
    validation_cases, test_cases = train_test_split(
        temporary_cases,
        test_size=0.50,
        random_state=SEED,
        shuffle=True
    )

    class_df["split"] = "unassigned"

    class_df.loc[
        class_df["case_id"].isin(train_cases),
        "split"
    ] = "train"

    class_df.loc[
        class_df["case_id"].isin(validation_cases),
        "split"
    ] = "validation"

    class_df.loc[
        class_df["case_id"].isin(test_cases),
        "split"
    ] = "internal_test"

    mendeley_split_frames.append(
        class_df
    )

    print(
        f"\n{class_name.upper()} Mendeley cases:"
    )

    print(
        f"  Train cases:      {len(train_cases)}"
    )

    print(
        f"  Validation cases: {len(validation_cases)}"
    )

    print(
        f"  Test cases:       {len(test_cases)}"
    )


mendeley_split_df = pd.concat(
    mendeley_split_frames,
    ignore_index=True
)


# ============================================================
# 10. ADD ALL EXTERNAL IMAGES TO TRAINING ONLY
# ============================================================

kaggle_df = kaggle_df.copy()

kaggle_df["split"] = "train"

combined_split_df = pd.concat(
    [
        mendeley_split_df,
        kaggle_df
    ],
    ignore_index=True
)


# ============================================================
# 11. VERIFY NO MENDELEY CASE LEAKAGE
# ============================================================

mendeley_check_df = combined_split_df[
    combined_split_df["source"] == "mendeley"
]

case_split_counts = (
    mendeley_check_df
    .groupby("case_id")["split"]
    .nunique()
)

leaking_cases = case_split_counts[
    case_split_counts > 1
]

if len(leaking_cases) > 0:
    raise ValueError(
        f"Leakage detected in {len(leaking_cases)} "
        "Mendeley cases."
    )

print("\nMendeley case-leakage check: PASSED")


# Check that every external image is training only.
invalid_kaggle_rows = combined_split_df[
    (combined_split_df["source"] == "kaggle")
    &
    (combined_split_df["split"] != "train")
]

if len(invalid_kaggle_rows) > 0:
    raise ValueError(
        "Some added Kaggle images were not assigned "
        "to training."
    )

print(
    "Kaggle training-only check: PASSED"
)


# ============================================================
# 12. SAVE SPLIT MANIFEST
# ============================================================

combined_split_df = combined_split_df.sort_values(
    by=[
        "split",
        "class_name",
        "source",
        "filename"
    ]
).reset_index(drop=True)

combined_split_df.to_csv(
    SPLIT_MANIFEST_PATH,
    index=False
)

print(
    "\nSplit manifest saved at:",
    SPLIT_MANIFEST_PATH
)


# ============================================================
# 13. PRINT FINAL SPLIT DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("FINAL SPLIT DISTRIBUTION")
print("=" * 75)

distribution = pd.crosstab(
    [
        combined_split_df["split"],
        combined_split_df["class_name"]
    ],
    combined_split_df["source"],
    margins=True
)

print(distribution)


train_df = combined_split_df[
    combined_split_df["split"] == "train"
].copy()

validation_df = combined_split_df[
    combined_split_df["split"] == "validation"
].copy()

internal_test_df = combined_split_df[
    combined_split_df["split"] == "internal_test"
].copy()


print("\nImage totals:")
print(
    f"  Training:      {len(train_df)}"
)
print(
    f"  Validation:    {len(validation_df)}"
)
print(
    f"  Internal test: {len(internal_test_df)}"
)

print("\nTraining-source totals:")
print(
    train_df["source"].value_counts()
)

print(
    "\nAll added Kaggle images in training:",
    len(kaggle_df)
)


# ============================================================
# 14. CREATE PATH AND LABEL ARRAYS
# ============================================================

train_paths = train_df[
    "path"
].to_numpy()

train_labels = train_df[
    "label"
].to_numpy(dtype=np.int32)

validation_paths = validation_df[
    "path"
].to_numpy()

validation_labels = validation_df[
    "label"
].to_numpy(dtype=np.int32)

internal_test_paths = internal_test_df[
    "path"
].to_numpy()

internal_test_labels = internal_test_df[
    "label"
].to_numpy(dtype=np.int32)


# ============================================================
# 15. TF.DATA IMAGE PIPELINE
# ============================================================

def load_and_prepare_image(path, label):
    """
    Read, decode and resize one image.

    EfficientNetV2S includes its own input rescaling when
    include_preprocessing=True, so images remain in 0–255 scale.
    """

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([
        None,
        None,
        3
    ])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    label = tf.one_hot(
        label,
        depth=NUM_CLASSES
    )

    return image, label


def create_dataset(
    paths,
    labels,
    training=False
):
    """
    Create a TensorFlow dataset.

    Random augmentation is inside the model, so it is applied
    only while model.fit() is training.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_prepare_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    # Validation and test images are safe to cache because
    # they are not randomly augmented.
    if not training:
        dataset = dataset.cache()

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


train_ds = create_dataset(
    train_paths,
    train_labels,
    training=True
)

validation_ds = create_dataset(
    validation_paths,
    validation_labels,
    training=False
)

internal_test_ds = create_dataset(
    internal_test_paths,
    internal_test_labels,
    training=False
)


# ============================================================
# 16. CHECK ONE BATCH
# ============================================================

for images, labels_batch in train_ds.take(1):

    print("\nTraining batch check:")
    print(
        "  Image shape:",
        images.shape
    )
    print(
        "  Label shape:",
        labels_batch.shape
    )
    print(
        "  Image dtype:",
        images.dtype
    )


# ============================================================
# 17. CALCULATE CLASS WEIGHTS
# ============================================================

train_class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

if np.any(train_class_counts == 0):
    raise ValueError(
        "At least one class has no training images."
    )

training_total = train_class_counts.sum()

class_weights = {
    class_index: (
        training_total /
        (
            NUM_CLASSES
            * class_count
        )
    )
    for class_index, class_count
    in enumerate(train_class_counts)
}

print("\nTraining class counts and weights:")

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"  {class_name:<12}: "
        f"{train_class_counts[class_index]} images | "
        f"weight = {class_weights[class_index]:.4f}"
    )


# ============================================================
# 18. DATA AUGMENTATION
#
# Applied only during training.
#
# Vertical flipping is intentionally not used because a
# vertically inverted brain image may be anatomically unusual.
# ============================================================

data_augmentation = models.Sequential(
    [
        layers.RandomFlip(
            mode="horizontal"
        ),

        layers.RandomRotation(
            factor=0.04,
            fill_mode="reflect"
        ),

        layers.RandomZoom(
            height_factor=(-0.08, 0.08),
            width_factor=(-0.08, 0.08),
            fill_mode="reflect"
        ),

        layers.RandomTranslation(
            height_factor=0.04,
            width_factor=0.04,
            fill_mode="reflect"
        ),

        layers.RandomContrast(
            factor=0.10
        )
    ],
    name="medical_image_augmentation"
)


# ============================================================
# 19. BUILD A FRESH EFFICIENTNETV2S MODEL
#
# No previous MRI model is loaded.
#
# ImageNet provides only general visual initialization.
# ============================================================

print("\n" + "=" * 75)
print("BUILDING FRESH EFFICIENTNETV2S MODEL")
print("=" * 75)

backbone = EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    include_preprocessing=True
)

backbone.trainable = False


inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    name="mri_input"
)

x = data_augmentation(
    inputs
)

x = backbone(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = layers.BatchNormalization(
    name="classification_batch_norm"
)(x)

x = layers.Dropout(
    0.35,
    name="dropout_1"
)(x)

x = layers.Dense(
    256,
    activation="swish",
    name="classification_dense"
)(x)

x = layers.Dropout(
    0.25,
    name="dropout_2"
)(x)

# float32 output is recommended when using mixed precision.
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    dtype="float32",
    name="tumor_predictions"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="Fresh_EfficientNetV2S_V2"
)

model.summary()


# ============================================================
# 20. CALLBACK FACTORY
# ============================================================

def create_callbacks(checkpoint_path):
    """
    Create new callbacks for each training phase.
    """

    return [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-8,
            verbose=1
        )
    ]


# ============================================================
# 21. PHASE 1 — TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n" + "=" * 75)
print("PHASE 1 — FROZEN EFFICIENTNETV2S BACKBONE")
print("=" * 75)

backbone.trainable = False

model.compile(
    optimizer=optimizers.Adam(
        learning_rate=PHASE1_LR
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase1 = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weights,
    callbacks=create_callbacks(
        PHASE1_MODEL_PATH
    )
)


# Load the best Phase 1 model.
model = models.load_model(
    PHASE1_MODEL_PATH
)

backbone = model.get_layer(
    "efficientnetv2-s"
)


# ============================================================
# 22. PHASE 2 — UNFREEZE LAST 30 BACKBONE LAYERS
# ============================================================

print("\n" + "=" * 75)
print("PHASE 2 — UNFREEZE LAST 30 BACKBONE LAYERS")
print("=" * 75)

backbone.trainable = True

for layer in backbone.layers:
    layer.trainable = False

for layer in backbone.layers[-30:]:
    layer.trainable = True

# Keep BatchNormalization layers frozen for stable fine-tuning.
for layer in backbone.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False


trainable_layers_phase2 = sum(
    layer.trainable
    for layer in backbone.layers
)

print(
    "Trainable backbone layers:",
    trainable_layers_phase2
)

model.compile(
    optimizer=optimizers.Adam(
        learning_rate=PHASE2_LR
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase2 = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=create_callbacks(
        PHASE2_MODEL_PATH
    )
)


# Load the best Phase 2 model.
model = models.load_model(
    PHASE2_MODEL_PATH
)

backbone = model.get_layer(
    "efficientnetv2-s"
)


# ============================================================
# 23. PHASE 3 — UNFREEZE LAST 50% OF BACKBONE
# ============================================================

print("\n" + "=" * 75)
print("PHASE 3 — DEEP FINE-TUNING LAST 50%")
print("=" * 75)

backbone.trainable = True

total_backbone_layers = len(
    backbone.layers
)

unfreeze_from = (
    total_backbone_layers // 2
)

for layer_index, layer in enumerate(
    backbone.layers
):

    layer.trainable = (
        layer_index >= unfreeze_from
    )

    # Keep BatchNormalization layers frozen.
    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False


trainable_layers_phase3 = sum(
    layer.trainable
    for layer in backbone.layers
)

print(
    f"Backbone layers: {total_backbone_layers}"
)

print(
    f"Trainable backbone layers: "
    f"{trainable_layers_phase3}"
)

model.compile(
    optimizer=optimizers.Adam(
        learning_rate=PHASE3_LR
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase3 = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=PHASE3_EPOCHS,
    class_weight=class_weights,
    callbacks=create_callbacks(
        PHASE3_MODEL_PATH
    )
)


# ============================================================
# 24. SELECT THE BEST PHASE USING VALIDATION ACCURACY
# ============================================================

phase_model_paths = {
    "Phase 1": PHASE1_MODEL_PATH,
    "Phase 2": PHASE2_MODEL_PATH,
    "Phase 3": PHASE3_MODEL_PATH
}

phase_validation_results = {}

print("\n" + "=" * 75)
print("SELECTING BEST TRAINING PHASE")
print("=" * 75)

for phase_name, model_path in phase_model_paths.items():

    candidate_model = models.load_model(
        model_path
    )

    candidate_loss, candidate_accuracy = (
        candidate_model.evaluate(
            validation_ds,
            verbose=0
        )
    )

    phase_validation_results[phase_name] = {
        "loss": float(candidate_loss),
        "accuracy": float(candidate_accuracy),
        "path": model_path
    }

    print(
        f"{phase_name:<10} | "
        f"Val loss: {candidate_loss:.4f} | "
        f"Val accuracy: {candidate_accuracy * 100:.2f}%"
    )


best_phase_name = max(
    phase_validation_results,
    key=lambda phase_name: (
        phase_validation_results[
            phase_name
        ]["accuracy"]
    )
)

best_phase_path = (
    phase_validation_results[
        best_phase_name
    ]["path"]
)

print(
    f"\nSelected best phase: "
    f"{best_phase_name}"
)

print(
    "Selected checkpoint:",
    best_phase_path
)

best_model = models.load_model(
    best_phase_path
)

best_model.save(
    FINAL_MODEL_PATH
)

print(
    "\nFinal model saved at:",
    FINAL_MODEL_PATH
)


# ============================================================
# 25. GENERAL EVALUATION FUNCTION
# ============================================================

def evaluate_dataset(
    trained_model,
    dataset,
    dataset_name,
    class_names
):
    """
    Evaluate a model and produce:
    - loss
    - accuracy
    - classification report
    - confusion matrix
    - per-class accuracy
    """

    print("\n" + "=" * 75)
    print(dataset_name.upper())
    print("=" * 75)

    loss_value, accuracy_value = (
        trained_model.evaluate(
            dataset,
            verbose=1
        )
    )

    true_labels = []
    predicted_labels = []
    prediction_probabilities = []

    for image_batch, label_batch in dataset:

        probabilities = trained_model.predict(
            image_batch,
            verbose=0
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        actual = np.argmax(
            label_batch.numpy(),
            axis=1
        )

        true_labels.extend(
            actual.tolist()
        )

        predicted_labels.extend(
            predictions.tolist()
        )

        prediction_probabilities.extend(
            probabilities.tolist()
        )

    true_labels = np.array(
        true_labels
    )

    predicted_labels = np.array(
        predicted_labels
    )

    prediction_probabilities = np.array(
        prediction_probabilities
    )

    verified_accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    print(
        f"\nLoss:     {loss_value:.4f}"
    )

    print(
        f"Accuracy: {verified_accuracy * 100:.2f}%"
    )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\nClassification report:")
    print(report_text)

    report_dict = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES))
    )

    per_class_accuracy = {}

    for class_index, class_name in enumerate(
        class_names
    ):

        class_total = matrix[
            class_index
        ].sum()

        class_correct = matrix[
            class_index,
            class_index
        ]

        if class_total > 0:
            class_accuracy = (
                class_correct /
                class_total
            )
        else:
            class_accuracy = 0.0

        per_class_accuracy[class_name] = (
            float(class_accuracy)
        )

        print(
            f"{class_name:<12}: "
            f"{class_correct}/{class_total} correct "
            f"({class_accuracy * 100:.2f}%)"
        )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={
            "fontsize": 13,
            "fontweight": "bold"
        }
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.title(
        f"{dataset_name}\n"
        f"Accuracy: "
        f"{verified_accuracy * 100:.2f}%"
    )

    plt.tight_layout()
    plt.show()

    return {
        "loss": float(loss_value),
        "accuracy": float(
            verified_accuracy
        ),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "per_class_accuracy": per_class_accuracy
    }


# ============================================================
# 26. INTERNAL MENDELEY TEST EVALUATION
# ============================================================

internal_results = evaluate_dataset(
    trained_model=best_model,
    dataset=internal_test_ds,
    dataset_name=(
        "Fresh EfficientNetV2S — "
        "Internal Mendeley Test"
    ),
    class_names=CLASS_NAMES
)


# ============================================================
# 27. LOAD UNTOUCHED EXTERNAL KAGGLE TEST
# ============================================================

external_records = []

print("\n" + "=" * 75)
print("SCANNING UNTOUCHED EXTERNAL KAGGLE TEST")
print("=" * 75)

for class_name in CLASS_NAMES:

    external_class_folder = os.path.join(
        EXTERNAL_TEST_ROOT,
        class_name,
        "test"
    )

    if not os.path.isdir(external_class_folder):
        raise FileNotFoundError(
            "Missing untouched external test folder:\n"
            f"{external_class_folder}"
        )

    number_in_class = 0

    for filename in sorted(
        os.listdir(external_class_folder)
    ):

        full_path = os.path.join(
            external_class_folder,
            filename
        )

        if not os.path.isfile(full_path):
            continue

        if (
            Path(filename).suffix.lower()
            not in VALID_EXTENSIONS
        ):
            continue

        external_records.append({
            "path": full_path,
            "label": CLASS_TO_INDEX[
                class_name
            ],
            "class_name": class_name
        })

        number_in_class += 1

    print(
        f"{class_name:<12}: "
        f"{number_in_class} images"
    )


external_df = pd.DataFrame(
    external_records
)

if external_df.empty:
    raise ValueError(
        "No untouched external images were found."
    )

external_paths = external_df[
    "path"
].to_numpy()

external_labels = external_df[
    "label"
].to_numpy(dtype=np.int32)

external_test_ds = create_dataset(
    external_paths,
    external_labels,
    training=False
)

print(
    "\nTotal untouched external test images:",
    len(external_df)
)


# ============================================================
# 28. EXTERNAL KAGGLE TEST EVALUATION
# ============================================================

external_results = evaluate_dataset(
    trained_model=best_model,
    dataset=external_test_ds,
    dataset_name=(
        "Fresh EfficientNetV2S — "
        "Untouched External Kaggle Test"
    ),
    class_names=CLASS_NAMES
)


# ============================================================
# 29. COMBINE TRAINING HISTORIES
# ============================================================

def combine_history_values(
    histories,
    metric_name
):
    """
    Join one metric across multiple training phases.
    """

    values = []

    for history_object in histories:

        values.extend(
            history_object.history.get(
                metric_name,
                []
            )
        )

    return values


all_histories = [
    history_phase1,
    history_phase2,
    history_phase3
]

combined_training_accuracy = (
    combine_history_values(
        all_histories,
        "accuracy"
    )
)

combined_validation_accuracy = (
    combine_history_values(
        all_histories,
        "val_accuracy"
    )
)

combined_training_loss = (
    combine_history_values(
        all_histories,
        "loss"
    )
)

combined_validation_loss = (
    combine_history_values(
        all_histories,
        "val_loss"
    )
)


# ============================================================
# 30. TRAINING CURVES
# ============================================================

phase1_length = len(
    history_phase1.history["loss"]
)

phase2_length = len(
    history_phase2.history["loss"]
)

phase2_start = phase1_length
phase3_start = (
    phase1_length
    + phase2_length
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    combined_training_accuracy,
    label="Training accuracy"
)

plt.plot(
    combined_validation_accuracy,
    label="Validation accuracy"
)

plt.axvline(
    phase2_start - 0.5,
    linestyle="--",
    label="Start Phase 2"
)

plt.axvline(
    phase3_start - 0.5,
    linestyle="--",
    label="Start Phase 3"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    "Fresh EfficientNetV2S — Accuracy"
)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    combined_training_loss,
    label="Training loss"
)

plt.plot(
    combined_validation_loss,
    label="Validation loss"
)

plt.axvline(
    phase2_start - 0.5,
    linestyle="--",
    label="Start Phase 2"
)

plt.axvline(
    phase3_start - 0.5,
    linestyle="--",
    label="Start Phase 3"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Fresh EfficientNetV2S — Loss"
)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 31. SAVE RESULTS
# ============================================================

all_results = {
    "dataset": {
        "total_v2_images": int(
            len(dataset_df)
        ),
        "mendeley_images": int(
            len(mendeley_df)
        ),
        "added_kaggle_training_images": int(
            len(kaggle_df)
        ),
        "training_images": int(
            len(train_df)
        ),
        "validation_images": int(
            len(validation_df)
        ),
        "internal_test_images": int(
            len(internal_test_df)
        ),
        "external_test_images": int(
            len(external_df)
        )
    },

    "selected_phase": best_phase_name,

    "phase_validation_results":
        phase_validation_results,

    "internal_mendeley_test":
        internal_results,

    "external_kaggle_test":
        external_results
}

with open(
    RESULTS_PATH,
    "w"
) as results_file:

    json.dump(
        all_results,
        results_file,
        indent=4
    )


# ============================================================
# 32. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FRESH EFFICIENTNETV2S EXPERIMENT COMPLETED")
print("=" * 75)

print("\nTraining design:")
print(
    "  Mendeley training images: "
    f"{len(train_df[train_df['source'] == 'mendeley'])}"
)

print(
    "  Added Kaggle training images: "
    f"{len(train_df[train_df['source'] == 'kaggle'])}"
)

print(
    "  Mendeley validation images: "
    f"{len(validation_df)}"
)

print(
    "  Mendeley internal test images: "
    f"{len(internal_test_df)}"
)

print(
    "  Untouched external test images: "
    f"{len(external_df)}"
)

print(
    f"\nSelected best phase: "
    f"{best_phase_name}"
)

print(
    "\nInternal Mendeley test accuracy: "
    f"{internal_results['accuracy'] * 100:.2f}%"
)

print(
    "External images test accuracy: "
    f"{external_results['accuracy'] * 100:.2f}%"
)

print("\nFinal model:")
print(FINAL_MODEL_PATH)

print("\nSplit manifest:")
print(SPLIT_MANIFEST_PATH)

print("\nResults file:")
print(RESULTS_PATH)

print(
    "\n✅ No previously trained MRI model was loaded."
)

print(
    "✅ All 650 added External images were assigned "
    "to training only."
)

print(
    "✅ Mendeley validation and internal test images "
    "were not used for learning."
)

print(
    "✅ Untouched External test images were used only "
    "for the final external evaluation."
)
