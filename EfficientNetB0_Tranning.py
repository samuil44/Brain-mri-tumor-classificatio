# ============================================================
# FRESH EFFICIENTNETB0 V2 — FOUR-CLASS MRI CLASSIFICATION
#
# Uses the exact existing V2 split:
#
# TRAIN:
#   - Mendeley training split
#   - All added extrnal training images
#
# VALIDATION:
#   - Mendeley validation only
#
# INTERNAL TEST:
#   - Mendeley internal test only
#
# EXTERNAL TEST:
#   - Untouched External test folders only
#
# Important:
#   - No previous MRI model is loaded.
#   - Starts fresh from ImageNet weights.
#   - Uses the same manifest as EfficientNetV2S.
#   - Saves every phase separately.
# ============================================================


# ============================================================
# 1. SAFE GOOGLE DRIVE MOUNT
# ============================================================

import os

if not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
else:
    print("✅ Google Drive is already mounted.")


# ============================================================
# 2. IMPORTS
# ============================================================

import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

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

from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# 3. PATHS
# ============================================================

FINAL_PROJECT = (
    "/content/drive/MyDrive/Final Project"
)

MANIFEST_PATH = os.path.join(
    FINAL_PROJECT,
    "Fresh_EfficientNetV2S_V2",
    "V2_split_manifest.csv"
)

EXTERNAL_TEST_ROOT = os.path.join(
    FINAL_PROJECT,
    "external_MRI_test",
    "retraining"
)

OUTPUT_FOLDER = os.path.join(
    FINAL_PROJECT,
    "Fresh_EfficientNetB0_V2"
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
    "phase3_last60_layers.keras"
)

FINAL_MODEL_PATH = os.path.join(
    OUTPUT_FOLDER,
    "fresh_efficientnetb0_V2_final.keras"
)

RESULTS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "efficientnetb0_V2_results.json"
)

INTERNAL_PREDICTIONS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "efficientnetb0_internal_predictions.csv"
)

EXTERNAL_PREDICTIONS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "efficientnetb0_external_predictions.csv"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 4. SETTINGS
# ============================================================

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
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Faster and safer than very long training.
PHASE1_EPOCHS = 8
PHASE2_EPOCHS = 6
PHASE3_EPOCHS = 6

PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
PHASE3_LR = 2e-6


# ============================================================
# 5. REPRODUCIBILITY AND GPU
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.backend.clear_session()

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    mixed_precision.set_global_policy(
        "mixed_float16"
    )
else:
    mixed_precision.set_global_policy(
        "float32"
    )

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", gpus)
print(
    "Mixed precision:",
    mixed_precision.global_policy()
)


# ============================================================
# 6. CHECK REQUIRED PATHS
# ============================================================

if not os.path.isfile(MANIFEST_PATH):
    raise FileNotFoundError(
        "Split manifest was not found:\n"
        f"{MANIFEST_PATH}"
    )

if not os.path.isdir(EXTERNAL_TEST_ROOT):
    raise FileNotFoundError(
        "External test folder was not found:\n"
        f"{EXTERNAL_TEST_ROOT}"
    )

print("\n✅ Required paths found.")


# ============================================================
# 7. LOAD THE EXISTING V2 SPLIT MANIFEST
# ============================================================

manifest_df = pd.read_csv(
    MANIFEST_PATH
)

required_columns = {
    "path",
    "class_name",
    "source",
    "case_id",
    "split"
}

missing_columns = (
    required_columns
    - set(manifest_df.columns)
)

if missing_columns:
    raise ValueError(
        "Manifest is missing columns:\n"
        f"{sorted(missing_columns)}"
    )

manifest_df["class_name"] = (
    manifest_df["class_name"]
    .astype(str)
    .str.strip()
    .str.lower()
)

manifest_df["source"] = (
    manifest_df["source"]
    .astype(str)
    .str.strip()
    .str.lower()
)

manifest_df["split"] = (
    manifest_df["split"]
    .astype(str)
    .str.strip()
    .str.lower()
)

manifest_df["label"] = (
    manifest_df["class_name"]
    .map(CLASS_TO_INDEX)
)

if manifest_df["label"].isna().any():
    raise ValueError(
        "Unexpected class name found in manifest."
    )

manifest_df["label"] = (
    manifest_df["label"]
    .astype(np.int32)
)

print(
    "\nManifest rows:",
    len(manifest_df)
)


# ============================================================
# 8. VERIFY IMAGE PATHS
# ============================================================

manifest_df["file_exists"] = (
    manifest_df["path"]
    .astype(str)
    .apply(os.path.isfile)
)

missing_files = manifest_df[
    manifest_df["file_exists"] == False
]

if len(missing_files) > 0:
    print(
        missing_files[
            ["path", "class_name", "split"]
        ].head(10)
    )

    raise FileNotFoundError(
        f"{len(missing_files)} manifest images "
        "were not found."
    )

manifest_df.drop(
    columns=["file_exists"],
    inplace=True
)

print("✅ All manifest image paths exist.")


# ============================================================
# 9. VERIFY NO MENDELEY CASE LEAKAGE
# ============================================================

mendeley_df = manifest_df[
    manifest_df["source"] == "mendeley"
]

case_split_counts = (
    mendeley_df
    .groupby("case_id")["split"]
    .nunique()
)

leaking_cases = case_split_counts[
    case_split_counts > 1
]

if len(leaking_cases) > 0:
    raise ValueError(
        f"Leakage detected in "
        f"{len(leaking_cases)} Mendeley cases."
    )

print("✅ Mendeley case-leakage check passed.")


# ============================================================
# 10. VERIFY EXTERNAL IMAGES ARE TRAINING-ONLY
# ============================================================

invalid_kaggle_rows = manifest_df[
    (manifest_df["source"] == "kaggle")
    &
    (manifest_df["split"] != "train")
]

if len(invalid_kaggle_rows) > 0:
    raise ValueError(
        "Some added Kaggle images are outside training."
    )

print("✅ Added Kaggle images are training-only.")


# ============================================================
# 11. CREATE SPLIT TABLES
# ============================================================

train_df = manifest_df[
    manifest_df["split"] == "train"
].copy()

validation_df = manifest_df[
    manifest_df["split"] == "validation"
].copy()

internal_test_df = manifest_df[
    manifest_df["split"] == "internal_test"
].copy()

if train_df.empty:
    raise ValueError("Training split is empty.")

if validation_df.empty:
    raise ValueError("Validation split is empty.")

if internal_test_df.empty:
    raise ValueError("Internal test split is empty.")

print("\n" + "=" * 75)
print("EFFICIENTNETB0 V2 DATA DISTRIBUTION")
print("=" * 75)

distribution = pd.crosstab(
    [
        manifest_df["split"],
        manifest_df["class_name"]
    ],
    manifest_df["source"],
    margins=True
)

print(distribution)

print("\nImage totals:")
print("Training:     ", len(train_df))
print("Validation:   ", len(validation_df))
print("Internal test:", len(internal_test_df))

print("\nTraining source totals:")
print(
    train_df["source"]
    .value_counts()
)


# ============================================================
# 12. PATH AND LABEL ARRAYS
# ============================================================

train_paths = (
    train_df["path"]
    .astype(str)
    .to_numpy()
)

train_labels = (
    train_df["label"]
    .to_numpy(dtype=np.int32)
)

validation_paths = (
    validation_df["path"]
    .astype(str)
    .to_numpy()
)

validation_labels = (
    validation_df["label"]
    .to_numpy(dtype=np.int32)
)

internal_test_paths = (
    internal_test_df["path"]
    .astype(str)
    .to_numpy()
)

internal_test_labels = (
    internal_test_df["label"]
    .to_numpy(dtype=np.int32)
)


# ============================================================
# 13. IMAGE PIPELINE
#
# EfficientNetB0 includes its preprocessing internally.
# Keep image values in the 0–255 range.
# ============================================================

def load_and_prepare_image(
    path,
    label
):

    image_bytes = tf.io.read_file(
        path
    )

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
# 14. CHECK ONE BATCH
# ============================================================

for images, labels_batch in train_ds.take(1):

    print("\nTraining batch check:")
    print("Image shape:", images.shape)
    print("Label shape:", labels_batch.shape)
    print("Image dtype:", images.dtype)

    print(
        "Pixel range:",
        float(tf.reduce_min(images)),
        "to",
        float(tf.reduce_max(images))
    )


# ============================================================
# 15. CLASS WEIGHTS
# ============================================================

train_class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

if np.any(train_class_counts == 0):
    raise ValueError(
        "At least one class has zero training images."
    )

training_total = train_class_counts.sum()

class_weights = {
    class_index: (
        training_total
        /
        (
            NUM_CLASSES
            * class_count
        )
    )
    for class_index, class_count
    in enumerate(train_class_counts)
}

print("\nTraining class weights:")

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_name:<12}: "
        f"{train_class_counts[class_index]} images | "
        f"weight = {class_weights[class_index]:.4f}"
    )


# ============================================================
# 16. DATA AUGMENTATION
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
# 17. BUILD FRESH EFFICIENTNETB0
# ============================================================

print("\n" + "=" * 75)
print("BUILDING FRESH EFFICIENTNETB0 V2")
print("=" * 75)

backbone = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
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

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    dtype="float32",
    name="tumor_predictions"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="Fresh_EfficientNetB0_V2"
)

model.summary()


# ============================================================
# 18. CALLBACKS
# ============================================================

def create_callbacks(
    checkpoint_path
):

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
            patience=3,
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
# 19. PHASE 1 — FROZEN BACKBONE
# ============================================================

print("\n" + "=" * 75)
print("PHASE 1 — FROZEN EFFICIENTNETB0 BACKBONE")
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
    ),
    verbose=1
)


# ============================================================
# 20. LOAD BEST PHASE 1
# ============================================================

model = models.load_model(
    PHASE1_MODEL_PATH
)

backbone = model.get_layer(
    "efficientnetb0"
)

print("✅ Best Phase 1 model loaded.")


# ============================================================
# 21. PHASE 2 — UNFREEZE LAST 30 LAYERS
# ============================================================

print("\n" + "=" * 75)
print("PHASE 2 — UNFREEZE LAST 30 BACKBONE LAYERS")
print("=" * 75)

backbone.trainable = True

for layer in backbone.layers:
    layer.trainable = False

for layer in backbone.layers[-30:]:
    layer.trainable = True

for layer in backbone.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False

trainable_phase2 = sum(
    int(layer.trainable)
    for layer in backbone.layers
)

print(
    "Trainable backbone layers:",
    trainable_phase2
)

model.compile(
    optimizer=optimizers.Adam(
        learning_rate=PHASE2_LR,
        clipnorm=1.0
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
    ),
    verbose=1
)


# ============================================================
# 22. LOAD BEST PHASE 2
# ============================================================

model = models.load_model(
    PHASE2_MODEL_PATH
)

backbone = model.get_layer(
    "efficientnetb0"
)

print("✅ Best Phase 2 model loaded.")


# ============================================================
# 23. PHASE 3 — UNFREEZE LAST 60 LAYERS
# ============================================================

print("\n" + "=" * 75)
print("PHASE 3 — UNFREEZE LAST 60 BACKBONE LAYERS")
print("=" * 75)

backbone.trainable = True

for layer in backbone.layers:
    layer.trainable = False

for layer in backbone.layers[-60:]:
    layer.trainable = True

for layer in backbone.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False

trainable_phase3 = sum(
    int(layer.trainable)
    for layer in backbone.layers
)

print(
    "Trainable backbone layers:",
    trainable_phase3
)

model.compile(
    optimizer=optimizers.Adam(
        learning_rate=PHASE3_LR,
        clipnorm=1.0
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
    ),
    verbose=1
)


# ============================================================
# 24. SELECT BEST PHASE USING VALIDATION ONLY
# ============================================================

phase_paths = {
    "Phase 1": PHASE1_MODEL_PATH,
    "Phase 2": PHASE2_MODEL_PATH,
    "Phase 3": PHASE3_MODEL_PATH
}

phase_validation_results = {}

print("\n" + "=" * 75)
print("SELECTING BEST EFFICIENTNETB0 PHASE")
print("=" * 75)

for phase_name, phase_path in phase_paths.items():

    candidate_model = models.load_model(
        phase_path
    )

    loss_value, accuracy_value = (
        candidate_model.evaluate(
            validation_ds,
            verbose=0
        )
    )

    phase_validation_results[
        phase_name
    ] = {
        "loss": float(loss_value),
        "accuracy": float(accuracy_value),
        "path": phase_path
    }

    print(
        f"{phase_name:<10} | "
        f"Val loss: {loss_value:.4f} | "
        f"Val accuracy: {accuracy_value * 100:.2f}%"
    )


best_phase_name = sorted(
    phase_validation_results.keys(),
    key=lambda phase_name: (
        -phase_validation_results[
            phase_name
        ]["accuracy"],
        phase_validation_results[
            phase_name
        ]["loss"]
    )
)[0]

best_phase_path = (
    phase_validation_results[
        best_phase_name
    ]["path"]
)

best_model = models.load_model(
    best_phase_path
)

best_model.save(
    FINAL_MODEL_PATH
)

print(
    "\nSelected best phase:",
    best_phase_name
)

print(
    "Final model saved at:",
    FINAL_MODEL_PATH
)


# ============================================================
# 25. EVALUATION FUNCTION
# ============================================================

def evaluate_dataset(
    trained_model,
    dataset,
    dataframe,
    dataset_name,
    output_csv_path
):

    print("\n" + "=" * 75)
    print(dataset_name.upper())
    print("=" * 75)

    probability_batches = []
    true_label_batches = []

    start_time = time.time()

    for image_batch, label_batch in dataset:

        batch_probabilities = (
            trained_model.predict(
                image_batch,
                verbose=0
            )
        )

        probability_batches.append(
            batch_probabilities
        )

        true_label_batches.append(
            np.argmax(
                label_batch.numpy(),
                axis=1
            )
        )

    elapsed_time = (
        time.time()
        - start_time
    )

    probabilities = np.concatenate(
        probability_batches,
        axis=0
    )

    true_labels = np.concatenate(
        true_label_batches,
        axis=0
    ).astype(np.int32)

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    confidences = np.max(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES))
    )

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Inference time: {elapsed_time:.2f} seconds"
    )

    print("\nClassification report:")

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print(report_text)

    report_dict = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    per_class_accuracy = {}

    print("Per-class accuracy:")

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_total = matrix[
            class_index
        ].sum()

        class_correct = matrix[
            class_index,
            class_index
        ]

        class_accuracy = (
            class_correct / class_total
            if class_total > 0
            else 0.0
        )

        per_class_accuracy[
            class_name
        ] = float(
            class_accuracy
        )

        print(
            f"  {class_name:<12}: "
            f"{class_correct}/{class_total} "
            f"({class_accuracy * 100:.2f}%)"
        )

    print("\nConfusion matrix:")
    print(matrix)

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={
            "fontsize": 13,
            "fontweight": "bold"
        }
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.title(
        f"{dataset_name}\n"
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    plt.tight_layout()
    plt.show()

    output_df = dataframe.copy()

    output_df[
        "true_class"
    ] = [
        CLASS_NAMES[index]
        for index in true_labels
    ]

    output_df[
        "predicted_class"
    ] = [
        CLASS_NAMES[index]
        for index in predicted_labels
    ]

    output_df[
        "confidence"
    ] = confidences

    output_df[
        "correct"
    ] = (
        true_labels
        ==
        predicted_labels
    )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        output_df[
            f"prob_{class_name}"
        ] = probabilities[
            :,
            class_index
        ]

    output_df.to_csv(
        output_csv_path,
        index=False
    )

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
        "per_class_accuracy": per_class_accuracy,
        "probabilities": probabilities,
        "predictions": predicted_labels
    }


# ============================================================
# 26. INTERNAL TEST EVALUATION
# ============================================================

internal_results = evaluate_dataset(
    trained_model=best_model,
    dataset=internal_test_ds,
    dataframe=internal_test_df.reset_index(
        drop=True
    ),
    dataset_name=(
        "Fresh EfficientNetB0 V2 — "
        "Internal Mendeley Test"
    ),
    output_csv_path=(
        INTERNAL_PREDICTIONS_PATH
    )
)


# ============================================================
# 27. SCAN UNTOUCHED EXTERNAL TEST
# ============================================================

external_records = []

print("\n" + "=" * 75)
print("SCANNING UNTOUCHED EXTERNAL TEST")
print("=" * 75)

for class_name in CLASS_NAMES:

    class_test_folder = os.path.join(
        EXTERNAL_TEST_ROOT,
        class_name,
        "test"
    )

    if not os.path.isdir(class_test_folder):
        raise FileNotFoundError(
            "Missing external test folder:\n"
            f"{class_test_folder}"
        )

    class_count = 0

    for filename in sorted(
        os.listdir(class_test_folder)
    ):

        image_path = os.path.join(
            class_test_folder,
            filename
        )

        if not os.path.isfile(image_path):
            continue

        if (
            Path(filename)
            .suffix
            .lower()
            not in VALID_EXTENSIONS
        ):
            continue

        external_records.append({
            "path": image_path,
            "filename": filename,
            "class_name": class_name,
            "label": CLASS_TO_INDEX[
                class_name
            ]
        })

        class_count += 1

    print(
        f"{class_name:<12}: "
        f"{class_count}"
    )

external_df = pd.DataFrame(
    external_records
)

if external_df.empty:
    raise ValueError(
        "No external test images were found."
    )

external_paths = (
    external_df["path"]
    .astype(str)
    .to_numpy()
)

external_labels = (
    external_df["label"]
    .to_numpy(dtype=np.int32)
)

external_test_ds = create_dataset(
    external_paths,
    external_labels,
    training=False
)

print(
    "\nTotal external images:",
    len(external_df)
)


# ============================================================
# 28. EXTERNAL TEST EVALUATION
# ============================================================

external_results = evaluate_dataset(
    trained_model=best_model,
    dataset=external_test_ds,
    dataframe=external_df.reset_index(
        drop=True
    ),
    dataset_name=(
        "Fresh EfficientNetB0 V2 — "
        "Untouched External Kaggle Test"
    ),
    output_csv_path=(
        EXTERNAL_PREDICTIONS_PATH
    )
)


# ============================================================
# 29. TRAINING CURVES
# ============================================================

all_histories = [
    history_phase1,
    history_phase2,
    history_phase3
]


def combine_history(
    histories,
    metric_name
):

    values = []

    for history in histories:
        values.extend(
            history.history.get(
                metric_name,
                []
            )
        )

    return values


training_accuracy = combine_history(
    all_histories,
    "accuracy"
)

validation_accuracy = combine_history(
    all_histories,
    "val_accuracy"
)

training_loss = combine_history(
    all_histories,
    "loss"
)

validation_loss = combine_history(
    all_histories,
    "val_loss"
)

phase1_length = len(
    history_phase1.history["loss"]
)

phase2_length = len(
    history_phase2.history["loss"]
)

phase2_start = phase1_length

phase3_start = (
    phase1_length
    +
    phase2_length
)


plt.figure(figsize=(10, 5))

plt.plot(
    training_accuracy,
    label="Training accuracy"
)

plt.plot(
    validation_accuracy,
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
plt.title("Fresh EfficientNetB0 V2 — Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    training_loss,
    label="Training loss"
)

plt.plot(
    validation_loss,
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
plt.title("Fresh EfficientNetB0 V2 — Loss")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 30. SAVE RESULTS
# ============================================================

def json_safe_result(
    result
):

    return {
        "accuracy": result[
            "accuracy"
        ],
        "confusion_matrix": result[
            "confusion_matrix"
        ],
        "classification_report": result[
            "classification_report"
        ],
        "per_class_accuracy": result[
            "per_class_accuracy"
        ]
    }


results_to_save = {

    "selected_phase":
        best_phase_name,

    "phase_validation_results":
        phase_validation_results,

    "dataset": {
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
        ),
        "mendeley_training_images": int(
            len(
                train_df[
                    train_df["source"]
                    ==
                    "mendeley"
                ]
            )
        ),
        "kaggle_training_images": int(
            len(
                train_df[
                    train_df["source"]
                    ==
                    "kaggle"
                ]
            )
        )
    },

    "internal_test":
        json_safe_result(
            internal_results
        ),

    "external_test":
        json_safe_result(
            external_results
        )
}

with open(
    RESULTS_PATH,
    "w"
) as results_file:

    json.dump(
        results_to_save,
        results_file,
        indent=4
    )


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FRESH EFFICIENTNETB0 V2 TRAINING COMPLETED")
print("=" * 75)

print(
    "\nSelected phase:",
    best_phase_name
)

print(
    "\nInternal accuracy:",
    f"{internal_results['accuracy'] * 100:.2f}%"
)

print(
    "External accuracy:",
    f"{external_results['accuracy'] * 100:.2f}%"
)

print("\nFinal model:")
print(FINAL_MODEL_PATH)

print("\nInternal predictions:")
print(INTERNAL_PREDICTIONS_PATH)

print("\nExternal predictions:")
print(EXTERNAL_PREDICTIONS_PATH)

print("\nResults:")
print(RESULTS_PATH)

print(
    "\n✅ Exact V2 split manifest was reused."
)

print(
    "✅ Added Kaggle images remained training-only."
)

print(
    "✅ Untouched external test was used only "
    "after model selection."
)

print(
    "✅ No previous MRI model was loaded."
)
