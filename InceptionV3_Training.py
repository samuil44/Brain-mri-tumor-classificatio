# ============================================================
# ONE-CELL INCEPTIONV3 PHASE 1 EVALUATION
#
# This cell:
#   1. Loads the best saved Phase 1 InceptionV3 model
#   2. Loads the exact internal test split from the manifest
#   3. Loads the untouched external images test
#   4. Evaluates both datasets
#   5. Shows classification reports and confusion matrices
#   6. Saves probabilities for the future ensemble
# ============================================================

import os
import json
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


# ============================================================
# 1. PATHS
# ============================================================

PHASE1_MODEL_PATH = (
    "/content/drive/MyDrive/Final Project/"
    "Fresh_InceptionV3_V2/"
    "phase1_frozen_backbone.keras"
)

MANIFEST_PATH = (
    "/content/drive/MyDrive/Final Project/"
    "Fresh_EfficientNetV2S_V2/"
    "V2_split_manifest.csv"
)

EXTERNAL_TEST_ROOT = (
    "/content/drive/MyDrive/Final Project/"
    "external_MRI_test/retraining"
)

OUTPUT_FOLDER = (
    "/content/drive/MyDrive/Final Project/"
    "Fresh_InceptionV3_V2"
)

FINAL_PHASE1_MODEL_PATH = os.path.join(
    OUTPUT_FOLDER,
    "fresh_inceptionv3_V2_phase1_final.keras"
)

INTERNAL_PREDICTIONS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "inceptionv3_phase1_internal_predictions.csv"
)

EXTERNAL_PREDICTIONS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "inceptionv3_phase1_external_predictions.csv"
)

RESULTS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "inceptionv3_phase1_evaluation_results.json"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
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
AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not os.path.isfile(PHASE1_MODEL_PATH):
    raise FileNotFoundError(
        "Phase 1 model was not found:\n"
        f"{PHASE1_MODEL_PATH}"
    )

if not os.path.isfile(MANIFEST_PATH):
    raise FileNotFoundError(
        "Split manifest was not found:\n"
        f"{MANIFEST_PATH}"
    )

if not os.path.isdir(EXTERNAL_TEST_ROOT):
    raise FileNotFoundError(
        "External test root was not found:\n"
        f"{EXTERNAL_TEST_ROOT}"
    )

print("✅ Required paths found.")


# ============================================================
# 4. LOAD THE BEST PHASE 1 MODEL
# ============================================================

print("\nLoading InceptionV3 Phase 1 model...")

model = tf.keras.models.load_model(
    PHASE1_MODEL_PATH
)

# Save a clearly named copy for the ensemble.
model.save(
    FINAL_PHASE1_MODEL_PATH
)

print("✅ Phase 1 model loaded successfully.")
print("Model input shape:", model.input_shape)
print("Saved ensemble-ready copy:")
print(FINAL_PHASE1_MODEL_PATH)


# ============================================================
# 5. LOAD INTERNAL TEST FROM EXISTING MANIFEST
# ============================================================

manifest_df = pd.read_csv(
    MANIFEST_PATH
)

required_columns = {
    "path",
    "class_name",
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

internal_df = manifest_df[
    manifest_df["split"] == "internal_test"
].copy()

if internal_df.empty:
    raise ValueError(
        "No internal-test rows were found in the manifest."
    )

if internal_df["label"].isna().any():
    raise ValueError(
        "Unexpected class name found in internal test."
    )

internal_df["label"] = (
    internal_df["label"]
    .astype(np.int32)
)

internal_df["file_exists"] = (
    internal_df["path"]
    .astype(str)
    .apply(os.path.isfile)
)

missing_internal = internal_df[
    internal_df["file_exists"] == False
]

if len(missing_internal) > 0:
    print(
        missing_internal[
            ["path", "class_name"]
        ].head(10)
    )

    raise FileNotFoundError(
        f"{len(missing_internal)} internal-test images "
        "were not found."
    )

internal_df.drop(
    columns=["file_exists"],
    inplace=True
)

internal_df = internal_df.reset_index(
    drop=True
)

print(
    "\nInternal test images:",
    len(internal_df)
)

print(
    internal_df["class_name"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. SCAN UNTOUCHED EXTERNAL TEST
# ============================================================

external_records = []

print("\nUntouched external test distribution:")

for class_name in CLASS_NAMES:

    class_test_folder = os.path.join(
        EXTERNAL_TEST_ROOT,
        class_name,
        "test"
    )

    if not os.path.isdir(class_test_folder):
        raise FileNotFoundError(
            "External test folder was not found:\n"
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

        extension = (
            Path(filename)
            .suffix
            .lower()
        )

        if extension not in VALID_EXTENSIONS:
            continue

        external_records.append({
            "path": image_path,
            "filename": filename,
            "class_name": class_name,
            "label": CLASS_TO_INDEX[class_name]
        })

        class_count += 1

    print(
        f"  {class_name:<12}: {class_count}"
    )

external_df = pd.DataFrame(
    external_records
)

if external_df.empty:
    raise ValueError(
        "No untouched external test images were found."
    )

external_df = external_df.reset_index(
    drop=True
)

print(
    "Total external test images:",
    len(external_df)
)


# ============================================================
# 7. IMAGE-LOADING PIPELINE
#
# The model itself already contains the InceptionV3
# preprocessing layer, so images remain in the 0–255 range.
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

    label = tf.cast(
        label,
        tf.int32
    )

    return image, label


def create_test_dataset(
    dataframe
):

    paths = (
        dataframe["path"]
        .astype(str)
        .to_numpy()
    )

    labels = (
        dataframe["label"]
        .to_numpy(dtype=np.int32)
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    dataset = dataset.map(
        load_and_prepare_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


internal_test_ds = create_test_dataset(
    internal_df
)

external_test_ds = create_test_dataset(
    external_df
)


# ============================================================
# 8. CHECK ONE BATCH
# ============================================================

for images, labels in internal_test_ds.take(1):

    print("\nDataset check:")
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print(
        "Pixel range:",
        float(tf.reduce_min(images)),
        "to",
        float(tf.reduce_max(images))
    )


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_dataset(
    trained_model,
    dataset,
    dataframe,
    dataset_name
):

    print("\n" + "=" * 75)
    print(dataset_name.upper())
    print("=" * 75)

    probability_batches = []
    true_label_batches = []

    for image_batch, label_batch in dataset:

        batch_probabilities = trained_model.predict(
            image_batch,
            verbose=0
        )

        probability_batches.append(
            batch_probabilities
        )

        true_label_batches.append(
            label_batch.numpy()
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

    print("Per-class accuracy:")

    per_class_accuracy = {}

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

        per_class_accuracy[class_name] = float(
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

    predictions_df = dataframe.copy()

    predictions_df[
        "true_class"
    ] = [
        CLASS_NAMES[index]
        for index in true_labels
    ]

    predictions_df[
        "predicted_class"
    ] = [
        CLASS_NAMES[index]
        for index in predicted_labels
    ]

    predictions_df[
        "confidence"
    ] = confidences

    predictions_df[
        "correct"
    ] = (
        true_labels
        ==
        predicted_labels
    )

    # Save all four probabilities.
    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        predictions_df[
            f"prob_{class_name}"
        ] = probabilities[
            :,
            class_index
        ]

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
        "per_class_accuracy": per_class_accuracy,
        "predictions_dataframe": predictions_df
    }


# ============================================================
# 10. INTERNAL TEST EVALUATION
# ============================================================

internal_results = evaluate_dataset(
    trained_model=model,
    dataset=internal_test_ds,
    dataframe=internal_df,
    dataset_name=(
        "InceptionV3 Phase 1 — "
        "Internal Mendeley Test"
    )
)


# ============================================================
# 11. EXTERNAL TEST EVALUATION
# ============================================================

external_results = evaluate_dataset(
    trained_model=model,
    dataset=external_test_ds,
    dataframe=external_df,
    dataset_name=(
        "InceptionV3 Phase 1 — "
        "Untouched External Kaggle Test"
    )
)


# ============================================================
# 12. SAVE PREDICTIONS
#
# These probability columns will later be used for:
# EfficientNetV2S + InceptionV3 soft voting.
# ============================================================

internal_results[
    "predictions_dataframe"
].to_csv(
    INTERNAL_PREDICTIONS_PATH,
    index=False
)

external_results[
    "predictions_dataframe"
].to_csv(
    EXTERNAL_PREDICTIONS_PATH,
    index=False
)


# ============================================================
# 13. SAVE RESULTS AS JSON
# ============================================================

results_to_save = {

    "model": {
        "architecture": "InceptionV3",
        "training_phase": "Phase 1 frozen backbone",
        "model_path": FINAL_PHASE1_MODEL_PATH,
        "image_size": list(IMG_SIZE)
    },

    "internal_test": {
        "number_of_images": int(
            len(internal_df)
        ),
        "accuracy": internal_results[
            "accuracy"
        ],
        "confusion_matrix": internal_results[
            "confusion_matrix"
        ],
        "classification_report": internal_results[
            "classification_report"
        ],
        "per_class_accuracy": internal_results[
            "per_class_accuracy"
        ]
    },

    "external_test": {
        "number_of_images": int(
            len(external_df)
        ),
        "accuracy": external_results[
            "accuracy"
        ],
        "confusion_matrix": external_results[
            "confusion_matrix"
        ],
        "classification_report": external_results[
            "classification_report"
        ],
        "per_class_accuracy": external_results[
            "per_class_accuracy"
        ]
    }
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
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("INCEPTIONV3 PHASE 1 EVALUATION COMPLETED")
print("=" * 75)

print(
    "\nInternal Mendeley accuracy: "
    f"{internal_results['accuracy'] * 100:.2f}%"
)

print(
    "Untouched external accuracy: "
    f"{external_results['accuracy'] * 100:.2f}%"
)

print("\nModel saved for ensemble:")
print(FINAL_PHASE1_MODEL_PATH)

print("\nInternal probabilities saved:")
print(INTERNAL_PREDICTIONS_PATH)

print("\nExternal probabilities saved:")
print(EXTERNAL_PREDICTIONS_PATH)

print("\nEvaluation results saved:")
print(RESULTS_PATH)

print(
    "\n✅ No additional training was performed."
)

print(
    "✅ The exact internal test split was preserved."
)

print(
    "✅ The untouched external test was evaluated separately."
)

print(
    "✅ Four-class probabilities were saved for "
    "the future ensemble."
)
