# ============================================================
# FINAL ONE-CELL SYSTEM
#
# 1. Upload one MRI image
# 2. Predict using the final fine-weight 3-model ensemble
# 3. If prediction is "notumor":
#       - Skip grayscale suspicious-region analysis
#       - Display no boxes
# 4. If a tumor class is predicted:
#       - Run improved grayscale analysis
#       - Display up to 3 suspicious-region candidates
#
# IMPORTANT:
# Candidate boxes support physician review.
# They are not confirmed tumor boundaries.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import gc
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from google.colab import files


# ============================================================
# 2. MOUNT GOOGLE DRIVE
# ============================================================

if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
else:
    print("✅ Google Drive is already mounted.")


# ============================================================
# 3. FINAL PROJECT SETTINGS
# ============================================================

FINAL_PROJECT = "/content/drive/MyDrive/Final Project"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

IMAGE_SIZE = (224, 224)


MODEL_PATHS = {
    "EfficientNetV2S": os.path.join(
        FINAL_PROJECT,
        "Fresh_EfficientNetV2S_V2",
        "fresh_efficientnetv2s_V2_deep_final.keras"
    ),

    "InceptionV3": os.path.join(
        FINAL_PROJECT,
        "Fresh_InceptionV3_V2",
        "fresh_inceptionv3_V2_phase1_final.keras"
    ),

    "EfficientNetB0": os.path.join(
        FINAL_PROJECT,
        "Fresh_EfficientNetB0_V2",
        "fresh_efficientnetb0_V2_final.keras"
    )
}


# Final validated ensemble weights.
# Selected using the validation set only in the completed
# three-model ensemble evaluation.
ENSEMBLE_WEIGHTS = {
    "EfficientNetV2S": 0.05,
    "InceptionV3": 0.40,
    "EfficientNetB0": 0.55
}


# Grayscale analysis settings.
MAX_CANDIDATES = 3
CANDIDATE_PERCENTILE = 94

MIN_AREA_FRACTION = 0.002
MAX_AREA_FRACTION = 0.18

USE_ASYMMETRY = True


# ============================================================
# 4. VERIFY FINAL MODEL FILES
# ============================================================

print("=" * 78)
print("VERIFYING FINAL MODELS")
print("=" * 78)

for model_name, model_path in MODEL_PATHS.items():

    exists = os.path.isfile(model_path)

    print(f"\n{model_name}")
    print(model_path)
    print("Exists:", exists)

    if not exists:
        raise FileNotFoundError(
            f"\n{model_name} was not found:\n"
            f"{model_path}"
        )

print("\n✅ All three final models were found.")


# ============================================================
# 5. UPLOAD ONE MRI IMAGE
# ============================================================

print("\n" + "=" * 78)
print("UPLOAD MRI IMAGE")
print("=" * 78)

uploaded = files.upload()

if not uploaded:
    raise ValueError("No MRI image was uploaded.")

uploaded_filename = next(iter(uploaded))

temporary_image_path = os.path.join(
    tempfile.gettempdir(),
    Path(uploaded_filename).name
)

with open(temporary_image_path, "wb") as output_file:
    output_file.write(
        uploaded[uploaded_filename]
    )

print("\n✅ Uploaded:")
print(temporary_image_path)


# ============================================================
# 6. LOAD AND PREPARE THE IMAGE
# ============================================================

original_pil_image = Image.open(
    temporary_image_path
).convert("RGB")

resized_pil_image = original_pil_image.resize(
    IMAGE_SIZE,
    Image.Resampling.BILINEAR
)

image_array = np.asarray(
    resized_pil_image,
    dtype=np.float32
)

image_batch = np.expand_dims(
    image_array,
    axis=0
)

rgb_image = np.clip(
    image_array,
    0,
    255
).astype(np.uint8)

gray_image = cv2.cvtColor(
    rgb_image,
    cv2.COLOR_RGB2GRAY
)

HEIGHT, WIDTH = gray_image.shape

print("\nPrepared image shape:", image_batch.shape)

print(
    "Pixel range:",
    float(image_batch.min()),
    "to",
    float(image_batch.max())
)


# ============================================================
# 7. NORMALIZATION HELPER
# ============================================================

def normalize_01(array):

    array = np.asarray(
        array,
        dtype=np.float32
    )

    minimum = float(np.min(array))
    maximum = float(np.max(array))

    if maximum <= minimum:
        return np.zeros_like(
            array,
            dtype=np.float32
        )

    return (
        array - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# 8. RUN FINAL THREE-MODEL ENSEMBLE
# ============================================================

individual_probabilities = {}

prediction_start_time = time.time()

print("\n" + "=" * 78)
print("INDIVIDUAL MODEL PREDICTIONS")
print("=" * 78)


for model_name, model_path in MODEL_PATHS.items():

    print(f"\nLoading {model_name}...")

    tf.keras.backend.clear_session()
    gc.collect()

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    probabilities = model.predict(
        image_batch,
        verbose=0
    )[0].astype(np.float64)

    probability_sum = float(
        np.sum(probabilities)
    )

    if probability_sum > 0:
        probabilities = probabilities / probability_sum

    individual_probabilities[
        model_name
    ] = probabilities

    predicted_index = int(
        np.argmax(probabilities)
    )

    print(
        f"{model_name:<18}: "
        f"{CLASS_NAMES[predicted_index]} "
        f"({probabilities[predicted_index] * 100:.2f}%)"
    )

    del model

    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# 9. CALCULATE FINAL WEIGHTED PREDICTION
# ============================================================

ensemble_probabilities = np.zeros(
    len(CLASS_NAMES),
    dtype=np.float64
)

for model_name in MODEL_PATHS:

    ensemble_probabilities += (
        ENSEMBLE_WEIGHTS[model_name]
        *
        individual_probabilities[model_name]
    )


ensemble_sum = float(
    np.sum(ensemble_probabilities)
)

if ensemble_sum > 0:
    ensemble_probabilities = (
        ensemble_probabilities / ensemble_sum
    )


final_class_index = int(
    np.argmax(ensemble_probabilities)
)

final_class_name = CLASS_NAMES[
    final_class_index
]

final_confidence = float(
    ensemble_probabilities[
        final_class_index
    ]
)

prediction_time = (
    time.time() - prediction_start_time
)


print("\n" + "=" * 78)
print("FINAL ENSEMBLE RESULT")
print("=" * 78)

print("\nPredicted class:", final_class_name)

print(
    "Confidence:",
    f"{final_confidence * 100:.2f}%"
)

print(
    "Classification processing time:",
    f"{prediction_time:.2f} seconds"
)

print("\nFinal class probabilities:")

for class_index, class_name in enumerate(
    CLASS_NAMES
):
    print(
        f"  {class_name:<12}: "
        f"{ensemble_probabilities[class_index] * 100:.2f}%"
    )


# ============================================================
# 10. NORMALIZE THE CLASS LABEL
# ============================================================

normalized_prediction = (
    str(final_class_name)
    .strip()
    .lower()
    .replace(" ", "")
    .replace("_", "")
    .replace("-", "")
)

is_no_tumor = normalized_prediction in {
    "notumor",
    "normal",
    "healthy",
    "noabnormality"
}


# ============================================================
# 11. STOP HERE WHEN THE RESULT IS NO-TUMOR
# ============================================================

if is_no_tumor:

    print("\n" + "=" * 78)
    print("GRAYSCALE ANALYSIS SKIPPED")
    print("=" * 78)

    print(
        "\n✅ The ensemble predicted the no-tumor class."
    )

    print(
        "Grayscale suspicious-region analysis was not run."
    )

    print(
        "No candidate boxes were generated."
    )


    figure = plt.figure(
        figsize=(12, 6)
    )


    # Original image
    axis = figure.add_subplot(
        1,
        2,
        1
    )

    axis.imshow(
        rgb_image
    )

    axis.set_title(
        "Uploaded Brain MRI",
        fontsize=13,
        fontweight="bold"
    )

    axis.axis("off")


    # Result panel
    axis = figure.add_subplot(
        1,
        2,
        2
    )

    axis.axis("off")

    result_text = (
        "FINAL MRI CLASSIFICATION\n\n"
        "Prediction: No Tumor\n\n"
        f"Confidence: {final_confidence * 100:.2f}%\n\n"
        "Suspicious-region analysis:\n"
        "Not performed\n\n"
        "No candidate boxes were generated because\n"
        "the final classification was no tumor."
    )

    axis.text(
        0.05,
        0.92,
        result_text,
        verticalalignment="top",
        fontsize=14,
        bbox={
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.95
        }
    )


    plt.suptitle(
        "Brain MRI Analysis",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 12. RUN GRAYSCALE ANALYSIS FOR TUMOR CLASSES ONLY
# ============================================================

else:

    print("\n" + "=" * 78)
    print("RUNNING GRAYSCALE SUSPICIOUS-REGION ANALYSIS")
    print("=" * 78)

    localization_start_time = time.time()


    # ========================================================
    # 12.1 CREATE APPROXIMATE HEAD MASK
    # ========================================================

    def create_head_mask(gray):

        blurred = cv2.GaussianBlur(
            gray,
            (7, 7),
            0
        )

        _, thresholded = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY
            +
            cv2.THRESH_OTSU
        )

        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15)
        )

        thresholded = cv2.morphologyEx(
            thresholded,
            cv2.MORPH_CLOSE,
            closing_kernel,
            iterations=2
        )

        number_of_labels, labels, statistics, _ = (
            cv2.connectedComponentsWithStats(
                thresholded,
                connectivity=8
            )
        )

        if number_of_labels <= 1:

            fallback = np.zeros_like(
                gray,
                dtype=np.uint8
            )

            cv2.ellipse(
                fallback,
                (
                    WIDTH // 2,
                    HEIGHT // 2
                ),
                (
                    int(WIDTH * 0.40),
                    int(HEIGHT * 0.42)
                ),
                0,
                0,
                360,
                1,
                -1
            )

            return fallback

        largest_label = (
            1
            +
            int(
                np.argmax(
                    statistics[
                        1:,
                        cv2.CC_STAT_AREA
                    ]
                )
            )
        )

        return (
            labels == largest_label
        ).astype(np.uint8)


    head_mask = create_head_mask(
        gray_image
    )


    # ========================================================
    # 12.2 REMOVE SKULL RIM
    # ========================================================

    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (19, 19)
    )

    inner_brain_mask = cv2.erode(
        head_mask,
        erosion_kernel,
        iterations=1
    )

    if np.sum(inner_brain_mask) < 1500:
        inner_brain_mask = head_mask.copy()


    deep_brain_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (25, 25)
    )

    deep_brain_mask = cv2.erode(
        head_mask,
        deep_brain_kernel,
        iterations=1
    )

    if np.sum(deep_brain_mask) < 1000:
        deep_brain_mask = inner_brain_mask.copy()


    # ========================================================
    # 12.3 DISTANCE FROM HEAD/SKULL BOUNDARY
    # ========================================================

    distance_from_boundary = cv2.distanceTransform(
        head_mask.astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    distance_from_boundary = normalize_01(
        distance_from_boundary
    )


    # ========================================================
    # 12.4 CONTRAST ENHANCEMENT
    # ========================================================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_gray = clahe.apply(
        gray_image
    )

    enhanced_01 = normalize_01(
        enhanced_gray
    )


    # ========================================================
    # 12.5 LOCAL BRIGHTNESS DIFFERENCE
    # ========================================================

    large_blur = cv2.GaussianBlur(
        enhanced_gray.astype(np.float32),
        (31, 31),
        0
    )

    bright_difference = np.maximum(
        enhanced_gray.astype(np.float32)
        -
        large_blur,
        0
    )

    bright_difference = normalize_01(
        bright_difference
    )


    # ========================================================
    # 12.6 LOCAL TEXTURE
    # ========================================================

    local_mean = cv2.GaussianBlur(
        enhanced_gray.astype(np.float32),
        (15, 15),
        0
    )

    local_squared_mean = cv2.GaussianBlur(
        enhanced_gray.astype(np.float32) ** 2,
        (15, 15),
        0
    )

    local_variance = np.maximum(
        local_squared_mean
        -
        local_mean ** 2,
        0
    )

    local_texture = normalize_01(
        np.sqrt(local_variance)
    )


    # ========================================================
    # 12.7 EDGE RESPONSE
    # ========================================================

    laplacian = cv2.Laplacian(
        enhanced_gray,
        cv2.CV_32F,
        ksize=3
    )

    edge_response = normalize_01(
        np.abs(laplacian)
    )


    # ========================================================
    # 12.8 LEFT-RIGHT ASYMMETRY
    # ========================================================

    if USE_ASYMMETRY:

        flipped_gray = cv2.flip(
            enhanced_gray,
            1
        )

        asymmetry_map = normalize_01(
            cv2.absdiff(
                enhanced_gray,
                flipped_gray
            )
        )

    else:

        asymmetry_map = np.zeros_like(
            enhanced_01,
            dtype=np.float32
        )


    # ========================================================
    # 12.9 CREATE SUSPICION MAP
    # ========================================================

    suspicion_map = (
        0.36 * bright_difference
        +
        0.28 * local_texture
        +
        0.16 * edge_response
        +
        0.12 * enhanced_01
        +
        0.08 * asymmetry_map
    )

    suspicion_map *= inner_brain_mask

    suspicion_map = cv2.GaussianBlur(
        suspicion_map,
        (9, 9),
        0
    )

    suspicion_map = normalize_01(
        suspicion_map
    )


    # ========================================================
    # 12.10 CREATE BINARY CANDIDATE MASK
    # ========================================================

    brain_values = suspicion_map[
        inner_brain_mask > 0
    ]

    if brain_values.size == 0:
        raise RuntimeError(
            "The inner-brain mask is empty."
        )

    candidate_threshold = float(
        np.percentile(
            brain_values,
            CANDIDATE_PERCENTILE
        )
    )

    candidate_mask = (
        (
            suspicion_map
            >=
            candidate_threshold
        )
        &
        (
            inner_brain_mask > 0
        )
    ).astype(np.uint8)


    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        ),
        iterations=1
    )

    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        ),
        iterations=2
    )


    # ========================================================
    # 12.11 FIND CONNECTED COMPONENTS
    # ========================================================

    (
        number_of_labels,
        labels,
        statistics,
        centroids
    ) = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8
    )

    brain_area = max(
        1,
        int(
            np.sum(inner_brain_mask)
        )
    )

    minimum_area = max(
        15,
        int(
            MIN_AREA_FRACTION
            *
            brain_area
        )
    )

    maximum_area = max(
        minimum_area + 1,
        int(
            MAX_AREA_FRACTION
            *
            brain_area
        )
    )


    # Head boundary ring.
    head_boundary = (
        head_mask
        -
        cv2.erode(
            head_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9)
            ),
            iterations=1
        )
    )


    # ========================================================
    # 12.12 SCORE CANDIDATES
    # ========================================================

    candidate_components = []


    for label_index in range(
        1,
        number_of_labels
    ):

        x = int(
            statistics[
                label_index,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            statistics[
                label_index,
                cv2.CC_STAT_TOP
            ]
        )

        width = int(
            statistics[
                label_index,
                cv2.CC_STAT_WIDTH
            ]
        )

        height = int(
            statistics[
                label_index,
                cv2.CC_STAT_HEIGHT
            ]
        )

        area = int(
            statistics[
                label_index,
                cv2.CC_STAT_AREA
            ]
        )


        if area < minimum_area:
            continue

        if area > maximum_area:
            continue


        component_mask = (
            labels == label_index
        ).astype(np.uint8)


        component_pixels = (
            component_mask > 0
        )


        mean_suspicion = float(
            np.mean(
                suspicion_map[
                    component_pixels
                ]
            )
        )

        maximum_suspicion = float(
            np.max(
                suspicion_map[
                    component_pixels
                ]
            )
        )

        mean_brightness = float(
            np.mean(
                bright_difference[
                    component_pixels
                ]
            )
        )

        mean_texture = float(
            np.mean(
                local_texture[
                    component_pixels
                ]
            )
        )


        # ----------------------------------------------------
        # Shape analysis
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue


        contour = max(
            contours,
            key=cv2.contourArea
        )

        contour_area = float(
            cv2.contourArea(contour)
        )

        perimeter = float(
            cv2.arcLength(
                contour,
                True
            )
        )


        if perimeter > 0:

            circularity = float(
                4.0
                *
                np.pi
                *
                contour_area
                /
                (
                    perimeter ** 2
                )
            )

        else:

            circularity = 0.0


        circularity = float(
            np.clip(
                circularity,
                0.0,
                1.0
            )
        )


        convex_hull = cv2.convexHull(
            contour
        )

        hull_area = float(
            cv2.contourArea(
                convex_hull
            )
        )


        if hull_area > 0:

            solidity = float(
                contour_area
                /
                hull_area
            )

        else:

            solidity = 0.0


        solidity = float(
            np.clip(
                solidity,
                0.0,
                1.0
            )
        )


        aspect_ratio = float(
            min(width, height)
            /
            max(width, height)
        )


        rectangular_area = max(
            1,
            width * height
        )

        fill_ratio = float(
            area
            /
            rectangular_area
        )


        # ----------------------------------------------------
        # Internal-brain position
        # ----------------------------------------------------

        mean_boundary_distance = float(
            np.mean(
                distance_from_boundary[
                    component_pixels
                ]
            )
        )


        deep_overlap = float(
            np.sum(
                component_mask
                *
                deep_brain_mask
            )
            /
            (
                area + 1e-8
            )
        )


        boundary_contact_ratio = float(
            np.sum(
                component_mask
                *
                head_boundary
            )
            /
            (
                area + 1e-8
            )
        )


        # ----------------------------------------------------
        # Local contrast
        # ----------------------------------------------------

        expanded_mask = cv2.dilate(
            component_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (15, 15)
            ),
            iterations=1
        )

        surrounding_ring = (
            expanded_mask
            -
            component_mask
        )

        surrounding_ring = (
            surrounding_ring > 0
        ).astype(np.uint8)

        surrounding_ring *= inner_brain_mask


        candidate_intensity = float(
            np.mean(
                enhanced_01[
                    component_pixels
                ]
            )
        )


        if np.sum(surrounding_ring) > 0:

            surrounding_intensity = float(
                np.mean(
                    enhanced_01[
                        surrounding_ring > 0
                    ]
                )
            )

        else:

            surrounding_intensity = (
                candidate_intensity
            )


        local_contrast = float(
            np.clip(
                candidate_intensity
                -
                surrounding_intensity,
                0.0,
                1.0
            )
        )


        # ----------------------------------------------------
        # Area score
        # ----------------------------------------------------

        area_fraction = float(
            area / brain_area
        )

        ideal_area_fraction = 0.025

        area_score = float(
            np.exp(
                -abs(
                    area_fraction
                    -
                    ideal_area_fraction
                )
                /
                0.045
            )
        )


        # ----------------------------------------------------
        # Improved score
        # ----------------------------------------------------

        improved_score = (
            0.23 * mean_suspicion
            +
            0.12 * maximum_suspicion
            +
            0.12 * mean_brightness
            +
            0.09 * mean_texture
            +
            0.13 * local_contrast
            +
            0.08 * solidity
            +
            0.06 * circularity
            +
            0.05 * aspect_ratio
            +
            0.05 * area_score
            +
            0.04 * mean_boundary_distance
            +
            0.03 * deep_overlap
        )


        # Penalize boundary contact.
        improved_score *= (
            1.0
            -
            0.75
            *
            boundary_contact_ratio
        )


        # Penalize very elongated components.
        if aspect_ratio < 0.25:
            improved_score *= 0.45

        elif aspect_ratio < 0.40:
            improved_score *= 0.70


        improved_score = float(
            np.clip(
                improved_score,
                0.0,
                1.0
            )
        )


        candidate_components.append(
            {
                "label": label_index,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "score": improved_score,
                "mean_suspicion":
                    mean_suspicion,
                "maximum_suspicion":
                    maximum_suspicion,
                "local_contrast":
                    local_contrast,
                "circularity":
                    circularity,
                "solidity":
                    solidity,
                "aspect_ratio":
                    aspect_ratio,
                "deep_overlap":
                    deep_overlap,
                "boundary_contact":
                    boundary_contact_ratio
            }
        )


    # ========================================================
    # 12.13 RANK TOP THREE CANDIDATES
    # ========================================================

    improved_ranking = sorted(
        candidate_components,
        key=lambda candidate: (
            candidate["score"]
        ),
        reverse=True
    )

    top_candidates = improved_ranking[
        :MAX_CANDIDATES
    ]


    # ========================================================
    # 12.14 DRAW CANDIDATE BOXES
    # ========================================================

    boxed_image = rgb_image.copy()

    box_colors = [
        (255, 0, 0),       # red
        (255, 165, 0),     # orange
        (255, 215, 0)      # yellow
    ]


    for candidate_number, candidate in enumerate(
        top_candidates,
        start=1
    ):

        x = candidate["x"]
        y = candidate["y"]
        width = candidate["width"]
        height = candidate["height"]


        margin_x = max(
            4,
            int(width * 0.10)
        )

        margin_y = max(
            4,
            int(height * 0.10)
        )


        x1 = max(
            0,
            x - margin_x
        )

        y1 = max(
            0,
            y - margin_y
        )

        x2 = min(
            WIDTH - 1,
            x + width + margin_x
        )

        y2 = min(
            HEIGHT - 1,
            y + height + margin_y
        )


        color = box_colors[
            min(
                candidate_number - 1,
                len(box_colors) - 1
            )
        ]


        cv2.rectangle(
            boxed_image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        cv2.putText(
            boxed_image,
            f"Candidate {candidate_number}",
            (
                x1,
                max(16, y1 - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # 12.15 CREATE SUSPICION OVERLAY
    # ========================================================

    colored_suspicion = plt.get_cmap(
        "jet"
    )(
        suspicion_map
    )[
        ...,
        :3
    ]

    normalized_rgb = (
        rgb_image.astype(np.float32)
        /
        255.0
    )

    suspicion_overlay = np.clip(
        0.60
        *
        normalized_rgb
        +
        0.40
        *
        colored_suspicion,
        0.0,
        1.0
    )


    localization_time = (
        time.time()
        -
        localization_start_time
    )


    # ========================================================
    # 12.16 DISPLAY FINAL DASHBOARD-STYLE RESULT
    # ========================================================

    figure = plt.figure(
        figsize=(20, 10)
    )


    # Original MRI
    axis = figure.add_subplot(
        2,
        3,
        1
    )

    axis.imshow(
        rgb_image
    )

    axis.set_title(
        "Original Brain MRI",
        fontweight="bold"
    )

    axis.axis("off")


    # Suspicion score
    axis = figure.add_subplot(
        2,
        3,
        2
    )

    score_plot = axis.imshow(
        suspicion_map,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axis.set_title(
        "Grayscale Suspicion Score",
        fontweight="bold"
    )

    axis.axis("off")

    figure.colorbar(
        score_plot,
        ax=axis,
        fraction=0.046,
        pad=0.04
    )


    # Suspicion overlay
    axis = figure.add_subplot(
        2,
        3,
        3
    )

    axis.imshow(
        suspicion_overlay
    )

    axis.set_title(
        "Suspicion Overlay",
        fontweight="bold"
    )

    axis.axis("off")


    # Candidate mask
    axis = figure.add_subplot(
        2,
        3,
        4
    )

    axis.imshow(
        candidate_mask,
        cmap="gray"
    )

    axis.set_title(
        "Detected Candidate Components",
        fontweight="bold"
    )

    axis.axis("off")


    # Boxed image
    axis = figure.add_subplot(
        2,
        3,
        5
    )

    axis.imshow(
        boxed_image
    )

    axis.set_title(
        "Automatically Detected\nSuspicious Regions",
        fontweight="bold"
    )

    axis.axis("off")


    # Information panel
    axis = figure.add_subplot(
        2,
        3,
        6
    )

    axis.axis("off")


    information_text = (
        "FINAL MRI CLASSIFICATION\n\n"
        f"Prediction: {final_class_name}\n"
        f"Confidence: {final_confidence * 100:.2f}%\n\n"
        f"Suspicious candidates: {len(top_candidates)}\n"
        f"Localization time: {localization_time:.2f} sec\n\n"
        "Candidate colors:\n"
        "Red: Candidate 1\n"
        "Orange: Candidate 2\n"
        "Yellow: Candidate 3\n\n"
        "The highlighted boxes are automatically\n"
        "detected suspicious-region candidates.\n\n"
        "They support physician review and do not\n"
        "represent confirmed tumor boundaries."
    )


    axis.text(
        0.03,
        0.96,
        information_text,
        verticalalignment="top",
        fontsize=12,
        bbox={
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.95
        }
    )


    plt.suptitle(
        "Brain MRI Classification and Suspicious-Region Assistance",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


    # ========================================================
    # 12.17 PRINT CANDIDATE DETAILS
    # ========================================================

    print("\n" + "=" * 78)
    print("SUSPICIOUS-REGION ANALYSIS COMPLETED")
    print("=" * 78)

    print(
        "\nPrediction:",
        final_class_name
    )

    print(
        "Confidence:",
        f"{final_confidence * 100:.2f}%"
    )

    print(
        "Candidate threshold:",
        f"{candidate_threshold:.3f}"
    )

    print(
        "Candidates detected:",
        len(top_candidates)
    )


    if not top_candidates:

        print(
            "\n⚠️ No stable suspicious region was detected."
        )

    else:

        for candidate_number, candidate in enumerate(
            top_candidates,
            start=1
        ):

            print(
                f"\nCandidate {candidate_number}"
            )

            print(
                "  Box:",
                (
                    candidate["x"],
                    candidate["y"],
                    candidate["width"],
                    candidate["height"]
                )
            )

            print(
                "  Score:",
                f"{candidate['score']:.3f}"
            )

            print(
                "  Local contrast:",
                f"{candidate['local_contrast']:.3f}"
            )

            print(
                "  Circularity:",
                f"{candidate['circularity']:.3f}"
            )

            print(
                "  Solidity:",
                f"{candidate['solidity']:.3f}"
            )


    print(
        "\nImportant:"
    )

    print(
        "The boxes represent suspicious-region candidates "
        "for physician review."
    )

    print(
        "They are not confirmed tumor boundaries or a "
        "replacement for clinical assessment."
    )
