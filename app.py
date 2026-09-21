# ============================================================
# BRAIN MRI TUMOR CLASSIFICATION DASHBOARD
#
# - Loads the 3 trained models ONCE per session (st.cache_resource)
#   so re-uploading an image never reloads the models.
# - Runs the fixed-weight ensemble on the uploaded image.
# - If a tumor class is predicted, runs the grayscale
#   suspicious-region analysis and shows up to 3 candidate
#   boxes drawn on the image (red / orange / yellow).
# - If "notumor" is predicted, no boxes are drawn.
# ============================================================

import os
import io
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain MRI Tumor Classification",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)

MODEL_DIR = "models"

MODEL_PATHS = {
    "EfficientNetV2S": os.path.join(MODEL_DIR, "efficientnetv2s.keras"),
    "InceptionV3": os.path.join(MODEL_DIR, "inceptionv3.keras"),
    "EfficientNetB0": os.path.join(MODEL_DIR, "efficientnetb0.keras"),
}

ENSEMBLE_WEIGHTS = {
    "EfficientNetV2S": 0.05,
    "InceptionV3": 0.40,
    "EfficientNetB0": 0.55,
}

MAX_CANDIDATES = 3
CANDIDATE_PERCENTILE = 94
MIN_AREA_FRACTION = 0.002
MAX_AREA_FRACTION = 0.18
USE_ASYMMETRY = True

BOX_COLORS_BGR = [
    (0, 0, 255),
    (0, 165, 255),
    (0, 215, 255),
]
BADGE_LABELS = ["Candidate 1", "Candidate 2", "Candidate 3"]
BADGE_HEX = ["#E24B4A", "#EF9F27", "#FAC775"]
BADGE_COLOR_NAMES = ["Red", "Orange", "Yellow"]

CLASS_COLORS = {
    "glioma": "#8B7FE8",
    "meningioma": "#3E9DE0",
    "pituitary": "#FF5C77",
    "notumor": "#12B5A6",
}
CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary tumor",
    "notumor": "No tumor detected",
}
CLASS_ICONS = {
    "glioma": "◆",
    "meningioma": "●",
    "pituitary": "▲",
    "notumor": "✓",
}

CLASS_EXPLANATIONS = {
    "glioma": (
        "Gliomas most frequently arise in the frontal and temporal lobes and "
        "can occur at any age. Depending on grade and location, they may "
        "present with headaches, seizures, or focal neurological deficits. "
        "Growth pattern and enhancement characteristics on MRI are typically "
        "used alongside grading to guide management."
    ),
    "meningioma": (
        "Meningiomas arise from the meninges rather than brain tissue itself "
        "and are usually extra-axial, well-circumscribed masses. They are "
        "often slow-growing and can be asymptomatic, but larger lesions may "
        "cause mass effect on adjacent structures depending on their location."
    ),
    "pituitary": (
        "Pituitary tumors arise in the sellar region and can affect hormone "
        "production or, when large, compress the optic chiasm and affect "
        "vision. Endocrine work-up alongside imaging is typically used to "
        "characterize functional status."
    ),
    "notumor": (
        "No tumor pattern was detected by the ensemble. This reflects the "
        "model's assessment of the uploaded scan only and does not rule out "
        "findings outside the classifier's scope."
    ),
}

# ============================================================
# PROJECT FACTS — used by the "About the Project" page.
# Updated to match the completed three-model ensemble evaluation.
# ============================================================

MODEL_ACCURACY = {
    "EfficientNetV2S": {"internal": 81.88, "external": 79.86},
    "InceptionV3": {"internal": 82.69, "external": 82.01},
    "EfficientNetB0": {"internal": 85.48, "external": 81.29},
    "Ensemble": {"internal": 86.99, "external": 84.89},
}

TEST_SET_COUNTS = {
    "internal": {"glioma": 222, "meningioma": 218, "notumor": 203, "pituitary": 218},
    "external": {"glioma": 70, "meningioma": 70, "notumor": 65, "pituitary": 73},
}

PIPELINE_STEPS = [
    ("MRI upload", "◐"),
    ("3 CNN models", "▦"),
    ("Weighted ensemble", "∑"),
    ("Suspicious-region scan", "◎"),
    ("Result + boxes", "✓"),
]

DATASET_SPLITS = [
    ("Training (original)", 3917, "#0B1F3A"),
    ("Training (added external)", 650, "#12B5A6"),
    ("Validation", 855, "#8B7FE8"),
    ("Internal test", 861, "#3E9DE0"),
    ("External test", 278, "#FF9F5C"),
]

PER_CLASS_ACCURACY = {
    "internal": {"glioma": 83.78, "meningioma": 85.32, "notumor": 80.79, "pituitary": 97.71},
    "external": {"glioma": 70.00, "meningioma": 72.86, "notumor": 98.46, "pituitary": 98.63},
}

PREPROCESSING_STEPS = [
    ("EfficientNetV2S", "224 × 224", "Preprocessing embedded in model"),
    ("InceptionV3", "224 × 224", "Preprocessing embedded in model"),
    ("EfficientNetB0", "224 × 224", "Preprocessing embedded in model"),
]

ACCURACY_PROGRESSION = [
    ("Baseline (EfficientNetB0)", 87.02, "#8B95A5"),
    ("Best single (EfficientNetV2S)", 89.76, "#8B95A5"),
    ("Deep fine-tuning", 90.24, "#8B95A5"),
    ("V12 ensemble — internal", 91.67, "#3E9DE0"),
    ("V12 ensemble — external", 58.50, "#E24B4A"),
    ("Final ensemble — internal", 86.99, "#12B5A6"),
    ("Final ensemble — external", 84.89, "#0E8F84"),
]

ENSEMBLE_WEIGHTS_DISPLAY = [
    ("EfficientNetV2S", 5.0, "#8B7FE8"),
    ("InceptionV3", 40.0, "#3E9DE0"),
    ("EfficientNetB0", 55.0, "#12B5A6"),
]



@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        models[model_name] = tf.keras.models.load_model(
            model_path, compile=False
        )
    return models


def normalize_01(array):
    array = np.asarray(array, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def create_head_mask(gray, width, height):
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresholded = cv2.morphologyEx(
        thresholded, cv2.MORPH_CLOSE, closing_kernel, iterations=2
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresholded, connectivity=8
    )
    if num_labels <= 1:
        fallback = np.zeros_like(gray, dtype=np.uint8)
        cv2.ellipse(
            fallback, (width // 2, height // 2),
            (int(width * 0.40), int(height * 0.42)), 0, 0, 360, 1, -1
        )
        return fallback
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


@st.cache_data(show_spinner=False)
def run_pipeline(image_bytes):
    models = load_models()

    original_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_pil = original_pil.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(resized_pil, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)
    rgb_image = np.clip(image_array, 0, 255).astype(np.uint8)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    height, width = gray_image.shape

    individual_probs = {}
    for model_name, model in models.items():
        probs = model.predict(image_batch, verbose=0)[0].astype(np.float64)
        total = float(np.sum(probs))
        if total > 0:
            probs = probs / total
        individual_probs[model_name] = probs

    ensemble_probs = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for model_name in MODEL_PATHS:
        ensemble_probs += ENSEMBLE_WEIGHTS[model_name] * individual_probs[model_name]
    total = float(np.sum(ensemble_probs))
    if total > 0:
        ensemble_probs = ensemble_probs / total

    final_index = int(np.argmax(ensemble_probs))
    final_class = CLASS_NAMES[final_index]
    final_confidence = float(ensemble_probs[final_index])

    is_no_tumor = final_class == "notumor"

    result = {
        "rgb_image": rgb_image,
        "final_class": final_class,
        "final_confidence": final_confidence,
        "class_probs": {c: float(ensemble_probs[i]) for i, c in enumerate(CLASS_NAMES)},
        "candidates": [],
        "boxed_image": rgb_image,
    }

    if is_no_tumor:
        return result

    head_mask = create_head_mask(gray_image, width, height)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    inner_brain_mask = cv2.erode(head_mask, erosion_kernel, iterations=1)
    if np.sum(inner_brain_mask) < 1500:
        inner_brain_mask = head_mask.copy()

    deep_brain_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    deep_brain_mask = cv2.erode(head_mask, deep_brain_kernel, iterations=1)
    if np.sum(deep_brain_mask) < 1000:
        deep_brain_mask = inner_brain_mask.copy()

    distance_from_boundary = normalize_01(
        cv2.distanceTransform(head_mask.astype(np.uint8), cv2.DIST_L2, 5)
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_image)
    enhanced_01 = normalize_01(enhanced_gray)

    large_blur = cv2.GaussianBlur(enhanced_gray.astype(np.float32), (31, 31), 0)
    bright_difference = normalize_01(
        np.maximum(enhanced_gray.astype(np.float32) - large_blur, 0)
    )

    local_mean = cv2.GaussianBlur(enhanced_gray.astype(np.float32), (15, 15), 0)
    local_sq_mean = cv2.GaussianBlur(enhanced_gray.astype(np.float32) ** 2, (15, 15), 0)
    local_variance = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_texture = normalize_01(np.sqrt(local_variance))

    laplacian = cv2.Laplacian(enhanced_gray, cv2.CV_32F, ksize=3)
    edge_response = normalize_01(np.abs(laplacian))

    if USE_ASYMMETRY:
        flipped_gray = cv2.flip(enhanced_gray, 1)
        asymmetry_map = normalize_01(cv2.absdiff(enhanced_gray, flipped_gray))
    else:
        asymmetry_map = np.zeros_like(enhanced_01, dtype=np.float32)

    suspicion_map = (
        0.36 * bright_difference + 0.28 * local_texture + 0.16 * edge_response
        + 0.12 * enhanced_01 + 0.08 * asymmetry_map
    )
    suspicion_map *= inner_brain_mask
    suspicion_map = cv2.GaussianBlur(suspicion_map, (9, 9), 0)
    suspicion_map = normalize_01(suspicion_map)

    brain_values = suspicion_map[inner_brain_mask > 0]
    if brain_values.size == 0:
        return result

    candidate_threshold = float(np.percentile(brain_values, CANDIDATE_PERCENTILE))
    candidate_mask = (
        (suspicion_map >= candidate_threshold) & (inner_brain_mask > 0)
    ).astype(np.uint8)
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )
    brain_area = max(1, int(np.sum(inner_brain_mask)))
    minimum_area = max(15, int(MIN_AREA_FRACTION * brain_area))
    maximum_area = max(minimum_area + 1, int(MAX_AREA_FRACTION * brain_area))

    head_boundary = head_mask - cv2.erode(
        head_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
    )

    candidates = []
    for label_index in range(1, num_labels):
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        w = int(stats[label_index, cv2.CC_STAT_WIDTH])
        h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])

        if area < minimum_area or area > maximum_area:
            continue

        component_mask = (labels == label_index).astype(np.uint8)
        component_pixels = component_mask > 0

        mean_suspicion = float(np.mean(suspicion_map[component_pixels]))
        max_suspicion = float(np.max(suspicion_map[component_pixels]))
        mean_brightness = float(np.mean(bright_difference[component_pixels]))
        mean_texture = float(np.mean(local_texture[component_pixels]))

        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = float(np.clip(
            4.0 * np.pi * contour_area / (perimeter ** 2) if perimeter > 0 else 0.0,
            0.0, 1.0
        ))
        convex_hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(convex_hull))
        solidity = float(np.clip(contour_area / hull_area if hull_area > 0 else 0.0, 0.0, 1.0))
        aspect_ratio = float(min(w, h) / max(w, h))

        mean_boundary_distance = float(np.mean(distance_from_boundary[component_pixels]))
        deep_overlap = float(np.sum(component_mask * deep_brain_mask) / (area + 1e-8))
        boundary_contact_ratio = float(np.sum(component_mask * head_boundary) / (area + 1e-8))

        expanded_mask = cv2.dilate(
            component_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1
        )
        surrounding_ring = ((expanded_mask - component_mask) > 0).astype(np.uint8)
        surrounding_ring *= inner_brain_mask

        candidate_intensity = float(np.mean(enhanced_01[component_pixels]))
        if np.sum(surrounding_ring) > 0:
            surrounding_intensity = float(np.mean(enhanced_01[surrounding_ring > 0]))
        else:
            surrounding_intensity = candidate_intensity
        local_contrast = float(np.clip(candidate_intensity - surrounding_intensity, 0.0, 1.0))

        area_fraction = float(area / brain_area)
        area_score = float(np.exp(-abs(area_fraction - 0.025) / 0.045))

        score = (
            0.23 * mean_suspicion + 0.12 * max_suspicion + 0.12 * mean_brightness
            + 0.09 * mean_texture + 0.13 * local_contrast + 0.08 * solidity
            + 0.06 * circularity + 0.05 * aspect_ratio + 0.05 * area_score
            + 0.04 * mean_boundary_distance + 0.03 * deep_overlap
        )
        score *= (1.0 - 0.75 * boundary_contact_ratio)
        if aspect_ratio < 0.25:
            score *= 0.45
        elif aspect_ratio < 0.40:
            score *= 0.70
        score = float(np.clip(score, 0.0, 1.0))

        candidates.append({
            "x": x, "y": y, "width": w, "height": h,
            "score": score, "local_contrast": local_contrast,
            "circularity": circularity, "solidity": solidity,
        })

    top_candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)[:MAX_CANDIDATES]

    boxed_image = rgb_image.copy()
    for i, candidate in enumerate(top_candidates):
        x, y, w, h = candidate["x"], candidate["y"], candidate["width"], candidate["height"]
        margin_x = max(4, int(w * 0.10))
        margin_y = max(4, int(h * 0.10))
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(width - 1, x + w + margin_x)
        y2 = min(height - 1, y + h + margin_y)
        color = BOX_COLORS_BGR[min(i, len(BOX_COLORS_BGR) - 1)]
        color_rgb = (color[2], color[1], color[0])
        cv2.rectangle(boxed_image, (x1, y1), (x2, y2), color_rgb, 2)

    result["candidates"] = top_candidates
    result["boxed_image"] = boxed_image
    return result


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    .stApp {
        background: #F4F7FA;
        font-family: 'Inter', sans-serif;
    }

    .bc-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 20px 26px;
        border-radius: 14px;
        background: #FFFFFF;
        border: 1px solid #E3E9F0;
        border-top: 5px solid #12B5A6;
        box-shadow: 0 2px 12px rgba(11, 31, 58, 0.05);
        margin-bottom: 22px;
        flex-wrap: wrap;
    }
    .bc-header-left { display: flex; align-items: center; gap: 14px; }
    .bc-header-icon {
        width: 44px; height: 44px; border-radius: 12px;
        background: linear-gradient(135deg, #0B1F3A, #12B5A6);
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; color: #FFFFFF; flex-shrink: 0;
    }
    .bc-header-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700; font-size: 21px; color: #0B1F3A;
    }
    .bc-header-sub { font-size: 12.5px; color: #6B7788; margin-top: 1px; }
    .bc-header-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; color: #12776E;
        background: #E7F7F4; border: 1px solid #C6EDE6;
        padding: 6px 12px; border-radius: 999px; white-space: nowrap;
    }

    section[data-testid="stSidebar"] {
        background: #0B1F3A;
    }
    section[data-testid="stSidebar"] * { color: #E7ECF3; }
    .bc-side-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700; font-size: 18px; color: #FFFFFF;
        margin-bottom: 2px;
    }
    .bc-side-sub { font-size: 11.5px; color: #7FD9CE; margin-bottom: 18px; }
    .bc-side-section-title {
        font-family: 'Inter', sans-serif; font-weight: 600; font-size: 11.5px;
        color: #7FD9CE; text-transform: uppercase; letter-spacing: 0.6px;
        margin: 16px 0 8px 0;
    }
    .bc-side-text { font-size: 12.5px; color: #C3CBD8; line-height: 1.55; }
    .bc-side-stat {
        font-family: 'IBM Plex Mono', monospace; font-size: 12px;
        color: #E7ECF3; background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 8px; padding: 6px 10px; margin-bottom: 6px;
    }
    .bc-legend-row {
        display: flex; align-items: center; gap: 8px;
        font-size: 12.5px; color: #C3CBD8; margin-bottom: 6px;
    }
    .bc-legend-dot {
        width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
    }

    .bc-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 12px rgba(11, 31, 58, 0.05);
        border: 1px solid #E3E9F0;
        height: 100%;
    }
    .bc-card-label {
        font-family: 'Inter', sans-serif; font-weight: 600; font-size: 12px;
        color: #5B6472; text-transform: uppercase; letter-spacing: 0.6px;
        margin-bottom: 10px;
    }

    .bc-hero {
        display: flex; align-items: center; justify-content: space-between;
        gap: 20px; background: #FFFFFF; border-radius: 14px;
        padding: 20px 24px; box-shadow: 0 2px 12px rgba(11, 31, 58, 0.05);
        border: 1px solid #E3E9F0; margin-bottom: 20px; flex-wrap: wrap;
    }
    .bc-eyebrow {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px;
        letter-spacing: 1.2px; color: #8B95A5; text-transform: uppercase;
        margin-bottom: 6px;
    }
    .bc-hero-icon { font-size: 30px; line-height: 1; }
    .bc-hero-label {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 28px; margin: 0; line-height: 1.15;
    }
    .bc-hero-conf {
        font-family: 'IBM Plex Mono', monospace; font-size: 12.5px;
        color: #6B7788; margin-top: 3px;
    }
    .bc-gauge {
        width: 92px; height: 92px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; margin-left: auto;
    }
    .bc-gauge-inner {
        width: 70px; height: 70px; border-radius: 50%; background: #FFFFFF;
        display: flex; align-items: center; justify-content: center;
    }
    .bc-gauge-pct {
        font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 14.5px;
    }

    .bc-prob-row { margin-bottom: 10px; }
    .bc-prob-top {
        display: flex; justify-content: space-between;
        font-family: 'IBM Plex Mono', monospace; font-size: 11.5px;
        color: #4A5568; margin-bottom: 4px;
    }
    .bc-prob-track {
        width: 100%; height: 8px; border-radius: 999px;
        background: #EDF1F6; overflow: hidden;
    }
    .bc-prob-fill { height: 100%; border-radius: 999px; }

    .bc-candidate {
        border-radius: 12px; padding: 12px 14px; background: #FFFFFF;
        border: 1px solid #E3E9F0; border-left: 5px solid var(--c);
        display: flex; align-items: flex-start; gap: 10px;
    }
    .bc-candidate-swatch {
        width: 26px; height: 26px; border-radius: 7px; background: var(--c);
        flex-shrink: 0; margin-top: 2px;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.06);
    }
    .bc-candidate-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 20px; height: 20px; border-radius: 50%; background: var(--c);
        color: #FFFFFF; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
        font-weight: 600; margin-right: 8px;
    }
    .bc-candidate-match {
        font-size: 12.5px; color: #2B2F38; margin-top: 2px;
    }
    .bc-candidate-score {
        font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: #6B7788;
        margin-top: 2px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #12B5A6, #0E8F84);
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 14.5px;
        border: none;
        border-radius: 10px;
        padding: 12px 22px;
        box-shadow: 0 4px 14px rgba(18, 181, 166, 0.35);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #14C7B7, #0FA396);
        box-shadow: 0 6px 18px rgba(18, 181, 166, 0.45);
        transform: translateY(-1px);
        color: #FFFFFF;
        border: none;
    }
    .stButton > button:active {
        transform: translateY(0px);
    }
    .stButton > button:focus:not(:active) {
        color: #FFFFFF;
        border: none;
    }

    .bc-disclaimer {
        display: flex; gap: 10px; align-items: flex-start;
        background: #FFF7E8; border: 1px solid #F5D999; border-radius: 12px;
        padding: 14px 16px; font-size: 13px; color: #6B4E14; margin-top: 18px;
    }

    /* ---------- about page ---------- */
    .bc-about-hero {
        background: linear-gradient(135deg, #0B1F3A, #12776E);
        border-radius: 16px; padding: 28px 30px; margin-bottom: 20px;
        color: #FFFFFF;
    }
    .bc-about-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 24px; margin-bottom: 6px;
    }
    .bc-about-sub {
        font-size: 13.5px; color: #C9E9E4; line-height: 1.6; max-width: 640px;
    }
    .bc-stat {
        background: #FFFFFF; border-radius: 14px; padding: 18px 20px;
        border: 1px solid #E3E9F0; box-shadow: 0 2px 12px rgba(11,31,58,0.05);
        text-align: center;
    }
    .bc-stat-value {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 26px; color: #12776E;
    }
    .bc-stat-label {
        font-size: 11.5px; color: #6B7788; text-transform: uppercase;
        letter-spacing: 0.5px; margin-top: 4px;
    }
    .bc-pipeline {
        display: flex; align-items: center; justify-content: space-between;
        gap: 6px; flex-wrap: wrap; margin: 4px 0 2px 0;
    }
    .bc-pipeline-step {
        display: flex; flex-direction: column; align-items: center; gap: 6px;
        flex: 1; min-width: 90px;
    }
    .bc-pipeline-icon {
        width: 44px; height: 44px; border-radius: 50%;
        background: #E7F7F4; color: #12776E; font-size: 18px;
        display: flex; align-items: center; justify-content: center;
    }
    .bc-pipeline-label {
        font-size: 11px; color: #4A5568; text-align: center; line-height: 1.3;
    }
    .bc-pipeline-arrow { color: #C3CBD8; font-size: 16px; }
    .bc-cbar-row { margin-bottom: 12px; }
    .bc-cbar-top {
        display: flex; justify-content: space-between;
        font-size: 12.5px; color: #2B2F38; margin-bottom: 4px;
    }
    .bc-cbar-track {
        width: 100%; height: 10px; border-radius: 999px;
        background: #EDF1F6; overflow: hidden;
    }
    .bc-cbar-fill { height: 100%; border-radius: 999px; }
    .bc-class-chip {
        background: #FFFFFF; border: 1px solid #E3E9F0; border-radius: 12px;
        padding: 10px 14px; text-align: center;
    }
    .bc-class-chip-count {
        font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #2B2F38;
    }
    .bc-section-head {
        display: flex; align-items: center; gap: 10px;
        margin: 6px 0 12px 0;
    }
    .bc-section-num {
        width: 26px; height: 26px; border-radius: 50%;
        background: #0B1F3A; color: #FFFFFF;
        display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600;
        flex-shrink: 0;
    }
    .bc-section-title-text {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 16px; color: #0B1F3A;
    }
    .bc-section-hint {
        font-size: 12px; color: #8B95A5; margin-top: -4px; margin-left: 36px;
    }

    /* grouped comparison bars (internal vs external) */
    .bc-group-legend {
        display: flex; gap: 16px; font-size: 11.5px; color: #6B7788; margin-bottom: 12px;
    }
    .bc-group-legend span { display: flex; align-items: center; gap: 5px; }
    .bc-group-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .bc-group-row { margin-bottom: 12px; }
    .bc-group-label {
        font-size: 12.5px; color: #2B2F38; margin-bottom: 4px; font-weight: 500;
    }
    .bc-group-track {
        width: 100%; height: 9px; border-radius: 999px; background: #EDF1F6;
        overflow: hidden; margin-bottom: 3px; position: relative;
    }
    .bc-group-fill { height: 100%; border-radius: 999px; }
    .bc-group-pct {
        font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: #8B95A5;
    }

    /* heatmap grid */
    .bc-heat-grid {
        display: grid; grid-template-columns: 90px repeat(4, 1fr); gap: 5px;
        align-items: center;
    }
    .bc-heat-head {
        font-size: 11px; color: #6B7788; text-align: center; font-weight: 600;
    }
    .bc-heat-row-label {
        font-size: 11.5px; color: #4A5568; font-weight: 500;
    }
    .bc-heat-cell {
        border-radius: 8px; padding: 10px 4px; text-align: center;
        font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600;
    }

    /* dataset donut */
    .bc-donut-wrap { display: flex; align-items: center; gap: 22px; }
    .bc-donut {
        width: 130px; height: 130px; border-radius: 50%; flex-shrink: 0;
    }
    .bc-donut-hole {
        width: 76px; height: 76px; border-radius: 50%; background: #FFFFFF;
        margin: 27px; display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #2B2F38; text-align: center;
    }
    .bc-donut-legend { font-size: 12px; color: #4A5568; }
    .bc-donut-legend-row { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }

    /* ensemble architecture */
    .bc-weight-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .bc-weight-name { font-size: 12.5px; color: #2B2F38; width: 118px; flex-shrink: 0; }
    .bc-weight-track {
        flex: 1; height: 16px; border-radius: 8px; background: #EDF1F6; overflow: hidden;
    }
    .bc-weight-fill {
        height: 100%; border-radius: 8px;
        display: flex; align-items: center; justify-content: flex-end; padding-right: 6px;
    }
    .bc-weight-pct {
        font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: #FFFFFF;
    }
    .bc-sum-box {
        margin-top: 10px; text-align: center; background: #0B1F3A; color: #FFFFFF;
        border-radius: 10px; padding: 10px; font-size: 12.5px; font-weight: 600;
    }

    /* preprocessing workflow */
    .bc-flow-card {
        background: #F7FAFC; border: 1px solid #E3E9F0; border-radius: 12px;
        padding: 12px 14px; text-align: center; height: 100%;
    }
    .bc-flow-model {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 13px;
        color: #0B1F3A; margin-bottom: 6px;
    }
    .bc-flow-detail {
        font-size: 11px; color: #6B7788; line-height: 1.5;
    }
    .bc-flow-mono {
        font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: #12776E;
    }

    /* timeline / progression bars */
    .bc-tl-row { margin-bottom: 11px; }
    .bc-tl-top {
        display: flex; justify-content: space-between; font-size: 12px;
        color: #2B2F38; margin-bottom: 3px;
    }
    .bc-tl-track {
        width: 100%; height: 9px; border-radius: 999px; background: #EDF1F6; overflow: hidden;
    }
    .bc-tl-fill { height: 100%; border-radius: 999px; }

    /* small chips replacing paragraph text */
    .bc-chip-row { display: flex; gap: 10px; flex-wrap: wrap; }
    .bc-chip-small {
        background: #E7F7F4; border: 1px solid #C6EDE6; color: #12776E;
        border-radius: 999px; padding: 7px 14px; font-size: 12px; font-weight: 500;
    }
    .bc-story-card {
        background:#FFFFFF; border:1px solid #E3E9F0; border-radius:14px;
        padding:20px 22px; box-shadow:0 2px 12px rgba(11,31,58,.05); height:100%;
    }
    .bc-story-title {
        font-family:'Space Grotesk',sans-serif; font-size:17px; font-weight:700;
        color:#0B1F3A; margin-bottom:8px;
    }
    .bc-story-text { font-size:13px; color:#4A5568; line-height:1.65; }
    .bc-stage {
        background:#F7FAFC; border-left:5px solid var(--stage); border-radius:10px;
        padding:14px 16px; margin-bottom:10px;
    }
    .bc-stage-label {
        font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600;
        color:var(--stage); text-transform:uppercase; letter-spacing:.5px;
    }
    .bc-stage-title { font-weight:700; color:#0B1F3A; margin:4px 0; }
    .bc-formula {
        background:#0B1F3A; color:#FFFFFF; border-radius:12px; padding:16px 18px;
        font-family:'IBM Plex Mono',monospace; font-size:13px; line-height:1.7;
        text-align:center; margin:12px 0;
    }
    .bc-callout {
        background:#E7F7F4; border:1px solid #C6EDE6; border-radius:12px;
        padding:14px 16px; color:#155E57; font-size:13px; line-height:1.6;
    }
    .bc-warning-note {
        background:#FFF7E8; border:1px solid #F5D999; border-radius:12px;
        padding:14px 16px; color:#6B4E14; font-size:13px; line-height:1.6;
    }
    .bc-compare-chart { background:#FFFFFF; border:1px solid #E3E9F0; border-radius:14px; padding:20px 22px; }
    .bc-compare-row { display:grid; grid-template-columns:145px 1fr 62px; gap:10px; align-items:center; margin:12px 0; }
    .bc-compare-label { font-size:12px; color:#4A5568; }
    .bc-compare-track { height:15px; background:#EDF1F6; border-radius:999px; overflow:hidden; }
    .bc-compare-fill { height:100%; border-radius:999px; }
    .bc-compare-value { font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; text-align:right; }
    .bc-split-flow { display:grid; grid-template-columns:1fr 44px 1.25fr 44px 1fr; gap:8px; align-items:center; }
    .bc-split-node { background:#FFFFFF; border:1px solid #DCE5EE; border-radius:13px; padding:16px; text-align:center; min-height:110px; display:flex; flex-direction:column; justify-content:center; }
    .bc-split-node strong { color:#0B1F3A; font-size:16px; }
    .bc-split-node span { color:#6B7788; font-size:11.5px; line-height:1.5; margin-top:5px; }
    .bc-split-arrow { text-align:center; color:#3E9DE0; font-size:24px; }
    .bc-split-stack { display:grid; gap:8px; }
    .bc-model-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
    .bc-model-tile { background:#FFFFFF; border:1px solid #DCE5EE; border-radius:13px; padding:15px; text-align:center; box-shadow:0 2px 7px rgba(11,31,58,.04); }
    .bc-model-circle { width:40px; height:40px; border-radius:50%; background:#E7F7F4; color:#12776E; display:flex; align-items:center; justify-content:center; margin:0 auto 7px; font-size:19px; }
    .bc-model-name { font-size:12px; font-weight:700; color:#0B1F3A; }
    .bc-model-weight { font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:700; color:#12776E; margin:3px 0; }
    .bc-calc-arrow { text-align:center; color:#3E9DE0; font-size:25px; line-height:1; margin:9px 0; }
    .bc-soft-box { background:#FFFFFF; border:1px solid #DCE5EE; border-radius:12px; padding:14px 16px; text-align:center; }
    .bc-soft-title { font-size:11px; font-weight:700; color:#0B1F3A; margin-bottom:6px; }
    .bc-soft-formula { font-family:'IBM Plex Mono',monospace; font-size:13px; color:#0B1F3A; }
    .bc-class-result { display:grid; grid-template-columns:125px 1fr 60px; gap:10px; align-items:center; margin:10px 0; }
    .bc-class-result-label { font-size:12px; font-weight:600; color:#2B2F38; }
    .bc-result-track { height:12px; border-radius:999px; background:#E9EDF2; overflow:hidden; }
    .bc-result-fill { height:100%; border-radius:999px; }
    .bc-result-value { font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:700; text-align:right; }
    .bc-winner { background:#E7F7F4; border:1px solid #BFE7DF; color:#155E57; border-radius:12px; padding:13px 16px; text-align:center; font-weight:700; }
    @media (max-width: 900px) {
        .bc-split-flow { grid-template-columns:1fr; }
        .bc-split-arrow { transform:rotate(90deg); }
        .bc-model-grid { grid-template-columns:1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown('<div class="bc-side-title">BrainCare</div>', unsafe_allow_html=True)
    st.markdown('<div class="bc-side-sub">AI-Powered MRI Tumor Classification</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🧠 Classifier", "📊 About the Project"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="bc-side-section-title">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="bc-side-text">This dashboard supports physician review of brain '
        'MRI scans by classifying images into four categories and highlighting '
        'candidate regions of interest. It does not replace clinical diagnosis.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="bc-side-section-title">Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="bc-side-stat">Ensemble: EfficientNetV2S + InceptionV3 + EfficientNetB0</div>', unsafe_allow_html=True)
    st.markdown('<div class="bc-side-stat">Internal test accuracy: 86.99%</div>', unsafe_allow_html=True)
    st.markdown('<div class="bc-side-stat">External test accuracy: 84.89%</div>', unsafe_allow_html=True)

    st.markdown('<div class="bc-side-section-title">Classes</div>', unsafe_allow_html=True)
    for cls in ["glioma", "meningioma", "pituitary", "notumor"]:
        st.markdown(
            f'<div class="bc-legend-row"><span class="bc-legend-dot" '
            f'style="background:{CLASS_COLORS[cls]};"></span>{CLASS_LABELS[cls]}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="bc-side-section-title">Note</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="bc-side-text">For clinical decision support only. '
        'All results require physician confirmation.</div>',
        unsafe_allow_html=True
    )

if page == "🧠 Classifier":
    st.markdown(
        """
        <div class="bc-about-hero">
            <div class="bc-about-title">Brain MRI Tumor Classification</div>
            <div class="bc-about-sub">
                Upload a brain MRI image for analysis by the three-model ensemble.
                The system classifies the scan into one of four categories and, when
                applicable, marks up to three candidate regions for physician review.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "Upload an MRI image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()

        with st.spinner("Analyzing..."):
            result = run_pipeline(image_bytes)

        final_class = result["final_class"]
        final_confidence = result["final_confidence"]
        class_probs = result["class_probs"]
        is_no_tumor = final_class == "notumor"

        accent = CLASS_COLORS[final_class]
        pct = final_confidence * 100

        st.markdown(
            f"""
            <div class="bc-hero" style="border-left: 6px solid {accent};">
                <div style="display:flex; align-items:center; gap:16px;">
                    <span class="bc-hero-icon" style="color:{accent};">{CLASS_ICONS[final_class]}</span>
                    <div>
                        <div class="bc-eyebrow">Predicted class</div>
                        <p class="bc-hero-label" style="color:{accent};">{CLASS_LABELS[final_class]}</p>
                        <div class="bc-hero-conf">confidence {pct:.1f}%</div>
                    </div>
                </div>
                <div class="bc-gauge" style="background: conic-gradient({accent} calc({pct}*1%), #EDF1F6 0);">
                    <div class="bc-gauge-inner">
                        <span class="bc-gauge-pct" style="color:{accent};">{pct:.0f}%</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            with st.container(border=True):
                st.markdown('<div class="bc-card-label">Original MRI</div>', unsafe_allow_html=True)
                st.image(result["rgb_image"], width="stretch")

        with col2:
            with st.container(border=True):
                st.markdown('<div class="bc-card-label">Localized MRI</div>', unsafe_allow_html=True)
                display_image = result["boxed_image"] if not is_no_tumor else result["rgb_image"]
                st.image(display_image, width="stretch")

        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)

        prob_col, info_col = st.columns([2, 3], gap="medium")

        with prob_col:
            with st.container(border=True):
                st.markdown('<div class="bc-card-label">Class probabilities</div>', unsafe_allow_html=True)
                for cls, p in sorted(class_probs.items(), key=lambda kv: kv[1], reverse=True):
                    c = CLASS_COLORS[cls]
                    st.markdown(
                        f"""
                        <div class="bc-prob-row">
                            <div class="bc-prob-top">
                                <span>{CLASS_LABELS[cls]}</span>
                                <span>{p * 100:.1f}%</span>
                            </div>
                            <div class="bc-prob-track">
                                <div class="bc-prob-fill" style="width:{p * 100:.1f}%; background:{c};"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with info_col:
            st.markdown(
                f"""
                <div class="bc-card" style="border-left: 5px solid {accent};">
                    <div class="bc-card-label">Clinical explanation</div>
                    <div style="font-size:13.5px; color:#2B2F38; line-height:1.55;">{CLASS_EXPLANATIONS[final_class]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if not is_no_tumor and result["candidates"]:
            st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="bc-card-label">Possible location results</div>', unsafe_allow_html=True)
            cols = st.columns(len(result["candidates"]))
            for i, (col, candidate) in enumerate(zip(cols, result["candidates"])):
                with col:
                    st.markdown(
                        f"""
                        <div class="bc-candidate" style="--c:{BADGE_HEX[i]};">
                            <div class="bc-candidate-swatch" style="background:{BADGE_HEX[i]};"></div>
                            <div>
                                <b>{BADGE_LABELS[i]}</b>
                                <div class="bc-candidate-match">Matches the {BADGE_COLOR_NAMES[i].lower()} box on the image</div>
                                <div class="bc-candidate-score">Suspicion level: {candidate['score'] * 100:.0f}%</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown(
            """
            <div class="bc-disclaimer">
                <span style="font-size:16px;">⚠</span>
                <span>This AI prediction is intended to support clinical decision-making
                and should always be confirmed by a qualified radiologist or physician.</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        with btn_col2:
            if st.button("🔄  Analyze Another MRI"):
                st.session_state.uploader_key += 1
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
    else:
        st.info("Upload an MRI image to begin.")
else:
    # ============================================================
    # ABOUT THE PROJECT PAGE — chart-first, minimal text
    # ============================================================

    st.markdown(
        """
        <div class="bc-about-hero">
            <div class="bc-about-title">Brain MRI Tumor Classification</div>
            <div class="bc-about-sub">
                Three CNNs vote together to classify a brain MRI scan into one of
                four categories, then a rule-based scan marks up to three candidate
                regions (red / orange / yellow) for physician review.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # PROJECT STORY — what the images were, why retraining was
    # necessary, and how data leakage was prevented
    # ============================================================
    st.markdown(
        """
        <div class="bc-section-head">
            <div class="bc-section-num">1</div>
            <div class="bc-section-title-text">What problem does this project solve?</div>
        </div>
        <div class="bc-story-card">
            <div class="bc-story-text">
                The system is a clinical decision-support prototype for classifying a
                <b>brain MRI image</b> into four categories: glioma, meningioma,
                pituitary tumor, or no tumor. Three CNN models examine the same image.
                Their probability scores are combined into one final prediction. If a
                tumor class is predicted, a separate rule-based image-analysis stage
                marks up to three <b>candidate suspicious regions</b> for physician review.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bc-section-head">
            <div class="bc-section-num">2</div>
            <div class="bc-section-title-text">The image journey: before and after retraining</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    before_col, after_col = st.columns(2, gap="medium")
    with before_col:
        st.markdown(
            """
            <div class="bc-story-card">
                <div class="bc-story-title">Before retraining</div>
                <div class="bc-story-text">
                    The first models learned from the prepared and balanced
                    <b>Mendeley T1-weighted contrast-enhanced MRI dataset</b>. Images were
                    organized into four classes, resized to <b>224 × 224</b>, and separated
                    into training, validation, and internal-test sets. On the familiar
                    internal test distribution, the earlier V12 ensemble reached
                    <b>91.67%</b> accuracy.
                    <br><br>
                    It was then tested on a completely untouched external MRI set from a
                    different public source. Accuracy fell to <b>58.50%</b>. This large gap
                    showed that the first ensemble had learned source-specific visual
                    patterns and did not generalize sufficiently to images with different
                    appearance, contrast, or acquisition characteristics.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with after_col:
        st.markdown(
            """
            <div class="bc-story-card">
                <div class="bc-story-title">After controlled retraining</div>
                <div class="bc-story-text">
                    After removing unsuitable external images, <b>650 external MRI images</b>
                    were added only to the training data so the models could learn a wider
                    range of real image appearances. A different group of <b>278 external
                    images</b> remained untouched and was never used for training, model
                    selection, or weight tuning.
                    <br><br>
                    The final models were evaluated on <b>861 internal test images</b> and
                    the <b>278 untouched external test images</b>. The final weighted ensemble
                    achieved <b>86.99% internal accuracy</b> and <b>84.89% external accuracy</b>.
                    The smaller gap is evidence of improved generalization, even though the
                    earlier internal-only score was higher.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="bc-card-label">What changed after retraining?</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="bc-compare-chart">
                <div style="font-size:12px;color:#6B7788;margin-bottom:8px;">
                    The earlier ensemble looked strong on familiar internal images, but its external result exposed a generalization problem.
                </div>
                <div class="bc-compare-row">
                    <div class="bc-compare-label">Before · Internal</div>
                    <div class="bc-compare-track"><div class="bc-compare-fill" style="width:91.67%;background:#3E9DE0;"></div></div>
                    <div class="bc-compare-value" style="color:#2379B5;">91.67%</div>
                </div>
                <div class="bc-compare-row">
                    <div class="bc-compare-label">Before · External</div>
                    <div class="bc-compare-track"><div class="bc-compare-fill" style="width:58.50%;background:#E24B4A;"></div></div>
                    <div class="bc-compare-value" style="color:#C43D3C;">58.50%</div>
                </div>
                <div style="height:1px;background:#E3E9F0;margin:15px 0;"></div>
                <div class="bc-compare-row">
                    <div class="bc-compare-label">Final · Internal</div>
                    <div class="bc-compare-track"><div class="bc-compare-fill" style="width:86.99%;background:#12B5A6;"></div></div>
                    <div class="bc-compare-value" style="color:#0E8F84;">86.99%</div>
                </div>
                <div class="bc-compare-row">
                    <div class="bc-compare-label">Final · External</div>
                    <div class="bc-compare-track"><div class="bc-compare-fill" style="width:84.89%;background:#0E8F84;"></div></div>
                    <div class="bc-compare-value" style="color:#0E8F84;">84.89%</div>
                </div>
                <div class="bc-callout" style="margin-top:14px;">
                    External accuracy improved by <b>26.39 percentage points</b>, and the internal–external gap decreased from <b>33.17</b> to <b>2.10 percentage points</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="bc-card-label">How the images were divided for the final system</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="bc-split-flow">
                <div class="bc-split-node" style="border-top:4px solid #0B1F3A;">
                    <strong>Original prepared data</strong>
                    <span>3,917 training images<br>855 validation images<br>861 internal test images</span>
                </div>
                <div class="bc-split-arrow">＋</div>
                <div class="bc-split-stack">
                    <div class="bc-split-node" style="min-height:72px;border-top:4px solid #12B5A6;">
                        <strong>650 external MRIs</strong>
                        <span>Added only to training for retraining</span>
                    </div>
                    <div class="bc-split-node" style="min-height:72px;border-top:4px solid #FF9F5C;">
                        <strong>278 external MRIs</strong>
                        <span>Kept untouched for the final external test</span>
                    </div>
                </div>
                <div class="bc-split-arrow">→</div>
                <div class="bc-split-stack">
                    <div class="bc-split-node" style="min-height:72px;border-top:4px solid #12776E;">
                        <strong>Training & validation</strong>
                        <span>Used to learn and select the models</span>
                    </div>
                    <div class="bc-split-node" style="min-height:72px;border-top:4px solid #3E9DE0;">
                        <strong>Two final tests</strong>
                        <span>861 internal + 278 untouched external</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bc-callout">
            <b>Why the untouched set matters:</b> an external image cannot be both a
            training example and a fair test example. Keeping the 278-image external test
            set isolated prevents data leakage and provides a more realistic estimate of
            performance on images from outside the original dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bc-section-head">
            <div class="bc-section-num">3</div>
            <div class="bc-section-title-text">How is the final result calculated?</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="bc-story-card" style="background:linear-gradient(180deg,#F2F8FF,#FFFFFF);">
            <div class="bc-story-text" style="margin-bottom:14px;">
                The same MRI is analyzed by all three models. Each model produces four
                probabilities. The system multiplies every probability by that model's
                weight, adds the three contributions class by class, and selects the
                class with the highest combined probability.
            </div>
            <div class="bc-model-grid">
                <div class="bc-model-tile">
                    <div class="bc-model-circle">✣</div>
                    <div class="bc-model-name">EfficientNetV2S</div>
                    <div class="bc-model-weight">5%</div>
                    <div style="font-size:11px;color:#8B95A5;">weight 0.05</div>
                </div>
                <div class="bc-model-tile">
                    <div class="bc-model-circle">✣</div>
                    <div class="bc-model-name">InceptionV3</div>
                    <div class="bc-model-weight">40%</div>
                    <div style="font-size:11px;color:#8B95A5;">weight 0.40</div>
                </div>
                <div class="bc-model-tile">
                    <div class="bc-model-circle">✣</div>
                    <div class="bc-model-name">EfficientNetB0</div>
                    <div class="bc-model-weight">55%</div>
                    <div style="font-size:11px;color:#8B95A5;">weight 0.55</div>
                </div>
            </div>
            <div class="bc-calc-arrow">↓</div>
            <div class="bc-soft-box">
                <div class="bc-soft-title">WEIGHTED SOFT VOTING — calculated separately for every class</div>
                <div class="bc-soft-formula">Final probability = (V2S × 0.05) + (InceptionV3 × 0.40) + (B0 × 0.55)</div>
            </div>
            <div class="bc-calc-arrow">↓</div>
            <div class="bc-soft-box" style="text-align:left;">
                <div class="bc-soft-title" style="text-align:center;">ILLUSTRATIVE COMBINED CLASS PROBABILITIES</div>
                <div class="bc-class-result">
                    <div class="bc-class-result-label">Glioma</div>
                    <div class="bc-result-track"><div class="bc-result-fill" style="width:75%;background:#8B47B8;"></div></div>
                    <div class="bc-result-value" style="color:#8B47B8;">75.00%</div>
                </div>
                <div class="bc-class-result">
                    <div class="bc-class-result-label">Meningioma</div>
                    <div class="bc-result-track"><div class="bc-result-fill" style="width:15.2%;background:#F07A12;"></div></div>
                    <div class="bc-result-value" style="color:#D86508;">15.20%</div>
                </div>
                <div class="bc-class-result">
                    <div class="bc-class-result-label">Pituitary tumor</div>
                    <div class="bc-result-track"><div class="bc-result-fill" style="width:6.3%;background:#2A9D55;"></div></div>
                    <div class="bc-result-value" style="color:#258B4C;">6.30%</div>
                </div>
                <div class="bc-class-result">
                    <div class="bc-class-result-label">No tumor</div>
                    <div class="bc-result-track"><div class="bc-result-fill" style="width:3.5%;background:#8B8F96;"></div></div>
                    <div class="bc-result-value" style="color:#747980;">3.50%</div>
                </div>
            </div>
            <div class="bc-calc-arrow">↓</div>
            <div class="bc-winner">🏆 Highest probability wins → Final predicted class: Glioma, 75%</div>
            <div style="font-size:11px;color:#8B95A5;text-align:center;margin-top:10px;">
                The values above are an explanatory example, not evaluation results from the test dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bc-section-head">
            <div class="bc-section-num">4</div>
            <div class="bc-section-title-text">Classification and localization are separate</div>
        </div>
        <div class="bc-warning-note">
            <b>Classification</b> is produced by the three-model weighted ensemble.
            <b>Localization</b> runs afterward only when a tumor class is predicted. It
            uses grayscale image characteristics to propose up to three candidate regions,
            displayed as red, orange, and yellow boxes. These boxes do not change the
            predicted class or confidence, and they are not confirmed tumor boundaries.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bc-section-head">
            <div class="bc-section-num">5</div>
            <div class="bc-section-title-text">Verified project results and dataset composition</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------- key stats ----------------
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            '<div class="bc-stat"><div class="bc-stat-value">86.99%</div>'
            '<div class="bc-stat-label">Internal test accuracy</div></div>',
            unsafe_allow_html=True
        )
    with s2:
        st.markdown(
            '<div class="bc-stat"><div class="bc-stat-value">84.89%</div>'
            '<div class="bc-stat-label">External test accuracy</div></div>',
            unsafe_allow_html=True
        )
    with s3:
        st.markdown(
            '<div class="bc-stat"><div class="bc-stat-value">6,561</div>'
            '<div class="bc-stat-label">Total MRI images used</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)

    # ================================================================
    # ROW 1 — Dataset distribution (donut) + Test-set composition (bars)
    # ================================================================
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown('<div class="bc-card-label">1 · Dataset distribution</div>', unsafe_allow_html=True)
            total = sum(v for _, v, _ in DATASET_SPLITS)
            gradient_parts = []
            acc = 0.0
            for _, v, color in DATASET_SPLITS:
                start = acc / total * 360
                acc += v
                end = acc / total * 360
                gradient_parts.append(f"{color} {start:.1f}deg {end:.1f}deg")
            gradient = "conic-gradient(" + ", ".join(gradient_parts) + ")"
            legend_html = "".join(
                f'<div class="bc-donut-legend-row">'
                f'<span class="bc-group-dot" style="background:{color};"></span>'
                f'{name}: <b>&nbsp;{v:,}</b></div>'
                for name, v, color in DATASET_SPLITS
            )
            st.markdown(
                f"""
                <div class="bc-donut-wrap">
                    <div class="bc-donut" style="background:{gradient};">
                        <div class="bc-donut-hole">{total:,}<br>images</div>
                    </div>
                    <div class="bc-donut-legend">{legend_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        with st.container(border=True):
            st.markdown('<div class="bc-card-label">2 · Test-set composition by class</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="bc-group-legend">'
                '<span><span class="bc-group-dot" style="background:#3E9DE0;"></span>Internal</span>'
                '<span><span class="bc-group-dot" style="background:#FF9F5C;"></span>External</span>'
                '</div>',
                unsafe_allow_html=True
            )
            classes = ["glioma", "meningioma", "notumor", "pituitary"]
            max_count = max(max(TEST_SET_COUNTS["internal"].values()), max(TEST_SET_COUNTS["external"].values()))
            for cls in classes:
                ic = TEST_SET_COUNTS["internal"][cls]
                ec = TEST_SET_COUNTS["external"][cls]
                st.markdown(
                    f"""
                    <div class="bc-group-row">
                        <div class="bc-group-label">{CLASS_LABELS[cls]}</div>
                        <div class="bc-group-track"><div class="bc-group-fill" style="width:{ic/max_count*100:.1f}%; background:#3E9DE0;"></div></div>
                        <div class="bc-group-pct">{ic} internal</div>
                        <div class="bc-group-track" style="margin-top:4px;"><div class="bc-group-fill" style="width:{ec/max_count*100:.1f}%; background:#FF9F5C;"></div></div>
                        <div class="bc-group-pct">{ec} external</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)

    # ================================================================
    # FINAL ENSEMBLE — Per-class accuracy heatmap
    # ================================================================
    with st.container(border=True):
        st.markdown('<div class="bc-card-label">Per-class accuracy heatmap (final ensemble)</div>', unsafe_allow_html=True)
        classes = ["glioma", "meningioma", "notumor", "pituitary"]

        def heat_color(pct):
            # interpolate light -> teal as accuracy rises
            t = max(0.0, min(1.0, (pct - 60) / 40))
            r = int(230 - t * (230 - 18))
            g = int(238 - t * (238 - 181))
            b = int(240 - t * (240 - 166))
            return f"rgb({r},{g},{b})"

        heat_html = '<div class="bc-heat-grid">'
        heat_html += '<div></div>'
        for cls in classes:
            heat_html += f'<div class="bc-heat-head">{CLASS_LABELS[cls]}</div>'
        heat_html += '<div class="bc-heat-row-label">Internal</div>'
        for cls in classes:
            v = PER_CLASS_ACCURACY["internal"][cls]
            heat_html += f'<div class="bc-heat-cell" style="background:{heat_color(v)};">{v:.0f}%</div>'
        heat_html += '<div class="bc-heat-row-label">External</div>'
        for cls in classes:
            v = PER_CLASS_ACCURACY["external"][cls]
            heat_html += f'<div class="bc-heat-cell" style="background:{heat_color(v)};">{v:.0f}%</div>'
        heat_html += '</div>'
        st.markdown(heat_html, unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:11px; color:#8B95A5; margin-top:10px;">'
            'Darker = higher accuracy. Glioma and meningioma remain the hardest external classes.</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bc-disclaimer">
            <span style="font-size:16px;">⚠</span>
            <span>The colored boxes support physician review and are not confirmed
            tumor boundaries. This tool does not replace clinical diagnosis.</span>
        </div>
        """,
        unsafe_allow_html=True
    )
