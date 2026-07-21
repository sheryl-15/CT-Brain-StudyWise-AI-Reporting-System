import re
import cv2
import pydicom
import numpy as np
import tensorflow as tf

from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dicomweb_client.api import DICOMwebClient
from tensorflow.keras.applications import ConvNeXtTiny


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_PREVIEW_DIR = BASE_DIR / "static" / "previews"

PACS_URL = "http://pacs.uniquewellness.co.in:8080/dcm4chee-arc/aets/DCM4CHEE/rs"
client = DICOMwebClient(url=PACS_URL, timeout=120)

DOWNLOAD_ROOT = BASE_DIR / "external_analysis"
MODEL_PATH = BASE_DIR / "models" / "mil_from_convnext_features_best.keras"

IMG_SIZE = 224
WINDOW_CENTER = 40
WINDOW_WIDTH = 80
FEATURE_BATCH_SIZE = 16
THRESHOLD = 0.15


class AttentionMIL(tf.keras.layers.Layer):
    def __init__(self, attention_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.dense1 = tf.keras.layers.Dense(attention_dim, activation="tanh")
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, features):
        attention = self.dense1(features)
        attention = self.dense2(attention)
        attention = tf.nn.softmax(attention, axis=1)
        weighted = features * attention
        return tf.reduce_sum(weighted, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"attention_dim": self.attention_dim})
        return config


print("Loading ConvNeXt feature extractor...")
feature_extractor = ConvNeXtTiny(
    include_top=False,
    weights="imagenet",
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
feature_extractor.trainable = False

print("Loading MIL model...")
model = tf.keras.models.load_model(
    str(MODEL_PATH),
    custom_objects={"AttentionMIL": AttentionMIL}
)
print("AI models loaded successfully.")


def safe_name(value):
    return re.sub(r'[<>:"/\\|?*]', "_", str(value))


def get_value(item, tag, default="--"):
    try:
        value = item.get(tag, {}).get("Value", [default])
        if not value:
            return default

        val = value[0]

        if isinstance(val, dict):
            return val.get("Alphabetic", default)

        return str(val)

    except Exception:
        return default


def clean(value):
    value = str(value).strip()
    return value if value else "--"


def extract_metadata_from_dataset(ds):
    return {
        "patient_name": clean(getattr(ds, "PatientName", "--")),
        "patient_id": clean(getattr(ds, "PatientID", "--")),
        "patient_age": clean(getattr(ds, "PatientAge", "--")),
        "patient_gender": clean(getattr(ds, "PatientSex", "--")),
        "study_date": clean(getattr(ds, "StudyDate", "--")),
        "study_description": clean(getattr(ds, "StudyDescription", "--")),
        "institution": clean(getattr(ds, "InstitutionName", "--")),
        "doctor": clean(getattr(ds, "ReferringPhysicianName", "--")),
    }


def get_metadata_from_pacs(study_uid, series_uid):
    try:
        print("Fetching instance metadata for Study Details...")

        instances = client.search_for_instances(
            study_instance_uid=study_uid,
            series_instance_uid=series_uid
        )

        if not instances:
            print("No instances found for metadata.")
            return {}

        first_instance = instances[0]
        sop_uid = get_value(first_instance, "00080018")

        if sop_uid == "--":
            print("SOP UID not found.")
            return {}

        ds = client.retrieve_instance(
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            sop_instance_uid=sop_uid
        )

        metadata = extract_metadata_from_dataset(ds)
        print("Metadata loaded:", metadata)
        return metadata

    except Exception as e:
        print("Metadata fetch failed:", str(e))
        return {}


def apply_window(img, center, width):
    lower = center - width / 2
    upper = center + width / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return img


def get_sort_key_from_dataset(ds):
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        try:
            return int(ds.InstanceNumber)
        except Exception:
            return 0


def get_sort_key_from_file(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        return get_sort_key_from_dataset(ds)
    except Exception:
        return 0


def save_series(datasets, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    datasets = sorted(datasets, key=get_sort_key_from_dataset)

    for idx, ds in enumerate(datasets, start=1):
        sop = str(getattr(ds, "SOPInstanceUID", f"IMG_{idx}"))
        file_path = output_folder / f"{idx:04d}_{safe_name(sop)}.dcm"
        ds.save_as(str(file_path), write_like_original=False)

    return len(datasets)


def read_dicom_image(dicom_path):
    ds = pydicom.dcmread(dicom_path)

    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    img = img * slope + intercept

    img = apply_window(img, WINDOW_CENTER, WINDOW_WIDTH)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img * 255.0
    img = np.stack([img, img, img], axis=-1)

    return img.astype(np.float32)


def extract_features_from_folder(series_folder):
    dcm_files = list(Path(series_folder).rglob("*.dcm"))

    if len(dcm_files) == 0:
        raise ValueError("No DICOM files found in selected series folder.")

    dcm_files = sorted(dcm_files, key=get_sort_key_from_file)

    images = []

    for dcm in dcm_files:
        try:
            images.append(read_dicom_image(dcm))
        except Exception as e:
            print("Skipped:", dcm.name, e)

    if len(images) == 0:
        raise ValueError("No valid DICOM images found.")

    images = np.array(images, dtype=np.float32)

    features = []

    for i in range(0, len(images), FEATURE_BATCH_SIZE):
        batch = images[i:i + FEATURE_BATCH_SIZE]
        feat = feature_extractor(batch, training=False)
        features.append(feat.numpy())

    features = np.concatenate(features, axis=0)
    features = np.expand_dims(features, axis=0)

    return tf.convert_to_tensor(features, dtype=tf.float32), len(images)


def extract_dicom_metadata(dicom_folder):
    dcm_files = list(Path(dicom_folder).rglob("*.dcm"))

    if not dcm_files:
        return {}

    ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    return extract_metadata_from_dataset(ds)


def save_preview_slices(dicom_folder, study_uid, series_uid):
    STATIC_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    dcm_files = list(Path(dicom_folder).rglob("*.dcm"))
    dcm_files = sorted(dcm_files, key=get_sort_key_from_file)

    if len(dcm_files) == 0:
        return []

    if len(dcm_files) < 3:
        selected_files = dcm_files
    else:
        selected_files = [
            dcm_files[len(dcm_files) // 4],
            dcm_files[len(dcm_files) // 2],
            dcm_files[(len(dcm_files) * 3) // 4],
        ]

    preview_urls = []

    for idx, dcm_path in enumerate(selected_files, start=1):
        img = read_dicom_image(dcm_path)
        gray = img[:, :, 0]

        filename = f"{safe_name(study_uid)}_{safe_name(series_uid)}_preview_{idx}.png"
        save_path = STATIC_PREVIEW_DIR / filename

        success = cv2.imwrite(str(save_path), gray)

        if success:
            preview_urls.append(f"/static/previews/{filename}")
        else:
            print("Failed to save preview:", save_path)

    return preview_urls


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/load-study", methods=["POST"])
def load_study():
    data = request.get_json()
    study_uid = data.get("study_uid", "").strip()

    print("Received Study UID:", study_uid)

    if not study_uid:
        return jsonify({"success": False, "error": "Study UID is required"})

    try:
        print("Searching series in PACS...")

        series_list = client.search_for_series(study_instance_uid=study_uid)

        print("Series found:", len(series_list))

        if not series_list:
            return jsonify({"success": False, "error": "No series found for this Study UID"})

        series_data = []

        for s in series_list:
            series_data.append({
                "series_uid": get_value(s, "0020000E"),
                "description": get_value(s, "0008103E", "No Description"),
                "modality": get_value(s, "00080060"),
                "series_number": get_value(s, "00200011"),
                "instances": get_value(s, "00201209"),
            })

        first_series_uid = series_data[0]["series_uid"]
        metadata = get_metadata_from_pacs(study_uid, first_series_uid)

        patient_data = {
            "name": metadata.get("patient_name", "--"),
            "id": metadata.get("patient_id", "--"),
            "age": metadata.get("patient_age", "--"),
            "gender": metadata.get("patient_gender", "--"),
            "study_date": metadata.get("study_date", "--"),
        }

        study_data = {
            "study_uid": study_uid,
            "description": metadata.get("study_description", "--"),
            "institution": metadata.get("institution", "--"),
            "doctor": metadata.get("doctor", "--"),
            "total_series": len(series_data),
        }

        print("Study Details loaded after Study UID search.")

        return jsonify({
            "success": True,
            "patient": patient_data,
            "study": study_data,
            "series": series_data
        })

    except Exception as e:
        print("ERROR in load-study:", str(e))
        return jsonify({"success": False, "error": str(e)})


@app.route("/analyze-series", methods=["POST"])
def analyze_series():
    data = request.get_json()

    study_uid = data.get("study_uid", "").strip()
    series_uid = data.get("series_uid", "").strip()
    series_description = data.get("series_description", "--")

    if not study_uid or not series_uid:
        return jsonify({"success": False, "error": "Study UID and Series UID are required"})

    try:
        print("Downloading selected series...")
        print("Study UID:", study_uid)
        print("Series UID:", series_uid)

        datasets = client.retrieve_series(
            study_instance_uid=study_uid,
            series_instance_uid=series_uid
        )

        output_folder = DOWNLOAD_ROOT / safe_name(study_uid) / safe_name(series_uid)

        slice_count = save_series(datasets, output_folder)
        print("Downloaded slices:", slice_count)

        metadata = extract_dicom_metadata(output_folder)
        preview_urls = save_preview_slices(output_folder, study_uid, series_uid)

        features, valid_slice_count = extract_features_from_folder(output_folder)

        print("Features extracted. Predicting...")

        prob = float(model(features, training=False).numpy()[0][0])
        prediction = "ABNORMAL" if prob >= THRESHOLD else "NORMAL"

        if prob >= 0.75:
            risk = "HIGH"
        elif prob >= 0.40:
            risk = "MODERATE"
        else:
            risk = "LOW"

        response = {
            "success": True,
            "prediction": prediction,
            "probability": round(prob * 100, 2),
            "risk": risk,
            "slice_count": valid_slice_count,
            "series_uid": series_uid,
            "series_description": series_description,
            "threshold": THRESHOLD,
            "metadata": metadata,
            "preview_urls": preview_urls,
            "generated_time": datetime.now().strftime("%d-%m-%Y %I:%M %p")
        }

        print("Prediction response:", response)
        return jsonify(response)

    except Exception as e:
        print("ERROR in analyze-series:", str(e))
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)