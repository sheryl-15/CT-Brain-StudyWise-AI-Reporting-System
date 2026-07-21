import re
import cv2
import pydicom
import numpy as np
import tensorflow as tf
from pathlib import Path
from dicomweb_client.api import DICOMwebClient
from tensorflow.keras.applications import ConvNeXtTiny


PACS_URL = "http://pacs.uniquewellness.co.in:8080/dcm4chee-arc/aets/DCM4CHEE/rs"

DOWNLOAD_ROOT = Path(r"D:\CT_Brain_StudyWise_Project\external_analysis")
MODEL_PATH = r"D:\CT_Brain_StudyWise_Project\models\mil_from_convnext_features_best.keras"

IMG_SIZE = 224
WINDOW_CENTER = 40
WINDOW_WIDTH = 80
FEATURE_BATCH_SIZE = 16
THRESHOLD = 0.15   # best threshold from your test analysis


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


def safe_name(value):
    return re.sub(r'[<>:"/\\|?*]', "_", str(value))


def apply_window(img, center, width):
    lower = center - width / 2
    upper = center + width / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return img


def get_sort_key(ds):
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        try:
            return int(ds.InstanceNumber)
        except Exception:
            return 0


def save_series(datasets, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    datasets = sorted(datasets, key=get_sort_key)

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


def extract_features_from_folder(series_folder, feature_extractor):
    dcm_files = list(Path(series_folder).rglob("*.dcm"))

    if len(dcm_files) == 0:
        raise ValueError("No DICOM files found in selected series folder.")

    dcm_files = sorted(
        dcm_files,
        key=lambda p: get_sort_key(pydicom.dcmread(p, stop_before_pixels=True))
    )

    images = []

    for dcm in dcm_files:
        try:
            images.append(read_dicom_image(dcm))
        except Exception as e:
            print("Skipped:", dcm.name, e)

    images = np.array(images, dtype=np.float32)

    features = []

    for i in range(0, len(images), FEATURE_BATCH_SIZE):
        batch = images[i:i + FEATURE_BATCH_SIZE]
        feat = feature_extractor(batch, training=False)
        features.append(feat.numpy())

    features = np.concatenate(features, axis=0)
    features = np.expand_dims(features, axis=0)

    return tf.convert_to_tensor(features, dtype=tf.float32), len(images)


def main():
    study_uid = input("Enter Study UID: ").strip()

    client = DICOMwebClient(url=PACS_URL)

    print("\nSearching series in study...")
    series_list = client.search_for_series(study_instance_uid=study_uid)

    if len(series_list) == 0:
        print("No series found for this study UID.")
        return

    print("\nAvailable Series:")
    print("=" * 80)

    for i, series in enumerate(series_list, start=1):
        series_uid = series.get("0020000E", {}).get("Value", ["UNKNOWN"])[0]
        desc = series.get("0008103E", {}).get("Value", ["No Description"])[0]
        modality = series.get("00080060", {}).get("Value", ["UNKNOWN"])[0]
        count = series.get("00201209", {}).get("Value", ["UNKNOWN"])[0]

        print(f"{i}. Series UID : {series_uid}")
        print(f"   Description: {desc}")
        print(f"   Modality   : {modality}")
        print(f"   Instances  : {count}")
        print("-" * 80)

    choice = int(input("\nSelect series number for prediction: "))

    selected_series = series_list[choice - 1]
    series_uid = selected_series.get("0020000E", {}).get("Value", ["UNKNOWN"])[0]

    print("\nDownloading selected series...")
    datasets = client.retrieve_series(
        study_instance_uid=study_uid,
        series_instance_uid=series_uid
    )

    output_folder = DOWNLOAD_ROOT / safe_name(study_uid) / safe_name(series_uid)
    count = save_series(datasets, output_folder)

    print("Downloaded DICOM files:", count)
    print("Saved at:", output_folder)

    print("\nLoading ConvNeXt feature extractor...")
    feature_extractor = ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    feature_extractor.trainable = False

    print("Extracting features...")
    features, slice_count = extract_features_from_folder(output_folder, feature_extractor)

    print("Loading MIL model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionMIL": AttentionMIL}
    )

    prob = model(features, training=False).numpy()[0][0]

    prediction = "ABNORMAL" if prob >= THRESHOLD else "NORMAL"

    print("\nSTUDY PREDICTION RESULT")
    print("=" * 60)
    print("Study UID:", study_uid)
    print("Series UID:", series_uid)
    print("Slices analyzed:", slice_count)
    print("Abnormal probability:", prob)
    print("Threshold:", THRESHOLD)
    print("Prediction:", prediction)


if __name__ == "__main__":
    main()