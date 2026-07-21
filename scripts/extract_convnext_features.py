import cv2
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.applications import ConvNeXtTiny

CSV_FILES = [
    r"D:\CT_Brain_StudyWise_Project\metadata\train_studies.csv",
    r"D:\CT_Brain_StudyWise_Project\metadata\val_studies.csv",
    r"D:\CT_Brain_StudyWise_Project\metadata\test_studies.csv"
]

FEATURE_DIR = Path(r"D:\CT_Brain_StudyWise_Project\features\convnext_tiny")

IMG_SIZE = 224
WINDOW_CENTER = 40
WINDOW_WIDTH = 80
FEATURE_BATCH_SIZE = 16


def apply_window(img, center, width):
    lower = center - width / 2
    upper = center + width / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return img


def get_sort_key(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        if hasattr(ds, "InstanceNumber"):
            return int(ds.InstanceNumber)
    except Exception:
        pass
    return 0


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


def load_study_images(study_path):
    study_path = Path(study_path)
    dicom_files = list(study_path.rglob("*.dcm"))
    dicom_files = sorted(dicom_files, key=get_sort_key)

    images = []

    for dcm in dicom_files:
        try:
            img = read_dicom_image(dcm)
            images.append(img)
        except Exception:
            continue

    if len(images) == 0:
        return None

    return np.array(images, dtype=np.float32)


def extract_features(images, feature_extractor):
    features = []

    for i in range(0, len(images), FEATURE_BATCH_SIZE):
        batch = images[i:i + FEATURE_BATCH_SIZE]
        feat = feature_extractor(batch, training=False)
        features.append(feat.numpy())

    return np.concatenate(features, axis=0)


def main():
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Building ConvNeXt Tiny feature extractor...")

    feature_extractor = ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    feature_extractor.trainable = False

    all_df = []

    for csv_path in CSV_FILES:
        df = pd.read_csv(csv_path)
        all_df.append(df)

    df = pd.concat(all_df).drop_duplicates(subset=["study_uid"]).reset_index(drop=True)

    print("Total studies for feature extraction:", len(df))

    for idx, row in df.iterrows():
        study_uid = str(row["study_uid"])
        study_path = row["study_path"]

        save_path = FEATURE_DIR / f"{study_uid}.npy"

        if save_path.exists():
            print(f"[{idx+1}/{len(df)}] Skipping existing: {study_uid}")
            continue

        print(f"[{idx+1}/{len(df)}] Extracting: {study_uid}")

        images = load_study_images(study_path)

        if images is None:
            print("No valid DICOM images found. Skipped.")
            continue

        features = extract_features(images, feature_extractor)

        np.save(save_path, features)

        print("Saved:", save_path)
        print("Slices:", len(images), "Feature shape:", features.shape)

    print("\nFeature extraction completed.")
    print("Saved at:", FEATURE_DIR)


if __name__ == "__main__":
    main()