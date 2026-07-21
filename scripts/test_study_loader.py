from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import cv2

CSV_PATH = r"D:\CT_Brain_StudyWise_Project\metadata\train_studies.csv"

MAX_SLICES = 160
IMG_SIZE = 224

WINDOW_CENTER = 40
WINDOW_WIDTH = 80


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

    img = (img * 255).astype(np.uint8)

    img = np.stack([img, img, img], axis=-1)

    return img


def select_slices(dicom_files):
    dicom_files = sorted(dicom_files, key=get_sort_key)

    total = len(dicom_files)

    if total > MAX_SLICES:
        indices = np.linspace(0, total - 1, MAX_SLICES).astype(int)
        selected = [dicom_files[i] for i in indices]
    else:
        selected = dicom_files

    return selected


df = pd.read_csv(CSV_PATH)

sample = df.iloc[0]

study_path = Path(sample["study_path"])
label = sample["label"]
study_uid = sample["study_uid"]

dicom_files = list(study_path.rglob("*.dcm"))

selected_files = select_slices(dicom_files)

images = []

for dcm in selected_files:
    img = read_dicom_image(dcm)
    images.append(img)

images = np.array(images)

print("=" * 60)
print("STUDY LOADER TEST")
print("=" * 60)

print("Study UID:", study_uid)
print("Label:", label)
print("Study path:", study_path)
print("Original DICOM slices:", len(dicom_files))
print("Selected slices:", len(selected_files))
print("Output image shape:", images.shape)
print("Min pixel:", images.min())
print("Max pixel:", images.max())
print("Data type:", images.dtype)

if len(dicom_files) > MAX_SLICES:
    print("Slice selection: sampled 160 slices")
else:
    print("Slice selection: used all available slices")