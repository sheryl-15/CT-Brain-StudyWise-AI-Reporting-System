from pathlib import Path
import pandas as pd

DATASET_ROOT = Path(r"D:\CT_Brain_StudyWise_Project\dataset_raw")

classes = ["NORMAL", "ABNORMAL"]

rows = []

for label in classes:
    class_dir = DATASET_ROOT / label

    for study_dir in class_dir.iterdir():
        if not study_dir.is_dir():
            continue

        series_dirs = [p for p in study_dir.iterdir() if p.is_dir()]
        dcm_files = list(study_dir.rglob("*.dcm"))

        rows.append({
            "study_uid": study_dir.name,
            "label": label,
            "series_count": len(series_dirs),
            "dicom_count": len(dcm_files),
            "study_path": str(study_dir)
        })

df = pd.DataFrame(rows)

output_csv = r"D:\CT_Brain_StudyWise_Project\metadata\verified_studywise_dataset.csv"
df.to_csv(output_csv, index=False)

print("=" * 60)
print("STUDY-WISE DATASET VERIFICATION")
print("=" * 60)

print("Total studies:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nDICOM count summary:")
print(df.groupby("label")["dicom_count"].describe())

print("\nStudies with 0 DICOM files:")
print(df[df["dicom_count"] == 0])

print("\nSaved at:")
print(output_csv)