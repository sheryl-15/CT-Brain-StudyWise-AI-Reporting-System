import pandas as pd

INPUT_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\verified_studywise_dataset.csv"
OUTPUT_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\clean_studywise_dataset.csv"

df = pd.read_csv(INPUT_CSV)

clean_df = df[df["dicom_count"] > 0].copy()

clean_df.to_csv(OUTPUT_CSV, index=False)

print("Clean dataset created")
print("Total usable studies:", len(clean_df))
print(clean_df["label"].value_counts())
print("Saved at:", OUTPUT_CSV)