import pandas as pd

INPUT_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\ct_brain_labels.csv"
OUTPUT_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\selected_300_studywise.csv"

df = pd.read_csv(INPUT_CSV)

df = df.drop_duplicates(subset=["study_uid"])

# Keep studies with enough slices
df = df[df["downloaded_slices"] >= 10]

normal_df = df[df["label"] == "NORMAL"].sample(n=150, random_state=42)
abnormal_df = df[df["label"] == "ABNORMAL"].sample(n=150, random_state=42)

selected_df = pd.concat([normal_df, abnormal_df])
selected_df = selected_df.sample(frac=1, random_state=42)

selected_df.to_csv(OUTPUT_CSV, index=False)

print("Selected study-wise dataset created")
print("NORMAL:", len(normal_df))
print("ABNORMAL:", len(abnormal_df))
print("TOTAL:", len(selected_df))
print("\nSaved at:", OUTPUT_CSV)

print("\nSelected Study UIDs:")
print(selected_df[["study_uid", "label", "disease", "downloaded_slices"]])