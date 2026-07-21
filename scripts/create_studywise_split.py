import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\clean_studywise_dataset.csv"

TRAIN_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\train_studies.csv"
VAL_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\val_studies.csv"
TEST_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\test_studies.csv"

df = pd.read_csv(INPUT_CSV)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("Study-wise split completed.")
print("\nTrain:")
print(train_df["label"].value_counts())

print("\nValidation:")
print(val_df["label"].value_counts())

print("\nTest:")
print(test_df["label"].value_counts())

print("\nSaved:")
print(TRAIN_CSV)
print(VAL_CSV)
print(TEST_CSV)