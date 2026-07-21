import pandas as pd

df = pd.read_csv(
    r"D:\CT_Brain_StudyWise_Project\metadata\verified_studywise_dataset.csv"
)

print(df["dicom_count"].describe())