import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PRED_CSV = r"D:\CT_Brain_StudyWise_Project\results\test_predictions.csv"

df = pd.read_csv(PRED_CSV)

df["actual_value"] = df["actual_label"].apply(lambda x: 1 if x == "ABNORMAL" else 0)

thresholds = [i / 100 for i in range(10, 91, 5)]

best_f1 = 0
best_threshold = 0

print("THRESHOLD ANALYSIS ON TEST SET")
print("=" * 70)
print("Threshold | Accuracy | Precision | Recall | F1")
print("-" * 70)

for threshold in thresholds:
    df["pred_value"] = df["abnormal_probability"].apply(lambda x: 1 if x >= threshold else 0)

    acc = accuracy_score(df["actual_value"], df["pred_value"])
    prec = precision_score(df["actual_value"], df["pred_value"], zero_division=0)
    rec = recall_score(df["actual_value"], df["pred_value"], zero_division=0)
    f1 = f1_score(df["actual_value"], df["pred_value"], zero_division=0)

    print(f"{threshold:9.2f} | {acc:8.4f} | {prec:9.4f} | {rec:6.4f} | {f1:6.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest threshold based on F1:", best_threshold)
print("Best F1:", best_f1)

df["best_pred_value"] = df["abnormal_probability"].apply(lambda x: 1 if x >= best_threshold else 0)
df["best_prediction"] = df["best_pred_value"].apply(lambda x: "ABNORMAL" if x == 1 else "NORMAL")

cm = confusion_matrix(df["actual_value"], df["best_pred_value"])

print("\nConfusion Matrix at Best Threshold:")
print(cm)

print("\nPrediction counts:")
print(df["best_prediction"].value_counts())