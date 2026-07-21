import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

TEST_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\test_studies.csv"
FEATURE_DIR = Path(r"D:\CT_Brain_StudyWise_Project\features\convnext_tiny")
MODEL_PATH = r"D:\CT_Brain_StudyWise_Project\models\regularized_mil_best.keras"
OUTPUT_CSV = r"D:\CT_Brain_StudyWise_Project\results\test_predictions.csv"


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
        bag = tf.reduce_sum(weighted, axis=1)

        return bag

    def get_config(self):
        config = super().get_config()
        config.update({"attention_dim": self.attention_dim})
        return config


def load_features(study_uid):
    path = FEATURE_DIR / f"{study_uid}.npy"

    if not path.exists():
        return None

    features = np.load(path).astype(np.float32)
    features = np.expand_dims(features, axis=0)

    return tf.convert_to_tensor(features, dtype=tf.float32)


print("Loading test data...")
test_df = pd.read_csv(TEST_CSV)

print("Loading best model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"AttentionMIL": AttentionMIL}
)

y_true = []
y_prob = []
rows = []

for _, row in test_df.iterrows():
    study_uid = str(row["study_uid"])
    actual_label = row["label"]

    features = load_features(study_uid)

    if features is None:
        print("Missing features:", study_uid)
        continue

    prob = model(features, training=False).numpy()[0][0]

    predicted_label = "ABNORMAL" if prob >= 0.5 else "NORMAL"

    y_true_value = 1 if actual_label == "ABNORMAL" else 0

    y_true.append(y_true_value)
    y_prob.append(prob)

    rows.append({
        "study_uid": study_uid,
        "actual_label": actual_label,
        "abnormal_probability": prob,
        "predicted_label": predicted_label,
        "correct": actual_label == predicted_label
    })

y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_CSV, index=False)

print("\nTEST SET EVALUATION")
print("=" * 60)

print("Total test studies:", len(y_true))
print("Accuracy :", accuracy)
print("AUC      :", auc)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["NORMAL", "ABNORMAL"],
    zero_division=0
))

print("\nSaved predictions at:")
print(OUTPUT_CSV)