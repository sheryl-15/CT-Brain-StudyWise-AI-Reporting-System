import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from datetime import datetime
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ConvNeXtTiny
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

TRAIN_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\train_studies.csv"
VAL_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\val_studies.csv"

MODEL_SAVE_PATH = r"D:\CT_Brain_StudyWise_Project\models\convnext_tiny_attention_mil_head_best.keras"
HISTORY_CSV = r"D:\CT_Brain_StudyWise_Project\results\convnext_mil_training_history.csv"

IMG_SIZE = 224
WINDOW_CENTER = 40
WINDOW_WIDTH = 80

EPOCHS = 20
LEARNING_RATE = 1e-4
FEATURE_BATCH_SIZE = 16
PATIENCE = 5


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


class AttentionMIL(layers.Layer):
    def __init__(self, attention_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.dense1 = layers.Dense(attention_dim, activation="tanh")
        self.dense2 = layers.Dense(1)

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


def build_mil_head():
    inp = layers.Input(shape=(None, 768))

    x = AttentionMIL()(inp)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    return Model(inp, out)


def extract_features(images, feature_extractor):
    features = []

    for i in range(0, len(images), FEATURE_BATCH_SIZE):
        batch = images[i:i + FEATURE_BATCH_SIZE]
        feat = feature_extractor(batch, training=False)
        features.append(feat.numpy())

    features = np.concatenate(features, axis=0)
    features = np.expand_dims(features, axis=0)

    return tf.convert_to_tensor(features, dtype=tf.float32)


def evaluate_model(df, feature_extractor, mil_head):
    y_true = []
    y_prob = []

    for _, row in df.iterrows():
        images = load_study_images(row["study_path"])

        if images is None:
            continue

        features = extract_features(images, feature_extractor)
        prob = mil_head(features, training=False).numpy()[0][0]

        label = 1 if row["label"] == "ABNORMAL" else 0

        y_true.append(label)
        y_prob.append(prob)

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return acc, auc, precision, recall, f1


def main():
    print("=" * 70)
    print("VARIABLE-LENGTH STUDY-WISE TRAINING")
    print("ConvNeXt Tiny + Attention MIL")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    print("Train studies:", len(train_df))
    print("Validation studies:", len(val_df))

    print("\nBuilding ConvNeXt Tiny feature extractor...")

    feature_extractor = ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    feature_extractor.trainable = False

    print("Building Attention MIL head...")
    mil_head = build_mil_head()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    best_val_auc = 0
    wait = 0
    history_rows = []

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        print("=" * 70)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 70)

        train_df = train_df.sample(frac=1, random_state=epoch).reset_index(drop=True)

        epoch_losses = []
        y_true_train = []
        y_prob_train = []

        for step, row in train_df.iterrows():
            images = load_study_images(row["study_path"])

            if images is None:
                continue

            label_value = 1.0 if row["label"] == "ABNORMAL" else 0.0
            label = tf.convert_to_tensor([[label_value]], dtype=tf.float32)

            features = extract_features(images, feature_extractor)

            with tf.GradientTape() as tape:
                pred = mil_head(features, training=True)
                loss = loss_fn(label, pred)

            grads = tape.gradient(loss, mil_head.trainable_variables)
            optimizer.apply_gradients(zip(grads, mil_head.trainable_variables))

            prob = float(pred.numpy()[0][0])

            epoch_losses.append(float(loss.numpy()))
            y_true_train.append(int(label_value))
            y_prob_train.append(prob)

            if (step + 1) % 10 == 0:
                print(
                    f"Step {step + 1}/{len(train_df)} | "
                    f"Loss: {np.mean(epoch_losses):.4f} | "
                    f"Last slices: {len(images)} | "
                    f"Pred: {prob:.4f}"
                )

        y_pred_train = [1 if p >= 0.5 else 0 for p in y_prob_train]

        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_auc = roc_auc_score(y_true_train, y_prob_train)
        train_loss = np.mean(epoch_losses)

        print("\nRunning validation...")

        val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate_model(
            val_df,
            feature_extractor,
            mil_head
        )

        print("\nEpoch Result")
        print("-" * 50)
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Train AUC  : {train_auc:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")
        print(f"Val AUC    : {val_auc:.4f}")
        print(f"Val Prec   : {val_precision:.4f}")
        print(f"Val Recall : {val_recall:.4f}")
        print(f"Val F1     : {val_f1:.4f}")

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "time": datetime.now()
        })

        pd.DataFrame(history_rows).to_csv(HISTORY_CSV, index=False)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            wait = 0
            mil_head.save(MODEL_SAVE_PATH)
            print("\nBest model saved.")
            print("Best Val AUC:", best_val_auc)
        else:
            wait += 1
            print("\nVal AUC did not improve.")
            print("Patience:", wait, "/", PATIENCE)

        if wait >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\nTraining completed.")
    print("Best model saved at:")
    print(MODEL_SAVE_PATH)
    print("History saved at:")
    print(HISTORY_CSV)


if __name__ == "__main__":
    main()