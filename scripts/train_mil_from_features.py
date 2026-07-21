import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime

from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


TRAIN_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\train_studies.csv"
VAL_CSV = r"D:\CT_Brain_StudyWise_Project\metadata\val_studies.csv"

FEATURE_DIR = Path(r"D:\CT_Brain_StudyWise_Project\features\convnext_tiny")

MODEL_PATH = r"D:\CT_Brain_StudyWise_Project\models\mil_from_convnext_features_best.keras"
HISTORY_CSV = r"D:\CT_Brain_StudyWise_Project\results\mil_from_features_history.csv"

EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 8


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


def build_model():
    inp = layers.Input(shape=(None, 768))

    x = AttentionMIL()(inp)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    return Model(inp, out)


def get_feature_path(study_uid):
    return FEATURE_DIR / f"{study_uid}.npy"


def load_features(study_uid):
    path = get_feature_path(study_uid)

    if not path.exists():
        return None

    features = np.load(path).astype(np.float32)

    features = np.expand_dims(features, axis=0)

    return tf.convert_to_tensor(features, dtype=tf.float32)


def evaluate_model(df, model):
    y_true = []
    y_prob = []

    for _, row in df.iterrows():
        study_uid = str(row["study_uid"])
        features = load_features(study_uid)

        if features is None:
            continue

        prob = model(features, training=False).numpy()[0][0]
        label = 1 if row["label"] == "ABNORMAL" else 0

        y_true.append(label)
        y_prob.append(prob)

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    print("=" * 70)
    print("FAST TRAINING: Attention MIL from ConvNeXt Features")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    print("Train studies:", len(train_df))
    print("Validation studies:", len(val_df))

    model = build_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    best_val_auc = 0
    wait = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 70)

        train_df = train_df.sample(frac=1, random_state=epoch).reset_index(drop=True)

        losses = []
        y_true_train = []
        y_prob_train = []

        for step, row in train_df.iterrows():
            study_uid = str(row["study_uid"])
            label_value = 1.0 if row["label"] == "ABNORMAL" else 0.0

            features = load_features(study_uid)

            if features is None:
                continue

            label = tf.convert_to_tensor([[label_value]], dtype=tf.float32)

            with tf.GradientTape() as tape:
                pred = model(features, training=True)
                loss = loss_fn(label, pred)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            prob = float(pred.numpy()[0][0])

            losses.append(float(loss.numpy()))
            y_true_train.append(int(label_value))
            y_prob_train.append(prob)

            if (step + 1) % 25 == 0:
                print(
                    f"Step {step + 1}/{len(train_df)} | "
                    f"Loss: {np.mean(losses):.4f} | "
                    f"Pred: {prob:.4f}"
                )

        y_pred_train = [1 if p >= 0.5 else 0 for p in y_prob_train]

        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_auc = roc_auc_score(y_true_train, y_prob_train)
        train_loss = np.mean(losses)

        val_metrics = evaluate_model(val_df, model)

        print("\nEpoch Result")
        print("-" * 50)
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Train AUC  : {train_auc:.4f}")
        print(f"Val Acc    : {val_metrics['accuracy']:.4f}")
        print(f"Val AUC    : {val_metrics['auc']:.4f}")
        print(f"Val Prec   : {val_metrics['precision']:.4f}")
        print(f"Val Recall : {val_metrics['recall']:.4f}")
        print(f"Val F1     : {val_metrics['f1']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "time": datetime.now()
        })

        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            wait = 0
            model.save(MODEL_PATH)
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
    print("Best model saved at:", MODEL_PATH)
    print("History saved at:", HISTORY_CSV)


if __name__ == "__main__":
    main()