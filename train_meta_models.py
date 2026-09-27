import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def specificity_score(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specs, total = [], cm.sum()
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))

def evaluate(y_true, y_pred, y_proba, labels, binary):
    avg = "binary" if binary else "macro"
    pos_label = labels[-1] if binary else None
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0) if binary
                     else precision_score(y_true, y_pred, average="macro", zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0) if binary
                       else recall_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": specificity_score(y_true, y_pred, labels),
        "f_score": f1_score(y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0) if binary
                   else f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1]) if binary \
            else roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc"] = float("nan")
    return {k: round(float(v), 4) for k, v in metrics.items()}

def train_dataset_metas(dataset_key):
    print(f"=== Processing {dataset_key} ===")
    meta_path = f"artifacts/{dataset_key}_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    scaler = joblib.load(f"artifacts/{dataset_key}_scaler.joblib")
    base_models = joblib.load(f"artifacts/{dataset_key}_base_models.joblib")
    selected_features = meta["selected_features"]
    labels = meta["labels"]
    binary = meta["binary"]
    n_classes = len(labels)

    if dataset_key == "brfss":
        df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv").drop_duplicates()
        y = df["Diabetes_012"].astype(int)
        X = df.drop(columns=["Diabetes_012"])
    else:
        df = pd.read_csv("diabetic_data.csv").replace("?", np.nan)
        df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"])
        df = df.drop_duplicates()
        y = (df["readmitted"] == "<30").astype(int)
        X = df.drop(columns=["readmitted"])
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype).startswith("str"):
                X[col] = X[col].astype(str)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
        X = X.apply(pd.to_numeric, errors="coerce")

    X = X.fillna(X.mean(numeric_only=True))
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X_selected = X_scaled[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Subsample training set if large to speed up meta training
    train_n = min(30000, len(X_train))
    sample_idx = X_train.sample(train_n, random_state=RANDOM_STATE).index
    X_train_sub = X_train.loc[sample_idx]
    y_train_sub = y_train.loc[sample_idx].values

    test_n = min(20000, len(X_test))
    test_sample_idx = X_test.sample(test_n, random_state=RANDOM_STATE).index
    X_test_sub = X_test.loc[test_sample_idx]
    y_test_sub = y_test.loc[test_sample_idx].values

    print(f"Generating base model predictions for meta training ({train_n} train, {test_n} test)...")
    base_preds_train = []
    base_preds_test = []
    for name, model in base_models.items():
        p_tr = model.predict_proba(X_train_sub)
        p_te = model.predict_proba(X_test_sub)
        base_preds_train.append(p_tr)
        base_preds_test.append(p_te)

    meta_X_train = np.hstack(base_preds_train)
    meta_X_test = np.hstack(base_preds_test)
    meta_dim = meta_X_train.shape[1]

    # Reshaped 3D for LSTM and CNN
    meta_seq_train = meta_X_train.reshape(-1, meta_dim, 1)
    meta_seq_test = meta_X_test.reshape(-1, meta_dim, 1)

    out_units = 1 if binary else n_classes
    out_act = "sigmoid" if binary else "softmax"
    loss = "binary_crossentropy" if binary else "categorical_crossentropy"

    if binary:
        y_train_target = y_train_sub.astype("float32")
    else:
        y_train_target = to_categorical(y_train_sub, num_classes=n_classes)

    # 1. Stack-LSTM
    print("Training Stack-LSTM...")
    lstm_model = Sequential([
        Input(shape=(meta_dim, 1)),
        LSTM(32, activation="relu"),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(out_units, activation=out_act)
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.01), loss=loss, metrics=["accuracy"])
    lstm_model.fit(meta_seq_train, y_train_target, epochs=12, batch_size=64, validation_split=0.15, verbose=0)

    p_lstm = lstm_model.predict(meta_seq_test, verbose=0)
    if binary:
        p_lstm_mat = np.column_stack([1 - p_lstm.reshape(-1), p_lstm.reshape(-1)])
        pred_lstm = (p_lstm_mat[:, 1] > 0.5).astype(int)
    else:
        p_lstm_mat = p_lstm
        pred_lstm = np.argmax(p_lstm_mat, axis=1)

    eval_lstm = evaluate(y_test_sub, pred_lstm, p_lstm_mat, labels, binary)
    print("Stack-LSTM Results:", eval_lstm)

    # 2. Stack-CNN
    print("Training Stack-CNN...")
    cnn_model = Sequential([
        Input(shape=(meta_dim, 1)),
        Conv1D(16, kernel_size=2, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(out_units, activation=out_act)
    ])
    cnn_model.compile(optimizer=Adam(learning_rate=0.01), loss=loss, metrics=["accuracy"])
    cnn_model.fit(meta_seq_train, y_train_target, epochs=12, batch_size=64, validation_split=0.15, verbose=0)

    p_cnn = cnn_model.predict(meta_seq_test, verbose=0)
    if binary:
        p_cnn_mat = np.column_stack([1 - p_cnn.reshape(-1), p_cnn.reshape(-1)])
        pred_cnn = (p_cnn_mat[:, 1] > 0.5).astype(int)
    else:
        p_cnn_mat = p_cnn
        pred_cnn = np.argmax(p_cnn_mat, axis=1)

    eval_cnn = evaluate(y_test_sub, pred_cnn, p_cnn_mat, labels, binary)
    print("Stack-CNN Results:", eval_cnn)

    # 3. Save Keras models
    lstm_model.save(f"artifacts/{dataset_key}_stack_lstm.keras")
    cnn_model.save(f"artifacts/{dataset_key}_stack_cnn.keras")
    # Also save to current directory for convenience
    lstm_model.save(f"{dataset_key}_stack_lstm.keras")
    cnn_model.save(f"{dataset_key}_stack_cnn.keras")

    # Update metadata
    meta["stack_LSTM_result"] = eval_lstm
    meta["stack_CNN_result"] = eval_cnn

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(f"{dataset_key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"=== {dataset_key} finished and saved successfully! ===")

if __name__ == "__main__":
    train_dataset_metas("brfss")
    train_dataset_metas("readmit")
    print("ALL META-MODELS TRAINED AND SAVED!")
