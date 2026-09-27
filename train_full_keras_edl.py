"""
Train Full Keras EDL (Ensemble Deep Learning) Pipeline
Matches exact Proposed System Architecture:
1. Diabetes Dataset(s) (BRFSS 2015 & Diabetic Readmission)
2. Data Preprocessing (Clean, Impute, MinMax Scale)
3. Feature Selection (ETC - Extra Trees Classifier)
4. 80% Train / 20% Test Split (Stratified)
5. DL Base-Level Models:
   - ANN (tabular patterns)
   - LSTM (sequential health trends)
   - CNN (local structural patterns)
6. EDL Meta-Level Stacking Models:
   - Stack-ANN
   - Stack-LSTM
   - Stack-CNN
7. Final Diabetes Risk Prediction Evaluation (Acc, Prec, Sens, Spec, F1, MCC, ROC-AUC)
"""
import os
import time
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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

os.makedirs("artifacts", exist_ok=True)

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
        if binary:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc"] = float("nan")
    return {k: round(float(v), 4) for k, v in metrics.items()}

def balance_classes(X, y, random_state=RANDOM_STATE, cap_multiplier=3):
    df = X.copy()
    df["_y"] = y.values
    max_n = df["_y"].value_counts().max()
    parts = []
    for cls, grp in df.groupby("_y"):
        n_target = min(max_n, len(grp) * cap_multiplier) if len(grp) < max_n else len(grp)
        parts.append(resample(grp, replace=True, n_samples=n_target, random_state=random_state))
    out = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out.drop(columns="_y"), out["_y"]

def proba_from_preds(raw_preds, binary):
    if binary:
        p = raw_preds.reshape(-1)
        return np.column_stack([1.0 - p, p])
    return raw_preds

def train_dataset(dataset_key):
    print(f"\n==================================================")
    print(f"       TRAINING EDL PIPELINE FOR: {dataset_key.upper()}")
    print(f"==================================================")

    if dataset_key == "brfss":
        df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv").drop_duplicates()
        y = df["Diabetes_012"].astype(int)
        X = df.drop(columns=["Diabetes_012"])
        class_names = ["No Diabetes", "Prediabetes", "Diabetes"]
        top_k = 15
        train_samples = 30000
    else:
        df = pd.read_csv("diabetic_data.csv").replace("?", np.nan)
        df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"])
        df = df.drop_duplicates()
        y = (df["readmitted"] == "<30").astype(int)
        X = df.drop(columns=["readmitted"])
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype).startswith("str"):
                X[col] = X[col].astype(str)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = dict(zip(le.classes_, [int(x) for x in le.transform(le.classes_)]))
        X = X.apply(pd.to_numeric, errors="coerce")
        joblib.dump(label_encoders, f"artifacts/{dataset_key}_label_encoders.joblib")
        joblib.dump(label_encoders, f"{dataset_key}_label_encoders.joblib")
        class_names = ["Not Readmitted <30d", "Readmitted <30d"]
        top_k = 20
        train_samples = 30000

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    labels = sorted(y.unique().tolist())
    binary = len(labels) == 2
    n_classes = len(labels)

    # 1. Clean & MinMax Normalization
    print("Step 1 & 2: Preprocessing and MinMax Scaling...")
    X = X.fillna(X.mean(numeric_only=True))
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 2. Feature Selection (ETC)
    print(f"Step 3: Feature Selection using Extra Trees Classifier (ETC)...")
    etc = ExtraTreesClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    sample_etc_n = min(30000, len(X_scaled))
    etc.fit(X_scaled.iloc[:sample_etc_n], y.iloc[:sample_etc_n])
    importances = pd.Series(etc.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
    selected_features = importances.head(top_k).index.tolist()
    print("Top features:", selected_features[:8])

    X_selected = X_scaled[selected_features]

    # 3. 80% Train / 20% Test Split
    print("Step 4: 80% Train / 20% Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Balance train set
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)
    if len(X_train_bal) > train_samples:
        idx = np.random.choice(len(X_train_bal), train_samples, replace=False)
        X_train_bal = X_train_bal.iloc[idx].reset_index(drop=True)
        y_train_bal = y_train_bal.iloc[idx].reset_index(drop=True)

    test_eval_n = min(15000, len(X_test))
    X_test_eval = X_test.iloc[:test_eval_n].values
    y_test_eval = y_test.iloc[:test_eval_n].values

    X_tr_np = X_train_bal.values
    y_tr_np = y_train_bal.values

    input_dim = X_tr_np.shape[1]
    out_units = 1 if binary else n_classes
    out_act = "sigmoid" if binary else "softmax"
    loss = "binary_crossentropy" if binary else "categorical_crossentropy"

    if binary:
        y_tr_target = y_tr_np.astype("float32")
    else:
        y_tr_target = to_categorical(y_tr_np, num_classes=n_classes)

    X_tr_seq = X_tr_np.reshape(-1, input_dim, 1)
    X_te_seq = X_test_eval.reshape(-1, input_dim, 1)

    # 4. DL Base-Level Models
    print("\n--- Training DL Base Models ---")
    # Base 1: ANN
    print("1/3 Training Base ANN...")
    ann = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(out_units, activation=out_act)
    ])
    ann.compile(optimizer=Adam(0.001), loss=loss, metrics=["accuracy"])
    ann.fit(X_tr_np, y_tr_target, epochs=4, batch_size=64, validation_split=0.15, verbose=0)
    ann_tr_p = proba_from_preds(ann.predict(X_tr_np, verbose=0), binary)
    ann_te_p = proba_from_preds(ann.predict(X_test_eval, verbose=0), binary)

    # Base 2: LSTM
    print("2/3 Training Base LSTM...")
    lstm = Sequential([
        Input(shape=(input_dim, 1)),
        LSTM(32, activation="relu"),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(out_units, activation=out_act)
    ])
    lstm.compile(optimizer=Adam(0.001), loss=loss, metrics=["accuracy"])
    lstm.fit(X_tr_seq, y_tr_target, epochs=3, batch_size=64, validation_split=0.15, verbose=0)
    lstm_tr_p = proba_from_preds(lstm.predict(X_tr_seq, verbose=0), binary)
    lstm_te_p = proba_from_preds(lstm.predict(X_te_seq, verbose=0), binary)

    # Base 3: CNN
    print("3/3 Training Base CNN...")
    cnn = Sequential([
        Input(shape=(input_dim, 1)),
        Conv1D(32, kernel_size=2, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(out_units, activation=out_act)
    ])
    cnn.compile(optimizer=Adam(0.001), loss=loss, metrics=["accuracy"])
    cnn.fit(X_tr_seq, y_tr_target, epochs=3, batch_size=64, validation_split=0.15, verbose=0)
    cnn_tr_p = proba_from_preds(cnn.predict(X_tr_seq, verbose=0), binary)
    cnn_te_p = proba_from_preds(cnn.predict(X_te_seq, verbose=0), binary)

    # Evaluate Base Models
    pred_ann = np.argmax(ann_te_p, axis=1) if not binary else (ann_te_p[:, 1] > 0.5).astype(int)
    pred_lstm = np.argmax(lstm_te_p, axis=1) if not binary else (lstm_te_p[:, 1] > 0.5).astype(int)
    pred_cnn = np.argmax(cnn_te_p, axis=1) if not binary else (cnn_te_p[:, 1] > 0.5).astype(int)

    base_results = {
        "ANN": evaluate(y_test_eval, pred_ann, ann_te_p, labels, binary),
        "LSTM": evaluate(y_test_eval, pred_lstm, lstm_te_p, labels, binary),
        "CNN": evaluate(y_test_eval, pred_cnn, cnn_te_p, labels, binary)
    }
    print("Base Models Results:", base_results)

    # 5. EDL Meta-Level Stacking Models
    print("\n--- Training EDL Meta-Level Stacking Models ---")
    meta_tr = np.hstack([ann_tr_p, lstm_tr_p, cnn_tr_p])
    meta_te = np.hstack([ann_te_p, lstm_te_p, cnn_te_p])
    meta_dim = meta_tr.shape[1]
    meta_tr_seq = meta_tr.reshape(-1, meta_dim, 1)
    meta_te_seq = meta_te.reshape(-1, meta_dim, 1)

    # Meta 1: Stack-ANN
    print("1/3 Training Stack-ANN...")
    stack_ann = Sequential([
        Input(shape=(meta_dim,)),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(out_units, activation=out_act)
    ])
    stack_ann.compile(optimizer=Adam(0.005), loss=loss, metrics=["accuracy"])
    stack_ann.fit(meta_tr, y_tr_target, epochs=8, batch_size=64, validation_split=0.15, verbose=0)
    stack_ann_p = proba_from_preds(stack_ann.predict(meta_te, verbose=0), binary)
    pred_stack_ann = np.argmax(stack_ann_p, axis=1) if not binary else (stack_ann_p[:, 1] > 0.5).astype(int)
    eval_stack_ann = evaluate(y_test_eval, pred_stack_ann, stack_ann_p, labels, binary)

    # Meta 2: Stack-LSTM
    print("2/3 Training Stack-LSTM...")
    stack_lstm = Sequential([
        Input(shape=(meta_dim, 1)),
        LSTM(24, activation="relu"),
        Dense(16, activation="relu"),
        Dense(out_units, activation=out_act)
    ])
    stack_lstm.compile(optimizer=Adam(0.005), loss=loss, metrics=["accuracy"])
    stack_lstm.fit(meta_tr_seq, y_tr_target, epochs=8, batch_size=64, validation_split=0.15, verbose=0)
    stack_lstm_p = proba_from_preds(stack_lstm.predict(meta_te_seq, verbose=0), binary)
    pred_stack_lstm = np.argmax(stack_lstm_p, axis=1) if not binary else (stack_lstm_p[:, 1] > 0.5).astype(int)
    eval_stack_lstm = evaluate(y_test_eval, pred_stack_lstm, stack_lstm_p, labels, binary)

    # Meta 3: Stack-CNN
    print("3/3 Training Stack-CNN...")
    stack_cnn = Sequential([
        Input(shape=(meta_dim, 1)),
        Conv1D(16, kernel_size=2, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation="relu"),
        Dense(out_units, activation=out_act)
    ])
    stack_cnn.compile(optimizer=Adam(0.005), loss=loss, metrics=["accuracy"])
    stack_cnn.fit(meta_tr_seq, y_tr_target, epochs=8, batch_size=64, validation_split=0.15, verbose=0)
    stack_cnn_p = proba_from_preds(stack_cnn.predict(meta_te_seq, verbose=0), binary)
    pred_stack_cnn = np.argmax(stack_cnn_p, axis=1) if not binary else (stack_cnn_p[:, 1] > 0.5).astype(int)
    eval_stack_cnn = evaluate(y_test_eval, pred_stack_cnn, stack_cnn_p, labels, binary)

    # Consensus Meta Ensemble (average of Stack-ANN, Stack-LSTM, Stack-CNN)
    consensus_p = (stack_ann_p + stack_lstm_p + stack_cnn_p) / 3.0
    pred_consensus = np.argmax(consensus_p, axis=1) if not binary else (consensus_p[:, 1] > 0.5).astype(int)
    eval_consensus = evaluate(y_test_eval, pred_consensus, consensus_p, labels, binary)

    print("Stack-ANN:", eval_stack_ann)
    print("Stack-LSTM:", eval_stack_lstm)
    print("Stack-CNN:", eval_stack_cnn)
    print("Meta Consensus:", eval_consensus)

    # 6. Save Keras Models & Artifacts
    print("Saving models and metadata...")
    models_dict = {
        "ann": ann, "lstm": lstm, "cnn": cnn,
        "stack_ann": stack_ann, "stack_lstm": stack_lstm, "stack_cnn": stack_cnn
    }
    for m_name, m_obj in models_dict.items():
        m_obj.save(f"artifacts/{dataset_key}_{m_name}.keras")
        m_obj.save(f"{dataset_key}_{m_name}.keras")

    joblib.dump(scaler, f"artifacts/{dataset_key}_scaler.joblib")
    joblib.dump(scaler, f"{dataset_key}_scaler.joblib")

    meta = {
        "dataset_key": dataset_key,
        "class_names": class_names,
        "labels": labels,
        "binary": binary,
        "selected_features": selected_features,
        "all_feature_columns": X.columns.tolist(),
        "feature_importances": importances.round(4).to_dict(),
        "base_model_results": base_results,
        "stack_ANN_result": eval_stack_ann,
        "stack_LSTM_result": eval_stack_lstm,
        "stack_CNN_result": eval_stack_cnn,
        "stack_consensus_result": eval_consensus,
        "feature_ranges": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in selected_features},
        "all_feature_medians": {c: float(X[c].median()) for c in X.columns},
    }

    with open(f"artifacts/{dataset_key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(f"{dataset_key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset {dataset_key} EDL pipeline completely finished and saved!")
    return meta

if __name__ == "__main__":
    t_start = time.time()
    train_dataset("brfss")
    train_dataset("readmit")
    print(f"\nALL EDL PIPELINE TRAINING COMPLETED IN {round(time.time() - t_start, 2)} SECONDS!")
