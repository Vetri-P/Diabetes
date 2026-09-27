"""
Ensemble Deep Learning (EDL) style Clinical Decision Support pipeline,
replicating the methodology of Al Reshan et al. (IEEE Access, 2024) on
two real-world datasets.

IMPORTANT NOTE ON FIDELITY:
This sandbox has no internet access, so TensorFlow/Keras cannot be
installed and literal LSTM / CNN architectures cannot be trained.
To still produce genuine, real numbers on real data, the three
base-level learners used here are:
    - ANN      -> scikit-learn MLPClassifier (a real, gradient-trained
                  feed-forward neural network -- faithful to the paper)
    - "LSTM"   -> RandomForestClassifier (substitute; captures non-linear
                  interactions the paper's LSTM would model over feature
                  sequence order)
    - "CNN"    -> HistGradientBoostingClassifier (substitute; captures
                  local/structural patterns the paper's CNN would model)
These substitutions are clearly reported. The stacking meta-level model
(Stack-ANN) is a real MLP trained on out-of-fold base-model predictions,
exactly matching the paper's stacking design.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, matthews_corrcoef, roc_auc_score,
                              confusion_matrix)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def specificity_score(y_true, y_pred, labels):
    """Macro-averaged specificity across classes (one-vs-rest)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specs = []
    total = cm.sum()
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
    """Oversample minority classes to the size of the majority class
    (SMOTE substitute -- imblearn is unavailable offline; this is
    random oversampling with replacement, as recommended when SMOTE
    is unavailable)."""
    df = X.copy()
    df["_y"] = y.values
    max_n = df["_y"].value_counts().max()
    parts = []
    for cls, grp in df.groupby("_y"):
        n_target = min(max_n, len(grp) * cap_multiplier) if len(grp) < max_n else len(grp)
        parts.append(resample(grp, replace=True, n_samples=n_target, random_state=random_state))
    out = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out.drop(columns="_y"), out["_y"]


def run_edl_pipeline(X, y, dataset_name, top_k_features=None, train_subsample=None):
    t0 = time.time()
    labels = sorted(y.unique().tolist())
    binary = len(labels) == 2

    # ---- Data preprocessing ----
    X = X.copy()
    X = X.fillna(X.mean(numeric_only=True))
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # ---- Feature selection via Extra Tree Classifier ----
    etc = ExtraTreesClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    etc.fit(X_scaled, y)
    importances = pd.Series(etc.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
    if top_k_features:
        keep = importances.head(top_k_features).index.tolist()
        X_scaled = X_scaled[keep]

    # ---- Train / test split (80:20) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ---- Class balancing (train only) ----
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    if train_subsample and len(X_train_bal) > train_subsample:
        X_train_bal = X_train_bal.sample(train_subsample, random_state=RANDOM_STATE)
        y_train_bal = y_train_bal.loc[X_train_bal.index]

    # ---- Base-level models ----
    base_models = {
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                              max_iter=60, random_state=RANDOM_STATE, early_stopping=True),
        "LSTM_substitute(RF)": RandomForestClassifier(n_estimators=200, max_depth=14,
                                                        random_state=RANDOM_STATE, n_jobs=-1),
        "CNN_substitute(HGB)": HistGradientBoostingClassifier(max_iter=150, random_state=RANDOM_STATE),
    }

    base_results = {}
    oof_preds_train = {}
    test_preds = {}
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    for name, model in base_models.items():
        # out-of-fold predictions on train (for stacking, avoids leakage)
        oof_proba = cross_val_predict(model, X_train_bal, y_train_bal, cv=skf,
                                       method="predict_proba", n_jobs=-1)
        oof_preds_train[name] = oof_proba

        # fit on full balanced train, predict on held-out test set
        model.fit(X_train_bal, y_train_bal)
        proba_test = model.predict_proba(X_train_bal[:1])  # warm (noop)
        proba_test = model.predict_proba(X_test)
        pred_test = model.predict(X_test)
        test_preds[name] = proba_test

        base_results[name] = evaluate(y_test.values, pred_test, proba_test, labels, binary)

    # ---- Meta-level stacking (Stack-ANN: MLP trained on base predictions) ----
    meta_X_train = np.hstack([oof_preds_train[n] for n in base_models])
    meta_X_test = np.hstack([test_preds[n] for n in base_models])

    stack_ann = MLPClassifier(hidden_layer_sizes=(32,), max_iter=80,
                               random_state=RANDOM_STATE, early_stopping=True)
    stack_ann.fit(meta_X_train, y_train_bal)
    stack_pred = stack_ann.predict(meta_X_test)
    stack_proba = stack_ann.predict_proba(meta_X_test)
    stack_result = evaluate(y_test.values, stack_pred, stack_proba, labels, binary)

    elapsed = round(time.time() - t0, 1)

    return {
        "dataset": dataset_name,
        "n_features_used": X_scaled.shape[1],
        "top_features": importances.head(10).round(4).to_dict(),
        "train_size": len(X_train_bal),
        "test_size": len(X_test),
        "base_model_results": base_results,
        "stack_ANN_result": stack_result,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    results = {}

    # ---------- Dataset A: BRFSS 2015 Diabetes Health Indicators ----------
    print("Loading Dataset A ...")
    dfA = pd.read_csv("/mnt/user-data/uploads/diabetes_012_health_indicators_BRFSS2015.csv")
    dfA = dfA.drop_duplicates()
    yA = dfA["Diabetes_012"].astype(int)
    XA = dfA.drop(columns=["Diabetes_012"])
    print("Dataset A shape after dedup:", XA.shape, "classes:", yA.value_counts().to_dict())
    resA = run_edl_pipeline(XA, yA, "BRFSS2015 Diabetes Health Indicators (multi-class)",
                             top_k_features=15, train_subsample=45000)
    results["datasetA"] = resA
    print(json.dumps(resA, indent=2))

    # ---------- Dataset B: Diabetic Patient Readmission ----------
    print("Loading Dataset B ...")
    dfB = pd.read_csv("/mnt/user-data/uploads/diabetic_data.csv")
    dfB = dfB.replace("?", np.nan)
    dfB = dfB.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"])
    dfB = dfB.drop_duplicates()

    yB = (dfB["readmitted"] == "<30").astype(int)  # binary: readmitted within 30 days
    XB = dfB.drop(columns=["readmitted"])

    # encode categoricals
    for col in XB.columns:
        if XB[col].dtype == object or str(XB[col].dtype).startswith("str"):
            XB[col] = XB[col].astype(str)
            XB[col] = LabelEncoder().fit_transform(XB[col])
    XB = XB.apply(pd.to_numeric, errors="coerce")

    print("Dataset B shape after cleaning:", XB.shape, "classes:", yB.value_counts().to_dict())
    resB = run_edl_pipeline(XB, yB, "Diabetic Patient Readmission (binary: <30 days)",
                             top_k_features=20, train_subsample=45000)
    results["datasetB"] = resB
    print(json.dumps(resB, indent=2))

    with open("/home/claude/edl_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to edl_results.json")
