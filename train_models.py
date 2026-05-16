
import os, joblib, warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model    import LogisticRegression, LinearRegression
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor, \
                                    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm             import SVC, SVR
from sklearn.tree            import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost                 import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
MODELS_DIR   = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Model definitions

CLF_MODELS = {
    "Logistic_Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
    "Random_Forest_Clf": RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced"),
    "XGBoost_Clf": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="mlogloss",
        use_label_encoder=False, verbosity=0),
    "Gradient_Boosting_Clf": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE),
}

REG_MODELS = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest_Reg": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost_Reg": XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, verbosity=0),
    "Gradient_Boosting_Reg": GradientBoostingRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE),
}

# Evaluation helpers

def eval_clf(model, X, y_true, prefix="val"):
    y_pred = model.predict(X)
    metrics = {
        f"{prefix}_accuracy":  accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    try:
        proba = model.predict_proba(X)
        metrics[f"{prefix}_roc_auc"] = roc_auc_score(
            y_true, proba, multi_class="ovr", average="weighted")
    except Exception:
        pass
    return metrics


def eval_reg(model, X, y_true, prefix="val"):
    y_pred = model.predict(X)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    mape   = np.mean(np.abs((y_true - y_pred) / y_true.clip(lower=1))) * 100
    return {
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae":  mae,
        f"{prefix}_r2":   r2,
        f"{prefix}_mape": mape,
    }


def _feature_importance_fig(model, feature_names, model_name):
    """Return a feature-importance matplotlib figure."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        try:
            importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        except Exception:
            return None
    idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="#3498db")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────

def train_classification_models(splits: dict, experiment_name: str = "EMIPredict_Classification"):
    mlflow.set_experiment(experiment_name)
    results = {}

    X_train, X_val = splits["X_train"], splits["X_val"]
    y_train, y_val = splits["y_clf_train"], splits["y_clf_val"]

    for name, model in CLF_MODELS.items():
        print(f"  Training {name} …")
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_type", "classification")
            mlflow.set_tag("model_name", name)

            # log hyper-params
            params = model.get_params()
            mlflow.log_params({k: str(v) for k, v in params.items()})

            model.fit(X_train, y_train)

            train_metrics = eval_clf(model, X_train, y_train, prefix="train")
            val_metrics   = eval_clf(model, X_val,   y_val,   prefix="val")
            all_metrics   = {**train_metrics, **val_metrics}
            mlflow.log_metrics(all_metrics)

            # feature importance plot
            fig = _feature_importance_fig(model, list(X_train.columns), name)
            if fig:
                img_path = os.path.join(MODELS_DIR, f"{name}_fi.png")
                fig.savefig(img_path, bbox_inches="tight")
                mlflow.log_artifact(img_path)
                plt.close(fig)

            # log model
            signature = infer_signature(X_val, model.predict(X_val))
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     signature=signature,
                                     registered_model_name=f"EMIPredict_{name}")

            run_id = mlflow.active_run().info.run_id
            results[name] = {"model": model, "metrics": all_metrics, "run_id": run_id}

    return results


def train_regression_models(splits: dict, experiment_name: str = "EMIPredict_Regression"):
    mlflow.set_experiment(experiment_name)
    results = {}

    X_train, X_val = splits["X_train"], splits["X_val"]
    y_train, y_val = splits["y_reg_train"], splits["y_reg_val"]

    for name, model in REG_MODELS.items():
        print(f"  Training {name} …")
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_type", "regression")
            mlflow.set_tag("model_name", name)

            params = model.get_params()
            mlflow.log_params({k: str(v) for k, v in params.items()})

            model.fit(X_train, y_train)

            train_metrics = eval_reg(model, X_train, y_train, prefix="train")
            val_metrics   = eval_reg(model, X_val,   y_val,   prefix="val")
            all_metrics   = {**train_metrics, **val_metrics}
            mlflow.log_metrics(all_metrics)

            fig = _feature_importance_fig(model, list(X_train.columns), name)
            if fig:
                img_path = os.path.join(MODELS_DIR, f"{name}_fi.png")
                fig.savefig(img_path, bbox_inches="tight")
                mlflow.log_artifact(img_path)
                plt.close(fig)

            signature = infer_signature(X_val, model.predict(X_val))
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     signature=signature,
                                     registered_model_name=f"EMIPredict_{name}")

            run_id = mlflow.active_run().info.run_id
            results[name] = {"model": model, "metrics": all_metrics, "run_id": run_id}

    return results


# ─────────────────────────────────────────────
# Best model selection
# ─────────────────────────────────────────────

def select_best_models(clf_results: dict, reg_results: dict) -> dict:
    """Pick best classification (by val_f1) and regression (by val_rmse)."""
    best_clf_name = max(clf_results, key=lambda n: clf_results[n]["metrics"].get("val_f1", 0))
    best_reg_name = min(reg_results, key=lambda n: reg_results[n]["metrics"].get("val_rmse", 1e9))

    best = {
        "classification": {
            "name": best_clf_name,
            "model": clf_results[best_clf_name]["model"],
            "metrics": clf_results[best_clf_name]["metrics"],
        },
        "regression": {
            "name": best_reg_name,
            "model": reg_results[best_reg_name]["model"],
            "metrics": reg_results[best_reg_name]["metrics"],
        },
    }
    print(f"\nBest classification model : {best_clf_name}")
    print(f"  Val F1   : {best['classification']['metrics']['val_f1']:.4f}")
    print(f"  Val Acc  : {best['classification']['metrics']['val_accuracy']:.4f}")
    print(f"\nBest regression model     : {best_reg_name}")
    print(f"  Val RMSE : {best['regression']['metrics']['val_rmse']:.2f}")
    print(f"  Val R²   : {best['regression']['metrics']['val_r2']:.4f}")
    return best


# ─────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────

def save_best_models(best: dict, encoders: dict, scaler):
    joblib.dump(best["classification"]["model"],  os.path.join(MODELS_DIR, "best_clf_model.pkl"))
    joblib.dump(best["regression"]["model"],      os.path.join(MODELS_DIR, "best_reg_model.pkl"))
    joblib.dump(encoders,                         os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(scaler,                           os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"Models saved to '{MODELS_DIR}/'")


def load_best_models():
    clf    = joblib.load(os.path.join(MODELS_DIR, "best_clf_model.pkl"))
    reg    = joblib.load(os.path.join(MODELS_DIR, "best_reg_model.pkl"))
    enc    = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return clf, reg, enc, scaler


# ─────────────────────────────────────────────
# Test-set final evaluation
# ─────────────────────────────────────────────

def final_evaluation(best: dict, splits: dict):
    print("\n===== FINAL TEST-SET EVALUATION =====")
    clf_m = eval_clf(best["classification"]["model"],
                     splits["X_test"], splits["y_clf_test"], prefix="test")
    reg_m = eval_reg(best["regression"]["model"],
                     splits["X_test"], splits["y_reg_test"], prefix="test")
    print("Classification:", clf_m)
    print("Regression    :", reg_m)
    return clf_m, reg_m


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from data_preprocessing import generate_dataset, prepare_data
    print("Generating dataset (this may take a minute) …")
    raw = generate_dataset(n_per_scenario=80000)
    splits, encoders, scaler, _ = prepare_data(raw)

    print("\nTraining classification models …")
    clf_results = train_classification_models(splits)

    print("\nTraining regression models …")
    reg_results = train_regression_models(splits)

    best = select_best_models(clf_results, reg_results)
    save_best_models(best, encoders, scaler)
    final_evaluation(best, splits)
    print("\nDone! Launch MLflow UI with:  mlflow ui --port 5000")