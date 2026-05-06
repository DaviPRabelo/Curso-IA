import pandas as pd
import numpy as np
import mlflow as mlf
import mlflow.keras
from mlflow import MlflowClient
import joblib
import os
import json
import argparse
import requests
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIGURAÇÕES
# =========================
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_SERVING_URI   = os.getenv("MLFLOW_SERVING_URI",  "http://localhost:8080")
MODEL_NAME           = "diabetes_modelv2"

# Métrica usada para decidir se o novo modelo é melhor
# Opções: "accuracy", "f1_score", "auc"
PROMOTION_METRIC = "auc"

FEATURES = [
    "age", "education_level", "income_level", "alcohol_consumption_per_week",
    "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day",
    "family_history_diabetes", "hypertension_history", "cardiovascular_history",
    "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate",
    "ldl_cholesterol", "triglycerides", "glucose_fasting", "glucose_postprandial",
    "insulin_level", "hba1c", "diabetes_risk_score", "cholesterol_ratio",
    "active_lifestyle_index", "gender_Male", "gender_Other", "ethnicity_Black",
    "ethnicity_Hispanic", "ethnicity_Other", "ethnicity_White",
    "employment_status_Retired", "employment_status_Student",
    "employment_status_Unemployed", "smoking_status_Former", "smoking_status_Never"
]

BOOL_FEATURES = [
    "family_history_diabetes", "hypertension_history", "cardiovascular_history",
    "gender_Male", "gender_Other", "ethnicity_Black", "ethnicity_Hispanic",
    "ethnicity_Other", "ethnicity_White", "employment_status_Retired",
    "employment_status_Student", "employment_status_Unemployed",
    "smoking_status_Former", "smoking_status_Never"
]

FLOAT_FEATURES = [
    "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day",
    "bmi", "waist_to_hip_ratio", "insulin_level", "hba1c",
    "diabetes_risk_score", "cholesterol_ratio", "active_lifestyle_index"
]

# =========================
# MÉTRICAS + ROC
# =========================
def eval_metrics(actual, pred, pred_proba=None):
    accuracy = metrics.accuracy_score(actual, pred)
    f1       = metrics.f1_score(actual, pred)
    fpr, tpr, _ = metrics.roc_curve(
        actual, pred_proba if pred_proba is not None else pred
    )
    auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"ROC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    plt.close()

    return accuracy, f1, auc

# =========================
# CARREGAR DADOS
# =========================
def load_data():
    df = pd.read_csv("diabetes_dataset_preprocessed.csv")
    X  = df.drop(["diagnosed_diabetes", "diabetes_stage"], axis=1)
    y  = df["diagnosed_diabetes"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# BUSCAR MÉTRICA DO MODELO EM PRODUCTION
# =========================
def get_production_metric(client: MlflowClient, metric: str):
    """
    Retorna o valor da métrica do modelo atualmente em Production.
    Retorna None se não houver nenhum modelo em Production.
    """
    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not prod_versions:
            return None
        run_id = prod_versions[0].run_id
        run    = client.get_run(run_id)
        return run.data.metrics.get(metric)
    except Exception:
        return None

# =========================
# MODO TREINO (primeira vez)
# =========================
def run_training():
    print("🚀 Iniciando experimento no MLflow...")

    x_train, x_test, y_train, y_test = load_data()

    scaler        = joblib.load("scaler.pkl")
    x_test_scaled = scaler.transform(x_test)

    model = load_model("modelo_diabetesv2.h5", compile=False)

    mlf.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlf.set_experiment("diabetes_experiments")

    with mlf.start_run() as run:
        pred_proba = model.predict(x_test_scaled)
        pred       = (pred_proba.flatten() > 0.5).astype(int)

        accuracy, f1, auc = eval_metrics(y_test, pred, pred_proba)

        mlf.log_metric("accuracy", accuracy)
        mlf.log_metric("f1_score", f1)
        mlf.log_metric("auc", auc)
        mlf.log_param("model_type",    "keras")
        mlf.log_param("training_mode", "initial")
        mlf.log_artifact("plots/ROC_curve.png")
        mlf.log_artifact("scaler.pkl")

        model_info = mlf.keras.log_model(
            model,
            artifact_path="diabetes_model",
            registered_model_name=MODEL_NAME,
        )

        client  = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_latest_versions(MODEL_NAME)[0].version
        client.transition_model_version_stage(
            name=MODEL_NAME, version=version, stage="Production"
        )

        print(f"✅ Modelo registrado: {MODEL_NAME} v{version} → Production")
        print(f"📊 Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(f"\n💡 Para retreinar: python mlf.py retrain")

# =========================
# MODO RETREINO
# =========================
def run_retrain(epochs: int = 10, learning_rate: float = 1e-4):
    """
    Retreina o modelo em Production com fine-tuning.
    Compara a métrica PROMOTION_METRIC com o modelo atual:
      - Melhorou → registra nova versão e promove para Production
      - Piorou   → loga em Staging para rastreabilidade, Production inalterado
    """
    print("🔁 Iniciando retreinamento...")

    x_train, x_test, y_train, y_test = load_data()
    scaler     = joblib.load("scaler.pkl")
    x_train_sc = scaler.transform(x_train).astype(np.float32)
    x_test_sc  = scaler.transform(x_test).astype(np.float32)

    mlf.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlf.set_experiment("diabetes_experiments")

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Métrica do modelo em Production
    prod_metric = get_production_metric(client, PROMOTION_METRIC)
    if prod_metric is not None:
        print(f"📌 Production atual — {PROMOTION_METRIC}: {prod_metric:.4f}")
    else:
        print("⚠️  Nenhum modelo em Production. Novo modelo será promovido automaticamente.")

    # Carrega modelo de Production para fine-tuning
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        model     = mlf.keras.load_model(model_uri)
        print(f"   Modelo carregado: {model_uri}")
    except Exception:
        print("   ⚠️  Falha ao carregar do MLflow — usando modelo local .h5")
        model = load_model("modelo_diabetes.h5", compile=False)

    # Recompila com LR menor para fine-tuning
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )

    print(f"\n🏋️  Fine-tuning — até {epochs} épocas | LR={learning_rate} | early stopping patience=3")
    history = model.fit(
        x_train_sc, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )
    epochs_run = len(history.history["loss"])

    # Avalia novo modelo
    pred_proba = model.predict(x_test_sc)
    pred       = (pred_proba.flatten() > 0.5).astype(int)
    accuracy, f1, auc = eval_metrics(y_test, pred, pred_proba)

    new_metric = {"accuracy": accuracy, "f1_score": f1, "auc": auc}[PROMOTION_METRIC]
    is_better  = prod_metric is None or new_metric > prod_metric

    print(f"\n📊 Novo modelo  — {PROMOTION_METRIC}: {new_metric:.4f}")
    if prod_metric is not None:
        delta_pct = (new_metric - prod_metric) * 100
        print(f"📊 Production   — {PROMOTION_METRIC}: {prod_metric:.4f}  (Δ {delta_pct:+.4f}%)")

    with mlf.start_run() as run:
        mlf.log_metric("accuracy",  accuracy)
        mlf.log_metric("f1_score",  f1)
        mlf.log_metric("auc",       auc)
        mlf.log_param("model_type",       "keras")
        mlf.log_param("training_mode",    "retrain")
        mlf.log_param("learning_rate",    learning_rate)
        mlf.log_param("epochs_run",       epochs_run)
        mlf.log_param("promoted",         str(is_better))
        mlf.log_param("promotion_metric", PROMOTION_METRIC)
        mlf.log_artifact("plots/ROC_curve.png")

        # Registra o modelo retreinado no registry
        mlf.keras.log_model(
            model,
            artifact_path="diabetes_model",
            registered_model_name=MODEL_NAME,
        )
        new_version = client.get_latest_versions(MODEL_NAME)[0].version

        if is_better:
            # Arquiva versão antiga em Production
            try:
                for v in client.get_latest_versions(MODEL_NAME, stages=["Production"]):
                    if v.version != new_version:
                        client.transition_model_version_stage(
                            name=MODEL_NAME, version=v.version, stage="Archived"
                        )
                        print(f"   📦 Versão {v.version} arquivada.")
            except Exception:
                pass

            client.transition_model_version_stage(
                name=MODEL_NAME, version=new_version, stage="Production"
            )
            print(f"\n✅ PROMOVIDO → Production  (v{new_version})")
        else:
            # Mantém em Staging para rastreabilidade
            client.transition_model_version_stage(
                name=MODEL_NAME, version=new_version, stage="Staging"
            )
            print(f"\n❌ NÃO PROMOVIDO — salvo em Staging (v{new_version})")
            print(f"   Production permanece inalterado.")

# =========================
# VALIDAÇÃO DE INPUT
# =========================
def validate_input(data: dict) -> dict:
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError(f"Campos ausentes: {missing}")

    cleaned = {}
    errors  = []
    for feat in FEATURES:
        val = data[feat]
        try:
            if feat in BOOL_FEATURES:
                if isinstance(val, str):
                    cleaned[feat] = val.strip().lower() in ("true", "1", "yes", "sim")
                else:
                    cleaned[feat] = bool(val)
            elif feat in FLOAT_FEATURES:
                cleaned[feat] = float(val)
            else:
                cleaned[feat] = int(val)
        except (ValueError, TypeError):
            errors.append(f"'{feat}': valor inválido '{val}'")

    if errors:
        raise ValueError("Erros de tipo:\n  " + "\n  ".join(errors))

    return cleaned

# =========================
# PRÉ-PROCESSAMENTO
# =========================
def preprocess(data: dict) -> list:
    cleaned  = validate_input(data)
    scaler   = joblib.load("scaler.pkl")
    df_input = pd.DataFrame([cleaned], columns=FEATURES)
    scaled   = scaler.transform(df_input)
    return scaled.tolist()

# =========================
# PREDIÇÃO VIA MLFLOW REST
# =========================
def predict(input_data: dict):
    scaled_row = preprocess(input_data)
    payload    = {"inputs": scaled_row}
    url        = f"{MLFLOW_SERVING_URI}/invocations"
    print(f"\n📡 Enviando para: {url}")

    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"❌ Não foi possível conectar ao MLflow Serving em {MLFLOW_SERVING_URI}")
        return
    except requests.exceptions.HTTPError:
        print(f"❌ Erro HTTP {resp.status_code}: {resp.text}")
        return

    result = resp.json()
    raw    = result.get("predictions", result)
    proba  = float(raw[0][0]) if isinstance(raw[0], list) else float(raw[0])
    label  = int(proba > 0.5)

    print("\n" + "=" * 45)
    print("         🩺 RESULTADO DA PREDIÇÃO")
    print("=" * 45)
    print(f"  Probabilidade de diabetes : {proba * 100:.2f}%")
    print(f"  Diagnóstico               : {'✅ POSITIVO' if label else '🟢 NEGATIVO'}")
    print("=" * 45)

    return {"probability": proba, "diagnosed": bool(label)}

# =========================
# INPUT INTERATIVO
# =========================
def interactive_input() -> dict:
    print("\n📋 Informe os valores (Enter = usar exemplo):")
    print("   Booleanos: true/false ou 1/0\n")

    examples = {
        "age": 58, "education_level": 1, "income_level": 1,
        "alcohol_consumption_per_week": 0, "diet_score": 5.7,
        "sleep_hours_per_day": 7.9, "screen_time_hours_per_day": 7.9,
        "family_history_diabetes": False, "hypertension_history": False,
        "cardiovascular_history": False, "bmi": 30.5, "waist_to_hip_ratio": 0.89,
        "systolic_bp": 134, "diastolic_bp": 78, "heart_rate": 68,
        "ldl_cholesterol": 160, "triglycerides": 145, "glucose_fasting": 136,
        "glucose_postprandial": 236, "insulin_level": 6.36, "hba1c": 8.18,
        "diabetes_risk_score": 29.6, "cholesterol_ratio": 5.83,
        "active_lifestyle_index": 0.51, "gender_Male": True, "gender_Other": False,
        "ethnicity_Black": False, "ethnicity_Hispanic": False,
        "ethnicity_Other": False, "ethnicity_White": False,
        "employment_status_Retired": False, "employment_status_Student": False,
        "employment_status_Unemployed": False, "smoking_status_Former": False,
        "smoking_status_Never": True,
    }

    data = {}
    for feat in FEATURES:
        raw = input(f"  {feat} [{examples[feat]}]: ").strip()
        data[feat] = raw if raw != "" else examples[feat]
    return data

# =========================
# PONTO DE ENTRADA
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🩺 Diabetes — MLflow Train, Retrain & Predict")
    sub    = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Treina e registra o modelo inicial no MLflow")

    retrain_p = sub.add_parser("retrain", help="Retreina e promove apenas se melhorar")
    retrain_p.add_argument("--epochs", type=int,   default=10,   help="Máximo de épocas (default: 10)")
    retrain_p.add_argument("--lr",     type=float, default=1e-4, help="Learning rate (default: 0.0001)")

    pred_p = sub.add_parser("predict", help="Predição via MLflow Model Serving")
    pred_p.add_argument("--json", type=str, metavar="JSON_STRING")
    pred_p.add_argument("--file", type=str, metavar="CAMINHO")

    args = parser.parse_args()

    if args.command is None or args.command == "train":
        print("🔥 Script iniciou")
        run_training()

    elif args.command == "retrain":
        run_retrain(epochs=args.epochs, learning_rate=args.lr)

    elif args.command == "predict":
        if args.file:
            with open(args.file, encoding="utf-8") as fh:
                input_data = json.load(fh)
            print(f"📂 Lendo: {args.file}")
        elif args.json:
            input_data = json.loads(args.json)
        else:
            input_data = interactive_input()

        predict(input_data)