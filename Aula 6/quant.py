import os
import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn import metrics

MODEL_PATH    = "modelo_diabetes.h5"
SCALER_PATH   = "scaler.pkl"
DATASET_PATH  = "diabetes_dataset_preprocessed.csv"
CALIB_SAMPLES = 200
BENCH_SAMPLES = 200
OUTPUT_DIR    = "quantization_comparison"

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

os.makedirs(OUTPUT_DIR, exist_ok=True)


print(" Carregando dados e artefatos...")

df       = pd.read_csv(DATASET_PATH)
X        = df[FEATURES].values.astype(np.float32)
y        = df["diagnosed_diabetes"].values
scaler   = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X).astype(np.float32)

_, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_calib  = X_scaled[:CALIB_SAMPLES]
X_bench  = X_test[:BENCH_SAMPLES]

print(" Carregando modelo Keras original...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def representative_dataset():
    for i in range(CALIB_SAMPLES):
        yield [X_calib[i:i+1].astype(np.float32)]

def convert_float32(model) -> bytes:
    """TFLite sem quantização — equivalente ao Float32 original."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    return conv.convert()

def convert_float16(model) -> bytes:
    """Quantização dos pesos para Float16 — boa para GPUs e Apple Silicon."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    return conv.convert()

def convert_int8(model) -> bytes:
    """Quantização completa INT8 com calibração — máxima velocidade em CPU."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.float32
    conv.inference_output_type = tf.float32
    return conv.convert()


CONFIGS = [
    ("Float32", "modelo_diabetes_float32.tflite", convert_float32),
    ("Float16", "modelo_diabetes_float16.tflite", convert_float16),
    ("INT8",    "modelo_diabetes_int8.tflite",    convert_int8),
]

tflite_models = {}

for name, filename, converter_fn in CONFIGS:
    print(f"\n  Convertendo → {name}...")
    model_bytes = converter_fn(model)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(model_bytes)
    tflite_models[name] = {"bytes": model_bytes, "path": path}
    print(f"   ✅ Salvo: {path}  ({os.path.getsize(path)/1024:.1f} KB)")

# =========================
# 5. BENCHMARK + MÉTRICAS
# =========================
print(f"\n  Benchmarking com {BENCH_SAMPLES} amostras cada...")

# Baseline: modelo Keras original (Float32)
start = time.perf_counter()
for i in range(BENCH_SAMPLES):
    model.predict(X_bench[i:i+1], verbose=0)
time_keras = (time.perf_counter() - start) / BENCH_SAMPLES * 1000

pred_keras = (model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
acc_keras  = metrics.accuracy_score(y_test, pred_keras)
f1_keras   = metrics.f1_score(y_test, pred_keras)
fpr_k, tpr_k, _ = metrics.roc_curve(y_test, model.predict(X_test, verbose=0).flatten())
auc_keras  = metrics.auc(fpr_k, tpr_k)

results = {
    "Keras Float32 (baseline)": {
        "size_kb":   os.path.getsize(MODEL_PATH) / 1024,
        "time_ms":   time_keras,
        "accuracy":  acc_keras,
        "f1":        f1_keras,
        "auc":       auc_keras,
        "fpr":       fpr_k,
        "tpr":       tpr_k,
    }
}

for name, data in tflite_models.items():
    interp = tf.lite.Interpreter(model_content=data["bytes"])
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    # Velocidade
    start = time.perf_counter()
    for i in range(BENCH_SAMPLES):
        interp.set_tensor(inp[0]['index'], X_bench[i:i+1])
        interp.invoke()
    t = (time.perf_counter() - start) / BENCH_SAMPLES * 1000

    # Predições completas
    preds, probas = [], []
    for i in range(len(X_test)):
        interp.set_tensor(inp[0]['index'], X_test[i:i+1])
        interp.invoke()
        p = float(interp.get_tensor(out[0]['index'])[0][0])
        probas.append(p)
        preds.append(int(p > 0.5))

    preds  = np.array(preds)
    probas = np.array(probas)
    fpr, tpr, _ = metrics.roc_curve(y_test, probas)

    results[f"TFLite {name}"] = {
        "size_kb":  os.path.getsize(data["path"]) / 1024,
        "time_ms":  t,
        "accuracy": metrics.accuracy_score(y_test, preds),
        "f1":       metrics.f1_score(y_test, preds),
        "auc":      metrics.auc(fpr, tpr),
        "fpr":      fpr,
        "tpr":      tpr,
    }

# =========================
# 6. TABELA DE RESULTADOS
# =========================
baseline_size = results["Keras Float32 (baseline)"]["size_kb"]
baseline_time = results["Keras Float32 (baseline)"]["time_ms"]
baseline_acc  = results["Keras Float32 (baseline)"]["accuracy"]

print("\n" + "=" * 90)
print(f"{'Modelo':<28} {'Tamanho':>9} {'Redução':>8} {'ms/amostra':>11} {'Speedup':>8} {'Accuracy':>9} {'F1':>7} {'AUC':>7}")
print("-" * 90)

for name, r in results.items():
    reducao = (1 - r["size_kb"] / baseline_size) * 100
    speedup = baseline_time / r["time_ms"]
    delta   = r["accuracy"] - baseline_acc
    flag    = "⚠️ " if abs(delta) > 0.01 else "✅"
    print(
        f"{flag} {name:<26} "
        f"{r['size_kb']:>8.1f}KB "
        f"{reducao:>+7.1f}% "
        f"{r['time_ms']:>10.3f}ms "
        f"{speedup:>7.2f}x "
        f"{r['accuracy']:>8.4f} "
        f"{r['f1']:>6.4f} "
        f"{r['auc']:>6.4f}"
    )

print("=" * 90)

# =========================
# 7. GRÁFICOS COMPARATIVOS
# =========================
print("\n Gerando gráficos comparativos...")

names      = list(results.keys())
sizes      = [r["size_kb"]  for r in results.values()]
times      = [r["time_ms"]  for r in results.values()]
accuracies = [r["accuracy"] for r in results.values()]
f1s        = [r["f1"]       for r in results.values()]
aucs       = [r["auc"]      for r in results.values()]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
short_names = ["Keras\nFloat32", "TFLite\nFloat32", "TFLite\nFloat16", "TFLite\nINT8"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Comparação de Quantização — Modelo Diabetes", fontsize=15, fontweight="bold", y=1.01)

def bar_chart(ax, values, title, ylabel, fmt="{:.1f}", highlight_best=None, best_is_min=True):
    bars = ax.bar(short_names, values, color=COLORS, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                fmt.format(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    if highlight_best is not None:
        idx = values.index(min(values) if best_is_min else max(values))
        bars[idx].set_edgecolor("gold")
        bars[idx].set_linewidth(2.5)

# 1. Tamanho
bar_chart(axes[0,0], sizes, " Tamanho do Arquivo", "Kilobytes (KB)", "{:.1f} KB", best_is_min=True)

# 2. Velocidade
bar_chart(axes[0,1], times, " Velocidade de Inferência", "ms / amostra", "{:.3f}ms", best_is_min=True)

# 3. Speedup
speedups = [baseline_time / t for t in times]
bar_chart(axes[0,2], speedups, " Speedup vs Keras Float32", "Vezes mais rápido (x)", "{:.2f}x", best_is_min=False)

# 4. Acurácia
bar_chart(axes[1,0], accuracies, " Acurácia", "Accuracy", "{:.4f}", best_is_min=False)
axes[1,0].set_ylim(min(accuracies) * 0.999, max(accuracies) * 1.001)

# 5. F1 Score
bar_chart(axes[1,1], f1s, " F1 Score", "F1 Score", "{:.4f}", best_is_min=False)
axes[1,1].set_ylim(min(f1s) * 0.999, max(f1s) * 1.001)

# 6. Curvas ROC sobrepostas
ax_roc = axes[1,2]
roc_colors = COLORS
for (name, r), color, sname in zip(results.items(), roc_colors, short_names):
    ax_roc.plot(r["fpr"], r["tpr"],
                label=f"{sname.replace(chr(10),' ')} (AUC={r['auc']:.4f})",
                color=color, linewidth=1.8)
ax_roc.plot([0,1],[0,1], "k--", linewidth=0.8, alpha=0.5)
ax_roc.set_title(" Curvas ROC", fontweight="bold", pad=8)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(fontsize=7.5, loc="lower right")
ax_roc.spines[["top","right"]].set_visible(False)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "comparacao_quantizacao.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Gráfico salvo: {chart_path}")

# =========================
# 8. RESUMO FINAL
# =========================
print("\n" + "=" * 55)
print("               RECOMENDAÇÃO FINAL")
print("=" * 55)

acc_drops = {n: abs(r["accuracy"] - baseline_acc) for n, r in results.items() if n != "Keras Float32 (baseline)"}
best = min(acc_drops, key=lambda n: (acc_drops[n] > 0.005, -results[n]["time_ms"] / baseline_time))

for name, r in results.items():
    if name == "Keras Float32 (baseline)":
        continue
    drop  = acc_drops[name]
    spdup = baseline_time / r["time_ms"]
    tag   = " ← RECOMENDADO" if name == best else ""
    print(f"  {name:<22}  speedup={spdup:.2f}x  Δacc={drop*100:+.4f}%{tag}")

print("=" * 55)
print(f"\n Todos os arquivos salvos em: ./{OUTPUT_DIR}/")