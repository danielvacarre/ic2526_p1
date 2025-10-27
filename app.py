# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from io import StringIO
from datetime import datetime
import os

st.set_page_config(page_title="Evaluador de F1-macro", page_icon="📊", layout="centered")

# ---------------------------
# Utilidades
# ---------------------------
def load_labels(default_path: str = "data/test_labels.csv") -> pd.DataFrame:
    """Carga labels por defecto (servidor) o desde upload en la sidebar."""
    with st.sidebar:
        st.header("⚙️ Configuración")
        use_uploaded = st.toggle("Subir labels (test_labels.csv) manualmente", value=False)
        uploaded = None
        if use_uploaded:
            uploaded = st.file_uploader("Cargar test_labels.csv", type=["csv"], key="labels_uploader")
    if use_uploaded and uploaded is not None:
        labels = pd.read_csv(uploaded)
    else:
        if not os.path.exists(default_path):
            st.error(f"No se encontró `{default_path}`. Sube los labels en la barra lateral.")
            st.stop()
        labels = pd.read_csv(default_path)
    # Normalizar columnas esperadas
    labels_cols = [c.lower() for c in labels.columns]
    col_map = {c: c.lower() for c in labels.columns}
    labels = labels.rename(columns=col_map)
    if "id" not in labels.columns or "target" not in labels.columns:
        st.error("`test_labels.csv` debe tener columnas: id,target")
        st.stop()
    # Asegurar tipos
    labels = labels[["id", "target"]].copy()
    return labels

def normalize_student_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas del alumno: id, prediction (acepta 'prediccion')."""
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)
    if "prediction" not in df.columns and "prediccion" in df.columns:
        df = df.rename(columns={"prediccion": "prediction"})
    required = {"id", "prediction"}
    if not required.issubset(df.columns):
        st.error("El CSV del alumno debe contener las columnas: id, prediccion (o id, prediction).")
        st.stop()
    # Eliminar filas con id nulo
    df = df.loc[~df["id"].isna()].copy()
    # Resolver duplicados por id (nos quedamos con la última ocurrencia)
    dup_count = df.duplicated(subset=["id"]).sum()
    if dup_count > 0:
        st.warning(f"Se han encontrado {dup_count} predicciones duplicadas por 'id'. Se usará la última.")
        df = df.drop_duplicates(subset=["id"], keep="last")
    # Asegurar numérico en prediction
    # Si vienen strings "0"/"1" o "0.7", intentamos convertir
    try:
        df["prediction"] = pd.to_numeric(df["prediction"])
    except Exception:
        st.error("La columna 'prediction/prediccion' debe ser numérica (0/1 o probabilidad).")
        st.stop()
    return df[["id", "prediction"]]

def binarize_if_needed(pred_series: pd.Series, threshold: float = 0.5) -> (np.ndarray, bool):
    """Convierte probabilidades a 0/1 si detecta valores fuera de {0,1}."""
    unique_vals = pd.unique(pred_series.dropna())
    only_01 = set(pd.Series(unique_vals).dropna().astype(int).unique()) <= {0,1} and np.allclose(unique_vals, np.round(unique_vals))
    if only_01:
        return pred_series.astype(int).values, False
    # Si parece probabilidad (0..1), binarizamos
    if (pred_series.min() >= 0.0) and (pred_series.max() <= 1.0):
        return (pred_series.values >= threshold).astype(int), True
    # Si no encaja, intentamos re-escalar por seguridad
    st.warning("Las predicciones no parecen ni {0,1} ni probabilidades 0..1. Se intentará umbral tras normalizar min-max.")
    x = pred_series.values.astype(float)
    x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
    return (x >= threshold).astype(int), True

def eval_predictions(labels: pd.DataFrame, student: pd.DataFrame):
    """Alinea por id y calcula métricas."""
    merged = labels.merge(student, on="id", how="inner", suffixes=("_true", "_student"))
    missing_in_student = labels.shape[0] - merged.shape[0]
    extra_ids = student.shape[0] - merged.shape[0]
    if missing_in_student > 0:
        st.warning(f"Faltan {missing_in_student} ids del conjunto de test en el CSV del alumno.")
    if extra_ids > 0:
        st.info(f"Se ignoraron {extra_ids} ids que no existen en los labels de test.")

    y_true = merged["target"].astype(int).values
    y_pred_raw = merged["prediction"].copy()
    y_pred, binarized = binarize_if_needed(y_pred_raw)

    if binarized:
        st.caption("Se detectaron probabilidades; se aplicó umbral 0.5 para binarizar.")

    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    return f1, acc, report, cm, merged.shape[0]

def maybe_persist_leaderboard(df_row: pd.DataFrame, path: str = "data/leaderboard.csv"):
    """Intenta persistir el leaderboard (opcional). Si no hay permisos, sigue en memoria."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            old = pd.read_csv(path)
            all_rows = pd.concat([old, df_row], ignore_index=True)
        else:
            all_rows = df_row
        all_rows.to_csv(path, index=False)
        return True
    except Exception:
        return False

# ---------------------------
# Interfaz
# ---------------------------
st.title("📊 Evaluador de F1-macro para predicciones de clasificación")
st.markdown(
    "Sube tu archivo **`predicciones.csv`** con columnas **`id,prediccion`** "
    "(o `id,prediction`). La app lo comparará con **labels de test** y mostrará tu **F1 macro (%)**."
)

labels = load_labels()

st.subheader("1) Sube tus predicciones (CSV)")
student_file = st.file_uploader("Archivo del alumno (id,prediccion)", type=["csv"], help="Asegúrate de incluir todas las filas del test.")

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = pd.DataFrame(columns=["timestamp","nombre","f1_macro","accuracy","n_eval"])

name = st.text_input("Tu nombre/alias (para el ranking)", max_chars=40)

if student_file is not None:
    try:
        student_df = pd.read_csv(student_file)
    except Exception as e:
        st.error(f"No se pudo leer el CSV: {e}")
        st.stop()

    student_df = normalize_student_df(student_df)

    st.subheader("2) Resultados")
    f1, acc, report, cm, n_eval = eval_predictions(labels, student_df)

    st.metric("F1-macro (%)", f"{f1*100:.2f}")
    st.caption(f"Accuracy de referencia: {acc*100:.2f}%  •  Filas evaluadas: {n_eval}")

    # Reporte compacto
    rep_df = pd.DataFrame(report).T
    st.markdown("**Clasification report (resumen)**")
    st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

    # Matriz de confusión
    cm_df = pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"])
    st.markdown("**Matriz de confusión**")
    st.dataframe(cm_df, use_container_width=True)

    # Guardar en leaderboard (en memoria)
    if name.strip():
        new_row = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "nombre": name.strip(),
            "f1_macro": round(f1*100, 4),
            "accuracy": round(acc*100, 4),
            "n_eval": int(n_eval),
        }])
        st.session_state.leaderboard = pd.concat([st.session_state.leaderboard, new_row], ignore_index=True)

        # Persistencia opcional a CSV (si hay permisos)
        if st.checkbox("Guardar puntuación en leaderboard.csv (servidor)"):
            ok = maybe_persist_leaderboard(new_row)
            if ok:
                st.success("Puntuación guardada en data/leaderboard.csv")
            else:
                st.info("No se pudo guardar en disco (sin permisos). Se mantiene en memoria.")

st.subheader("3) Leaderboard (esta sesión)")
if st.session_state.leaderboard.shape[0] > 0:
    lb = st.session_state.leaderboard.sort_values("f1_macro", ascending=False, ignore_index=True)
    st.dataframe(lb, use_container_width=True)
    best = lb.iloc[0]
    st.success(f"🏆 Mejor F1-macro: {best['f1_macro']:.2f}% — {best['nombre']}")
else:
    st.info("Aún no hay puntuaciones. ¡Sube un CSV para evaluar!")

