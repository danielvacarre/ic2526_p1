import io
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import f1_score

@st.cache_data(show_spinner=False)
def load_gt_from_github() -> pd.DataFrame:
    owner_repo = st.secrets["GT_REPO"]           # "danielvacarre/ic2526"
    path = st.secrets["GT_PATH"]                 # "test_labels.csv"
    ref = st.secrets.get("GT_REF", "main")       # rama o SHA
    headers = {
        "Authorization": f"Bearer {st.secrets['GH_TOKEN']}",
        # Pedimos el contenido en bruto para recibir bytes directamente
        "Accept": "application/vnd.github.raw",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}?ref={ref}"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    # r.content ya son los bytes del CSV
    return pd.read_csv(io.BytesIO(r.content))

# --- Tu app ---
st.title("Evaluator F1")

gt_df = load_gt_from_github()  # NO se expone al usuario
st.caption("Sube un CSV con columnas: id, pred")

uploaded = st.file_uploader("Tus predicciones", type=["csv"])
if uploaded:
    try:
        user_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV inválido: {e}")
        st.stop()

    required_user_cols = {"id", "pred"}
    required_gt_cols = {"id", "label"}

    if not required_user_cols.issubset(user_df.columns):
        st.error("Tu CSV debe tener columnas: id, pred")
        st.stop()
    if not required_gt_cols.issubset(gt_df.columns):
        st.error("El ground truth no tiene columnas: id, label")
        st.stop()

    merged = pd.merge(
        gt_df[list(required_gt_cols)],
        user_df[list(required_user_cols)],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        st.error("No hubo IDs coincidentes.")
    else:
        try:
            f1 = f1_score(merged["label"], merged["pred"], average="weighted")
            st.success(f"F1-score (weighted): {f1:.4f}")
        except Exception as e:
            st.error(f"No se pudo calcular F1: {e}")
