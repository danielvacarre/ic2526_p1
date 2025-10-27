import io, base64, hashlib, datetime as dt
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import f1_score

# ---------- CARGA DE GROUND TRUTH DESDE REPO PRIVADO ----------
@st.cache_data(show_spinner=False)
def load_gt_from_github() -> pd.DataFrame:
    owner_repo = st.secrets["GT_REPO"]
    path = st.secrets["GT_PATH"]
    ref = st.secrets.get("GT_REF", "master")
    headers = {
        "Authorization": f"Bearer {st.secrets['GH_TOKEN']}",
        "Accept": "application/vnd.github.raw",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}?ref={ref}"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

# ---------- UTIL: LECTURA Y ESCRITURA DEL LOG EN EL REPO PRIVADO ----------
def _gh_headers():
    return {
        "Authorization": f"Bearer {st.secrets['GH_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def read_log_from_github() -> tuple[pd.DataFrame | None, str | None]:
    """Devuelve (df, sha). Si no existe, (None, None)."""
    owner_repo = st.secrets["GT_REPO"]
    log_path = st.secrets["LOG_PATH"]
    ref = st.secrets.get("GT_REF", "master")
    url = f"https://api.github.com/repos/{owner_repo}/contents/{log_path}?ref={ref}"
    r = requests.get(url, headers=_gh_headers(), timeout=30)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    j = r.json()
    content_b64 = j["content"]
    sha = j["sha"]
    data = base64.b64decode(content_b64)
    df = pd.read_csv(io.BytesIO(data))
    return df, sha

def append_log_row_to_github(row: dict):
    """Apendiza una fila al CSV de logs en GitHub (crea si no existe).
       Maneja el SHA para evitar pisar cambios; reintenta una vez si hay conflicto."""
    owner_repo = st.secrets["GT_REPO"]
    log_path = st.secrets["LOG_PATH"]

    def _put(content_bytes: bytes, sha: str | None):
        url = f"https://api.github.com/repos/{owner_repo}/contents/{log_path}"
        body = {
            "message": f"append score {row.get('timestamp_utc','')}",
            "content": base64.b64encode(content_bytes).decode(),
        }
        if sha:
            body["sha"] = sha
        # (Opcional) establece autor del commit
        body["committer"] = {"name": "streamlit-bot", "email": "noreply@example.com"}
        r = requests.put(url, headers=_gh_headers(), json=body, timeout=30)
        if r.status_code == 409:  # SHA desactualizado (race condition)
            raise RuntimeError("conflict")
        r.raise_for_status()

    # 1º: intenta leer el log actual
    df, sha = read_log_from_github()
    if df is None:
        # crear desde cero
        new_df = pd.DataFrame([row])
        csv_bytes = new_df.to_csv(index=False).encode()
        _put(csv_bytes, sha=None)
        return

    # apendizar
    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    csv_bytes = new_df.to_csv(index=False).encode()
    try:
        _put(csv_bytes, sha=sha)
    except RuntimeError:
        # reintento: vuelve a leer y reintenta una vez
        df2, sha2 = read_log_from_github()
        if df2 is None:
            df2 = pd.DataFrame(columns=new_df.columns)
        new_df2 = pd.concat([df2, pd.DataFrame([row])], ignore_index=True)
        csv_bytes2 = new_df2.to_csv(index=False).encode()
        _put(csv_bytes2, sha=sha2)

# ---------- APP ----------
st.title("Evaluator F1")
st.caption("Sube un CSV con columnas: id, prediction")

# (Opcional) identificador del usuario
user_id = st.text_input("Identificador (opcional)", placeholder="nombre, email o alias")

gt_df = load_gt_from_github()

uploaded = st.file_uploader("Tus predicciones", type=["csv"])
if uploaded:
    try:
        user_bytes = uploaded.read()
        user_df = pd.read_csv(io.BytesIO(user_bytes))
    except Exception as e:
        st.error(f"CSV inválido: {e}")
        st.stop()

    required_user_cols = {"id", "prediction"}
    required_gt_cols = {"id", "target"}

    if not required_user_cols.issubset(user_df.columns):
        st.error("Tu CSV debe tener columnas: id, prediction")
        st.stop()
    if not required_gt_cols.issubset(gt_df.columns):
        st.error("El ground truth no tiene columnas: id, target")
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
        st.stop()

    try:
        f1 = f1_score(merged["target"], merged["prediction"], average="weighted")
        st.success(f"F1-score (weighted): {f1:.4f}")
    except Exception as e:
        st.error(f"No se pudo calcular F1: {e}")
        st.stop()

    # ----- Guardar en historial -----
    # hash del fichero del usuario (para detectar duplicados sin almacenar el contenido):
    file_sha256 = hashlib.sha256(user_bytes).hexdigest()
    row = {
        "timestamp_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "user_id": user_id or "",
        "file_sha256": file_sha256,
        "n_ids": int(len(merged)),
        "f1_weighted": float(f1),
    }
    try:
        append_log_row_to_github(row)
        st.info("Resultado guardado en el historial privado.")
    except Exception as e:
        st.warning(f"No se pudo guardar el historial: {e}")

    # ----- Mostrar historial (solo lectura) -----
    try:
        history_df, _ = read_log_from_github()
        if history_df is not None and not history_df.empty:
            st.subheader("Historial de envíos (privado)")
            st.dataframe(history_df.sort_values("timestamp_utc", ascending=False), use_container_width=True)
    except Exception:
        pass
