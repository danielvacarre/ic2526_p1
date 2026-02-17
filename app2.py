import io
import base64
import hashlib
import datetime as dt
from typing import Tuple, Optional

import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import f1_score

# ------------------------------ PAGE SETUP ------------------------------
st.set_page_config(page_title="Evaluator F1", page_icon="📊", layout="centered")
st.title("Evaluator F1")
st.caption("Sube un CSV con columnas: id, prediction. El ranking es público y se actualiza al enviar tu fichero.")

MODE_OPTIONS = ["Presencial", "Online"]

# ------------------------------ CONFIG ------------------------------
@st.cache_data(show_spinner=False)
def _gh_headers() -> dict:
    return {
        "Authorization": f"Bearer {st.secrets['GH_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

@st.cache_data(show_spinner=False)
def _gh_repo_paths() -> Tuple[str, str, str, str]:
    """Convenience: (owner_repo, gt_path, log_path, ref)"""
    owner_repo = st.secrets["GT_REPO"]
    gt_path = st.secrets["GT_PATH"]
    log_path = st.secrets["LOG_PATH"]
    ref = st.secrets.get("GT_REF", "master")
    return owner_repo, gt_path, log_path, ref

# ------------------------------ GROUND TRUTH LOADER ------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_gt_from_github() -> pd.DataFrame:
    """Carga el GT desde el repo privado. Soporta ficheros >1MB usando download_url.
    Reintenta de forma ligera ante fallos transitorios.
    """
    owner_repo, gt_path, _, ref = _gh_repo_paths()

    url = f"https://api.github.com/repos/{owner_repo}/contents/{gt_path}?ref={ref}"
    # Primero pedimos el JSON de metadata para evitar el límite de 1MB del 'Accept: raw'
    r = requests.get(url, headers=_gh_headers(), timeout=30)
    r.raise_for_status()
    meta = r.json()

    if isinstance(meta, list):
        raise RuntimeError("GT_PATH apunta a un directorio; debe ser un archivo CSV.")

    # Si GitHub devuelve el contenido embebido y es pequeño, úsalo; si no, usa download_url
    content_b64: Optional[str] = meta.get("content")
    encoding: Optional[str] = meta.get("encoding")
    download_url: Optional[str] = meta.get("download_url")

    if content_b64 and encoding == "base64":
        raw_bytes = base64.b64decode(content_b64)
    elif download_url:
        r2 = requests.get(download_url, headers={"Authorization": _gh_headers()["Authorization"]}, timeout=60)
        r2.raise_for_status()
        raw_bytes = r2.content
    else:
        # Fallback a solicitar el raw directamente (no debería ser necesario)
        r3 = requests.get(url, headers={**_gh_headers(), "Accept": "application/vnd.github.raw"}, timeout=60)
        r3.raise_for_status()
        raw_bytes = r3.content

    df = pd.read_csv(io.BytesIO(raw_bytes))
    # Validación mínima
    expected = {"id", "target"}
    if not expected.issubset(df.columns):
        raise ValueError("El ground truth no tiene columnas: id, target")

    # Garantiza unicidad de IDs en el GT
    if df["id"].duplicated().any():
        dup_count = int(df["id"].duplicated().sum())
        st.warning(f"Se encontraron {dup_count} IDs duplicados en el ground truth; se conservará la primera ocurrencia.")
        df = df.drop_duplicates(subset=["id"], keep="first")

    return df[["id", "target"]]

# ------------------------------ LOG HELPERS ------------------------------

def _put_contents(owner_repo: str, log_path: str, content_bytes: bytes, sha: Optional[str]) -> None:
    url = f"https://api.github.com/repos/{owner_repo}/contents/{log_path}"
    body = {
        "message": f"append score {dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}",
        "content": base64.b64encode(content_bytes).decode(),
        "committer": {"name": "streamlit-bot", "email": "noreply@example.com"},
    }
    if sha:
        body["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=body, timeout=60)
    if r.status_code == 409:
        raise RuntimeError("conflict")
    r.raise_for_status()

def _read_log_from_github_nocache() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    owner_repo, _, log_path, ref = _gh_repo_paths()
    url = f"https://api.github.com/repos/{owner_repo}/contents/{log_path}?ref={ref}"
    r = requests.get(url, headers=_gh_headers(), timeout=30)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    j = r.json()
    content_b64 = j.get("content", "")
    sha = j.get("sha")
    data = base64.b64decode(content_b64) if content_b64 else b""
    if not data:
        return pd.DataFrame(columns=["timestamp_utc", "user_id", "file_sha256", "n_ids", "f1_weighted", "mode"]), sha
    df = pd.read_csv(io.BytesIO(data))
    if "mode" not in df.columns:
        df["mode"] = ""
    return df, sha

@st.cache_data(show_spinner=False, ttl=10)
def read_log_from_github() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    # pequeña caché para visualizar, pero las escrituras siempre usan la versión sin caché
    return _read_log_from_github_nocache()


def append_log_row_to_github(row: dict):
    """Apendiza una fila al CSV de logs en GitHub (crea si no existe).
       Reintenta contra conflictos SHA haciendo re-read *sin caché*.
       Evita duplicados en la misma sesión con session_state.
    """
    owner_repo, _, log_path, _ = _gh_repo_paths()

    key = f"logged_{row['file_sha256']}_{row['f1_weighted']}_{row['n_ids']}_{row.get('mode','')}"
    if st.session_state.get(key):
        return

    # Intentos múltiples por concurrencia alta
    last_exc: Optional[Exception] = None
    for attempt in range(5):
        try:
            df, sha = _read_log_from_github_nocache()
            if df is None:
                new_df = pd.DataFrame([row])
                csv_bytes = new_df.to_csv(index=False).encode()
                _put_contents(owner_repo, log_path, csv_bytes, sha=None)
            else:
                # Alinear columnas esperadas
                for col in ["timestamp_utc", "user_id", "file_sha256", "n_ids", "f1_weighted", "mode"]:
                    if col not in df.columns:
                        df[col] = ""
                new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                csv_bytes = new_df.to_csv(index=False).encode()
                _put_contents(owner_repo, log_path, csv_bytes, sha)
            # Éxito: invalidar caché de lectura para que el ranking se actualice
            try:
                read_log_from_github.clear()
            except Exception:
                pass
            st.session_state[key] = True
            return
        except RuntimeError as e:
            last_exc = e
            # conflicto -> reintenta
            continue
        except Exception as e:
            last_exc = e
            break

    # Si llega aquí, falló tras varios intentos
    if last_exc:
        raise last_exc
        return

    # Alinea columnas esperadas
    for col in ["timestamp_utc", "user_id", "file_sha256", "n_ids", "f1_weighted", "mode"]:
        if col not in df.columns:
            df[col] = ""

    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    csv_bytes = new_df.to_csv(index=False).encode()
    try:
        _put_contents(owner_repo, log_path, csv_bytes, sha)
    except RuntimeError:
        # Reintento
        df2, sha2 = read_log_from_github()
        if df2 is None:
            df2 = pd.DataFrame(columns=new_df.columns)
        new_df2 = pd.concat([df2, pd.DataFrame([row])], ignore_index=True)
        csv_bytes2 = new_df2.to_csv(index=False).encode()
        _put_contents(owner_repo, log_path, csv_bytes2, sha2)
    finally:
        st.session_state[key] = True

# ------------------------------ HISTORY UI ------------------------------

def _render_leaderboard(df: pd.DataFrame, title: str):
    st.markdown(f"### 🏆 {title}")
    if df is None or df.empty:
        st.info("Aún no hay resultados.")
        return

    # Normaliza columnas obligatorias
    for col in ["timestamp_utc", "user_id", "file_sha256", "n_ids", "f1_weighted", "mode"]:
        if col not in df.columns:
            df[col] = ""

    # Quedarse con el mejor F1 por usuario (por modo)
    df = df.copy()
    df["rank_key"] = df["user_id"].astype(str).str.strip().str.lower()
    best_by_user = (
        df.sort_values(["rank_key", "f1_weighted", "timestamp_utc"], ascending=[True, False, False])
          .drop_duplicates(subset=["rank_key"], keep="first")
    )

    leaderboard = (
        best_by_user[["user_id", "f1_weighted", "n_ids", "timestamp_utc"]]
        .sort_values(["f1_weighted", "timestamp_utc"], ascending=[False, False])
        .reset_index(drop=True)
    )
    leaderboard.index = leaderboard.index + 1
    leaderboard.rename(columns={
        "user_id": "Nombre",
        "f1_weighted": "F1 (weighted)",
        "n_ids": "#IDs",
        "timestamp_utc": "Último envío",
    }, inplace=True)

    st.dataframe(leaderboard, use_container_width=True)


def show_public_leaderboards():
    try:
        history_df, _ = read_log_from_github()
    except Exception:
        history_df = None

    st.subheader("Ranking público")
    if history_df is None or history_df.empty:
        st.info("Aún no hay envíos publicados.")
        return

    # Normaliza columna 'mode'
    if "mode" not in history_df.columns:
        history_df["mode"] = ""

    # Tab por modalidad
    tabs = st.tabs(["Global", "Online", "Presencial", "Todos los envíos"])

    with tabs[0]:
        _render_leaderboard(history_df, "Mejores resultados (Global)")

    with tabs[1]:
        online = history_df[history_df["mode"].str.lower().eq("online")]
        _render_leaderboard(online, "Mejores resultados · Online")

    with tabs[2]:
        pres = history_df[history_df["mode"].str.lower().eq("presencial")]
        _render_leaderboard(pres, "Mejores resultados · Presencial")

    with tabs[3]:
        # Tabla completa, descendente por F1
        full = history_df.copy()
        full = full.sort_values(["f1_weighted", "timestamp_utc"], ascending=[False, False])
        st.dataframe(full, use_container_width=True)

# ------------------------------ MAIN UI ------------------------------

st.markdown("### 1) Sube tu CSV")
uploaded = st.file_uploader("Tus predicciones (CSV con columnas: id, prediction)", type=["csv"])

st.markdown("### 2) Identifícate y elige modalidad")
user_id = st.text_input("Nombre (obligatorio)", placeholder="Nombre y apellidos")
valid_name = bool(user_id and user_id.strip())

modes = st.multiselect(
    "Modalidad (selecciona una o ambas)",
    options=MODE_OPTIONS,
    default=["Online"],
    help="Usaremos esta selección para registrar tus resultados en el historial."
)

if not valid_name:
    st.warning("El nombre es obligatorio para poder calcular y registrar resultados.")
if not modes:
    st.warning("Debes seleccionar al menos una modalidad (Presencial u Online).")

with st.spinner("Cargando ground truth..."):
    gt_df = load_gt_from_github()

st.markdown("### 3) Calcula el F1")
run_eval = st.button("Calcular F1")

if run_eval:
    if not uploaded:
        st.error("Primero sube un CSV válido.")
    if not valid_name:
        st.error("El nombre es obligatorio.")
    if not modes:
        st.error("Selecciona al menos una modalidad.")

if run_eval and uploaded and valid_name and modes:
    try:
        user_bytes = uploaded.read()
        user_df = pd.read_csv(io.BytesIO(user_bytes))
    except Exception as e:
        st.error(f"CSV inválido: {e}")
        show_public_leaderboards()
        st.stop()

    required_user_cols = {"id", "prediction"}
    required_gt_cols = {"id", "target"}

    if not required_user_cols.issubset(user_df.columns):
        st.error("Tu CSV debe tener columnas: id, prediction")
        show_public_leaderboards()
        st.stop()
    if not required_gt_cols.issubset(gt_df.columns):
        st.error("El ground truth no tiene columnas: id, target")
        show_public_leaderboards()
        st.stop()

    # Limpieza mínima
    if user_df["id"].duplicated().any():
        du = int(user_df["id"].duplicated().sum())
        st.warning(f"Tu CSV tiene {du} IDs duplicados; se conservará la primera ocurrencia.")
        user_df = user_df.drop_duplicates(subset=["id"], keep="first")

    gt_df["id"], user_df["id"] = gt_df["id"].astype(str), user_df["id"].astype(str)

    # Eliminar filas con NA en prediction o target
    before = len(user_df)
    user_df = user_df.dropna(subset=["prediction"])
    if len(user_df) < before:
        st.info(f"Se eliminaron {before - len(user_df)} filas con prediction vacía.")

    merged = pd.merge(
        gt_df[list(required_gt_cols)],
        user_df[list(required_user_cols)],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        st.error("No hubo IDs coincidentes.")
        show_public_leaderboards()
        st.stop()

    # Alinea tipos de etiquetas
    try:
        if pd.api.types.is_numeric_dtype(merged["target"]) and not pd.api.types.is_numeric_dtype(merged["prediction"]):
            merged["prediction"] = pd.to_numeric(merged["prediction"], errors="coerce")
        elif not pd.api.types.is_numeric_dtype(merged["target"]) and pd.api.types.is_numeric_dtype(merged["prediction"]):
            merged["prediction"] = merged["prediction"].astype(str)
    except Exception:
        pass

    na_before = len(merged)
    merged = merged.dropna(subset=["target", "prediction"])
    if len(merged) < na_before:
        st.info(f"Se eliminaron {na_before - len(merged)} filas con etiquetas no válidas tras normalización.")

    # Cálculo del F1
    try:
        f1_w = f1_score(merged["target"], merged["prediction"], average="weighted")
        st.success(f"F1-score (weighted): {f1_w:.4f}")
        with st.expander("Detalles del conjunto evaluado"):
            st.write({
                "n_ids_merged": int(len(merged)),
                "n_gt": int(len(gt_df)),
                "n_user": int(len(user_df)),
                "n_unique_targets": int(merged["target"].nunique()),
                "n_unique_predictions": int(merged["prediction"].nunique()),
            })
    except Exception as e:
        st.error(f"No se pudo calcular F1: {e}")
        show_public_leaderboards()
        st.stop()

    # ----- Guardar en historial -----
    file_sha256 = hashlib.sha256(user_bytes).hexdigest()
    timestamp_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    ok_modes = []
errors = []
for m in modes:
    row = {
        "timestamp_utc": timestamp_utc,
        "user_id": user_id.strip(),
        "file_sha256": file_sha256,
        "n_ids": int(len(merged)),
        "f1_weighted": float(f1_w),
        "mode": m.lower(),
    }
    try:
        append_log_row_to_github(row)
        ok_modes.append(m)
    except Exception as e:
        errors.append(f"{m}: {e}")

if ok_modes:
    st.success(f"Resultado(s) publicado(s) en: {', '.join(ok_modes)}")
if errors:
    st.warning("No se pudo publicar en: " + ", ".join(errors))

# ----- Mostrar historial (siempre disponible) -----
show_public_leaderboards()
