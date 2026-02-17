import io
import base64
import hashlib
import datetime as dt
from typing import Tuple, Optional

import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import mean_squared_error

# ------------------------------ PAGE SETUP ------------------------------
st.set_page_config(page_title="Evaluator RMSE", page_icon="📉", layout="centered")
st.title("Evaluator RMSE")
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
    """Carga el GT desde el repo privado. Soporta ficheros >1MB usando download_url."""
    owner_repo, gt_path, _, ref = _gh_repo_paths()

    url = f"https://api.github.com/repos/{owner_repo}/contents/{gt_path}?ref={ref}"
    r = requests.get(url, headers=_gh_headers(), timeout=30)
    r.raise_for_status()
    meta = r.json()

    if isinstance(meta, list):
        raise RuntimeError("GT_PATH apunta a un directorio; debe ser un archivo CSV.")

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
        r3 = requests.get(url, headers={**_gh_headers(), "Accept": "application/vnd.github.raw"}, timeout=60)
        r3.raise_for_status()
        raw_bytes = r3.content

    df = pd.read_csv(io.BytesIO(raw_bytes))

    expected = {"id", "target"}
    if not expected.issubset(df.columns):
        raise ValueError("El ground truth no tiene columnas: id, target")

    if df["id"].duplicated().any():
        dup_count = int(df["id"].duplicated().sum())
        st.warning(f"Se encontraron {dup_count} IDs duplicados en el ground truth; se conservará la primera ocurrencia.")
        df = df.drop_duplicates(subset=["id"], keep="first")

    # Asegurar que target es numérico para RMSE
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["target"])
    if len(df) < before:
        st.warning(f"Se eliminaron {before - len(df)} filas del GT con target no numérico/NaN.")

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
        return pd.DataFrame(columns=["timestamp_utc", "user_id", "file_sha256", "n_ids", "rmse", "mode"]), sha
    df = pd.read_csv(io.BytesIO(data))
    if "mode" not in df.columns:
        df["mode"] = ""
    # compat: si venías de f1_weighted, deja columna vacía
    if "rmse" not in df.columns:
        df["rmse"] = pd.NA
    return df, sha

@st.cache_data(show_spinner=False, ttl=10)
def read_log_from_github() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    return _read_log_from_github_nocache()

def append_log_row_to_github(row: dict):
    """Apendiza una fila al CSV de logs en GitHub (crea si no existe).
       Reintenta contra conflictos SHA haciendo re-read *sin caché*.
       Evita duplicados en la misma sesión con session_state.
    """
    owner_repo, _, log_path, _ = _gh_repo_paths()

    key = f"logged_{row['file_sha256']}_{row['rmse']}_{row['n_ids']}_{row.get('mode','')}"
    if st.session_state.get(key):
        return

    last_exc: Optional[Exception] = None
    for _ in range(5):
        try:
            df, sha = _read_log_from_github_nocache()
            if df is None:
                new_df = pd.DataFrame([row])
                csv_bytes = new_df.to_csv(index=False).encode()
                _put_contents(owner_repo, log_path, csv_bytes, sha=None)
            else:
                for col in ["timestamp_utc", "user_id", "file_sha256", "n_ids", "rmse", "mode"]:
                    if col not in df.columns:
                        df[col] = ""
                new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                csv_bytes = new_df.to_csv(index=False).encode()
                _put_contents(owner_repo, log_path, csv_bytes, sha)
            try:
                read_log_from_github.clear()
            except Exception:
                pass
            st.session_state[key] = True
            return
        except RuntimeError as e:
            last_exc = e
            continue
        except Exception as e:
            last_exc = e
            break

    if last_exc:
        raise last_exc

# ------------------------------ HISTORY UI ------------------------------
def _render_leaderboard(df: pd.DataFrame, title: str):
    st.markdown(f"### 🏆 {title}")
    if df is None or df.empty:
        st.info("Aún no hay resultados.")
        return

    for col in ["timestamp_utc", "user_id", "file_sha256", "n_ids", "rmse", "mode"]:
        if col not in df.columns:
            df[col] = ""

    df = df.copy()
    df["rank_key"] = df["user_id"].astype(str).str.strip().str.lower()

    # En RMSE: más bajo es mejor
    best_by_user = (
        df.sort_values(["rank_key", "rmse", "timestamp_utc"], ascending=[True, True, False])
          .drop_duplicates(subset=["rank_key"], keep="first")
    )

    leaderboard = (
        best_by_user[["user_id", "rmse", "n_ids", "timestamp_utc"]]
        .sort_values(["rmse", "timestamp_utc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    leaderboard.index = leaderboard.index + 1
    leaderboard.rename(columns={
        "user_id": "Nombre",
        "rmse": "RMSE",
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

    if "mode" not in history_df.columns:
        history_df["mode"] = ""
    if "rmse" not in history_df.columns:
        history_df["rmse"] = pd.NA

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
        full = history_df.copy()
        full = full.sort_values(["rmse", "timestamp_utc"], ascending=[True, False])
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

st.markdown("### 3) Calcula el RMSE")
run_eval = st.button("Calcular RMSE")

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

    # Deduplicar IDs del alumno
    if user_df["id"].duplicated().any():
        du = int(user_df["id"].duplicated().sum())
        st.warning(f"Tu CSV tiene {du} IDs duplicados; se conservará la primera ocurrencia.")
        user_df = user_df.drop_duplicates(subset=["id"], keep="first")

    gt_df["id"] = gt_df["id"].astype(str)
    user_df["id"] = user_df["id"].astype(str)

    # Asegurar numéricos (RMSE)
    user_df["prediction"] = pd.to_numeric(user_df["prediction"], errors="coerce")
    before_u = len(user_df)
    user_df = user_df.dropna(subset=["prediction"])
    if len(user_df) < before_u:
        st.info(f"Se eliminaron {before_u - len(user_df)} filas con prediction no numérica/NaN.")

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

    # target ya numérico desde load_gt_from_github(), pero por seguridad:
    merged["target"] = pd.to_numeric(merged["target"], errors="coerce")
    na_before = len(merged)
    merged = merged.dropna(subset=["target", "prediction"])
    if len(merged) < na_before:
        st.info(f"Se eliminaron {na_before - len(merged)} filas con valores no válidos tras normalización.")

    # Cálculo del RMSE
    try:
        mse = float(mean_squared_error(merged["target"], merged["prediction"]))
        rmse = mse ** 0.5
        st.success(f"RMSE: {rmse:.6f}")
        with st.expander("Detalles del conjunto evaluado"):
            st.write({
                "n_ids_merged": int(len(merged)),
                "n_gt": int(len(gt_df)),
                "n_user": int(len(user_df)),
                "target_min": float(merged["target"].min()),
                "target_max": float(merged["target"].max()),
                "pred_min": float(merged["prediction"].min()),
                "pred_max": float(merged["prediction"].max()),
            })
    except Exception as e:
        st.error(f"No se pudo calcular RMSE: {e}")
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
            #"file_sha256": file_sha256,
            "n_ids": int(len(merged)),
            "rmse": float(rmse),
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
