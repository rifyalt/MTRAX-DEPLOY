# =====================================
# CORE IMPORTS
# =====================================
import streamlit as st
import pandas as pd
import os
import hashlib
import base64
import json
import requests
import numpy as np
import shutil
import time
import re
import hmac
import secrets

from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gdown
import xml.etree.ElementTree as ET

# =====================================
# OPTIONAL LSTM SUPPORT
# =====================================
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# =====================================
# 2FA / TOTP SUPPORT
# ── Ditambahkan: Pure-Python TOTP (RFC 6238) — tidak butuh library tambahan ──
# Jika pyotp & qrcode terinstall, QR code akan ditampilkan.
# Jika tidak, user cukup input manual key ke Google Authenticator.
# =====================================
import struct as _struct
import hmac   as _hmac_mod
import base64 as _b64_mod

try:
    import pyotp
    import qrcode
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False

def _hotp(key_b32: str, counter: int) -> str:
    """HMAC-based OTP — RFC 4226."""
    key  = _b64_mod.b32decode(key_b32.upper().replace(" ", ""), casefold=True)
    msg  = _struct.pack(">Q", counter)
    h    = _hmac_mod.new(key, msg, "sha1").digest()
    off  = h[-1] & 0x0F
    code = (_struct.unpack(">I", h[off:off+4])[0] & 0x7FFFFFFF) % 1_000_000
    return str(code).zfill(6)

def _totp_now(key_b32: str, window: int = 30) -> str:
    counter = int(time.time()) // window
    return _hotp(key_b32, counter)

def _totp_valid(key_b32: str, code: str, window: int = 30, drift: int = 1) -> bool:
    """Verify TOTP allowing ±drift windows (toleransi clock skew)."""
    counter = int(time.time()) // window
    for d in range(-drift, drift + 1):
        if _hmac_mod.compare_digest(_hotp(key_b32, counter + d), code.strip()):
            return True
    return False

def _generate_totp_secret() -> str:
    """Generate random 20-byte base32 TOTP secret."""
    return _b64_mod.b32encode(os.urandom(20)).decode()

def _provisioning_uri(secret: str, username: str, issuer: str = "MTRAX") -> str:
    from urllib.parse import quote
    return (
        f"otpauth://totp/{quote(issuer)}:{quote(username)}"
        f"?secret={secret}&issuer={quote(issuer)}&algorithm=SHA1&digits=6&period=30"
    )

def _make_qr_b64(uri: str) -> str:
    """Render QR code -> base64 PNG. Tries qrcode -> segno -> pyqrcode in order."""
    import io

    # attempt 1: qrcode (pil)
    try:
        import qrcode
        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#1BA0E2", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return _b64_mod.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    # attempt 2: segno
    try:
        import segno
        qr = segno.make(uri, error="M")
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=6, border=2,
                dark="#1BA0E2", light="white")
        buf.seek(0)
        return _b64_mod.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    # attempt 3: pyqrcode
    try:
        import pyqrcode
        qr = pyqrcode.create(uri)
        buf = io.BytesIO()
        qr.png(buf, scale=6, module_color=(156, 87, 137), background=(255, 255, 255))
        buf.seek(0)
        return _b64_mod.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    return ""

def _load_totp_secrets() -> dict:
    """
    Load TOTP secrets — semua user berbagi secret milik 'admin'.
    Hanya satu QR / kode Authenticator yang perlu di-setup oleh admin.
    Priority: secrets.toml [totp][admin] -> session_state (runtime-generated).
    """
    # Ambil / generate satu shared secret dari admin
    shared_secret = None
    try:
        val = st.secrets["totp"]["admin"]
        if val:
            shared_secret = val
    except Exception:
        pass

    if shared_secret is None:
        key = "_totp_secret_admin"
        if key not in st.session_state:
            st.session_state[key] = _generate_totp_secret()
        shared_secret = st.session_state[key]

    # Semua user memakai secret yang sama (shared admin TOTP)
    return {uname: shared_secret for uname in USERS.keys()}

# ======================================
# ADVANCED LOGIN SECURITY CONFIG
# ======================================

# Jika pakai secrets.toml (recommended)
MAX_LOGIN_ATTEMPTS = st.secrets.get("security", {}).get("max_login_attempts", 5)
LOCKOUT_SECONDS = st.secrets.get("security", {}).get("lockout_seconds", 300)  # 5 menit
SESSION_TIMEOUT = st.secrets.get("security", {}).get("session_timeout_minutes", 30) * 60

def secure_compare(plain_password, stored_hash):
    """
    Timing attack safe comparison
    """
    hashed_input = hashlib.sha256(plain_password.encode()).hexdigest()
    return hmac.compare_digest(hashed_input, stored_hash)

def validate_username(username):
    """
    Mencegah injection / karakter aneh
    """
    return bool(re.fullmatch(r"[A-Za-z0-9_]{3,30}", username))

def init_login_security():
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    if "lockout_until" not in st.session_state:
        st.session_state.lockout_until = 0
    if "login_time" not in st.session_state:
        st.session_state.login_time = None
    # ── 2FA states ──────────────────────────────────────────
    if "pending_2fa" not in st.session_state:
        st.session_state.pending_2fa = False
    if "pending_user" not in st.session_state:
        st.session_state.pending_user = ""
    if "pending_role" not in st.session_state:
        st.session_state.pending_role = ""
    if "totp_enrolled" not in st.session_state:
        st.session_state.totp_enrolled = {}   # {username: True/False}

def check_session_timeout():
    if st.session_state.get("login_time"):
        if time.time() - st.session_state.login_time > SESSION_TIMEOUT:
            _revoke_session_token(st.session_state.get("_session_token"))
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            _clear_session_cookie()
            try:
                if _SESSION_QUERY_KEY in st.query_params:
                    del st.query_params[_SESSION_QUERY_KEY]
            except Exception:
                pass
            st.warning("Session expired. Please login again.")
            st.stop()


# ======================================
# PERSISTENT LOGIN ACROSS PAGE RELOAD
# ── session_state Streamlit terikat ke koneksi WebSocket browser, jadi akan
#    kosong lagi setiap kali halaman di-reload (F5). Token sesi acak disimpan
#    di COOKIE browser lewat JS. (Catatan lama di sini pernah menyebut
#    st.query_params "tidak reliable setelah reload" merujuk ke GitHub issue
#    streamlit/streamlit#10406 — setelah dicek ulang, isu itu ternyata bug di
#    kode pelapornya sendiri, bukan di st.query_params; tim Streamlit
#    mengonfirmasi query_params PERSISTEN dengan benar setelah reload browser.
#    Karena itu, di bawah, st.query_params dipakai sebagai jalur PEMULIHAN
#    UTAMA yang reliable — cookie tetap dipertahankan sebagai lapisan
#    tambahan saja.)
#
#    Streamlit hanya bisa MEMBACA cookie secara native (st.context.cookies,
#    read-only, sejak v1.38+). Untuk MENULIS cookie, tetap perlu sedikit
#    JavaScript lewat components.html — Streamlit belum punya API native
#    untuk itu (github.com/streamlit/streamlit/issues/7892).
#
#    CATATAN KEAMANAN: cookie diset tanpa flag HttpOnly (karena ditulis lewat
#    JS di sisi klien, bukan header HTTP dari server), jadi secara teknis
#    bisa dibaca script lain di origin yang sama. Untuk aplikasi internal ini
#    risikonya wajar selama diakses di jaringan/perangkat tepercaya, dan
#    session_timeout_minutes di secrets.toml (default 30 menit) membatasi
#    umur token-nya.
# ======================================
_SESSION_COOKIE_NAME = "mtrax_session"
# ── FALLBACK RESTORE VIA URL QUERY PARAM ────────────────────────────────
# `st.context.cookies` adalah API yang relatif baru & masih "experimental"
# di Streamlit — kalau versi Streamlit yang dipakai di server produksi lebih
# lama, atribut ini bisa saja tidak ada / selalu gagal diakses, sehingga
# cookie TIDAK PERNAH berhasil dibaca kembali walau sudah benar tersimpan
# di browser -> user selalu ter-logout setiap kali klik reload/F5. Ini
# match persis dengan bug yang dilaporkan tetap terjadi walau timing
# penulisan cookie sudah diperbaiki.
#
# Sebagai jalur pemulihan yang jauh lebih reliable (tidak butuh JS sama
# sekali, tidak bergantung API Streamlit yang baru), token sesi JUGA
# disimpan di URL lewat `st.query_params`. Query string SELALU dipertahankan
# browser saat halaman di-reload (F5) — ini perilaku dasar browser, bukan
# fitur Streamlit yang bisa berubah-ubah reliabilitasnya.
_SESSION_QUERY_KEY = "s"

@st.cache_resource
def _get_session_store():
    """Dict bersama di semua sesi selama proses server ini hidup: token -> {username, role, expires_at}."""
    return {}

def _create_session_token(username: str, role: str) -> str:
    token = secrets.token_urlsafe(32)
    store = _get_session_store()
    store[token] = {
        "username": username,
        "role": role,
        "expires_at": time.time() + SESSION_TIMEOUT,
    }
    return token

def _validate_session_token(token: str):
    """Return (username, role) kalau token valid & belum kedaluwarsa, else None. Otomatis bersihkan token basi."""
    if not token:
        return None
    store = _get_session_store()
    entry = store.get(token)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        store.pop(token, None)
        return None
    # Sliding expiry: perpanjang masa berlaku selama masih aktif dipakai
    entry["expires_at"] = time.time() + SESSION_TIMEOUT
    return entry["username"], entry["role"]

def _revoke_session_token(token):
    if token:
        _get_session_store().pop(token, None)

def _set_session_cookie(token: str):
    """Tulis cookie sesi di browser lewat JS (Streamlit tidak punya API native untuk set cookie).

    ── CATATAN PERBAIKAN ───────────────────────────────────────────────────
    Sempat dicoba memicu `window.parent.location.reload()` dari DALAM script
    ini (persis setelah `document.cookie = ...`) supaya urutan cookie->reload
    terjamin tanpa bergantung ke `st.rerun()`. Ternyata pendekatan itu TIDAK
    reliable: iframe yang dipakai `components.html` tidak selalu diberi izin
    untuk menavigasi/reload frame induknya oleh browser, sehingga reload bisa
    gagal diam-diam atau menyebabkan reload terjadi SEBELUM cookie sempat
    tersimpan — hasilnya user malah tidak pernah berhasil masuk ke dashboard.

    Jadi pendekatan dikembalikan ke cara yang terbukti reliable: cookie
    ditulis di sini, lalu pemanggil (mis. _finish_login) memakai `st.rerun()`
    untuk lanjut ke dashboard DALAM SESI YANG SAMA — ini tidak bergantung
    sama sekali pada cookie/JS karena `st.session_state.authenticated` sudah
    di-set duluan sebelum rerun. Cookie di sini murni untuk keperluan lain:
    memulihkan sesi kalau nanti browser BENAR-BENAR di-refresh (F5). Supaya
    itu tetap reliable, pemanggil perlu memberi jeda (time.sleep) yang cukup
    sebelum memanggil st.rerun(), agar browser sempat memuat iframe & benar-
    benar mengeksekusi baris document.cookie di dalamnya sebelum DOM diganti
    oleh render hasil rerun.
    """
    import streamlit.components.v1 as _components
    max_age = int(SESSION_TIMEOUT)
    _components.html(
        f"""<script>
        document.cookie = "{_SESSION_COOKIE_NAME}={token}; max-age={max_age}; path=/; SameSite=Lax";
        </script>""",
        height=0
    )

def _clear_session_cookie():
    """Hapus cookie sesi di browser (dipanggil saat logout / timeout)."""
    import streamlit.components.v1 as _components
    _components.html(
        f"""<script>
        document.cookie = "{_SESSION_COOKIE_NAME}=; max-age=0; path=/; SameSite=Lax";
        </script>""",
        height=0
    )

def _try_restore_session_from_cookie():
    """Dipanggil sebelum pengecekan authenticated — pulihkan sesi dari cookie/URL kalau ada & valid."""
    if st.session_state.get("authenticated"):
        return

    token = None
    # 1) Coba dari cookie browser (kalau st.context.cookies tersedia di versi
    #    Streamlit ini, dan JS sempat sukses menulis cookie sebelumnya).
    try:
        token = st.context.cookies.get(_SESSION_COOKIE_NAME)
    except Exception:
        token = None

    # 2) Fallback ke query param URL kalau cookie tidak ditemukan/tidak bisa
    #    dibaca. Ini jalur utama yang reliable — lihat catatan di deklarasi
    #    _SESSION_QUERY_KEY di atas.
    if not token:
        try:
            token = st.query_params.get(_SESSION_QUERY_KEY)
        except Exception:
            token = None

    if not token:
        return
    result = _validate_session_token(token)
    if result is None:
        # Token di URL/cookie sudah tidak valid (kedaluwarsa/di-revoke) —
        # bersihkan supaya tidak terus dicoba tiap reload.
        try:
            if _SESSION_QUERY_KEY in st.query_params:
                del st.query_params[_SESSION_QUERY_KEY]
        except Exception:
            pass
        return
    username, role = result
    st.session_state.login_attempts = 0
    st.session_state.lockout_until  = 0
    st.session_state.authenticated  = True
    st.session_state.logged_in      = True
    st.session_state.username       = username
    st.session_state.role           = role
    st.session_state.login_time     = time.time()
    st.session_state.pending_2fa    = False
    st.session_state.pending_user   = ""
    st.session_state._session_token = token
    # Pastikan query param ikut tersimpan/ter-refresh di URL (kalau restore
    # ini berasal dari cookie, query param mungkin belum ada — tambahkan
    # supaya reload berikutnya makin reliable).
    try:
        st.query_params[_SESSION_QUERY_KEY] = token
    except Exception:
        pass

init_login_security()


# =====================================
# HELPER FUNCTIONS
# =====================================
def get_greeting():
    """Mendapatkan greeting sesuai waktu Jakarta"""
    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    hour = now.hour

    if hour < 11:
        greet = "Good Morning"
    elif hour < 15:
        greet = "Good Afternoon"
    elif hour < 18:
        greet = "Good Evening"
    else:
        greet = "Good Night"

    return greet, now


@st.cache_data(show_spinner=False)
def load_drive_data(folder_id, drop_cols):
    shutil.rmtree("data_temp", ignore_errors=True)
    os.makedirs("data_temp", exist_ok=True)

    gdown.download_folder(
        id=folder_id,
        output="data_temp",
        quiet=True,
        use_cookies=False
    )

    files = [
        f for f in os.listdir("data_temp")
        if f.endswith((".xlsx", ".xls"))
    ]

    dfs = []

    for f in files:
        df = pd.read_excel(os.path.join("data_temp", f))

        # drop kolom tidak perlu
        df = df.drop(
            columns=[c for c in drop_cols if c in df.columns],
            errors="ignore"
        )

        # 🔹 TRIM & CLEAN STRING DATA
        df = trim_string_columns(df)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

@st.cache_data(show_spinner=False)
def build_employee_cohort(df):
    required_cols = ["Employee Id", "Issue Time", "Travel Request Number"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    df = df.copy()
    df["Issue Time"] = pd.to_datetime(df["Issue Time"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Issue Time", "Employee Id"])

    df["OrderMonth"] = df["Issue Time"].dt.to_period("M")
    
    # Cohort = first booking month per employee
    df["CohortMonth"] = (
        df.groupby("Employee Id")["OrderMonth"]
        .transform("min")
    )

    # Cohort index (bulan ke-n sejak first booking)
    df["CohortIndex"] = (
        df["OrderMonth"].astype(int) -
        df["CohortMonth"].astype(int)
    )

    cohort = (
        df.groupby(["CohortMonth", "CohortIndex"])["Travel Request Number"]
        .nunique()
        .reset_index()
    )

    cohort_pivot = cohort.pivot(
        index="CohortMonth",
        columns="CohortIndex",
        values="Travel Request Number"
    ).fillna(0)

    cohort_pivot.index = cohort_pivot.index.astype(str)

    return cohort_pivot


#==========================#
# FUNGSI PERSONA CLUSTERING (KMeans) — DI-CACHE
# ── Perhitungan ini (parsing tanggal, agregasi, scaling, fit KMeans) tidak
#    tergantung pada Employee Id mana yang dipilih user di dropdown, jadi
#    hasilnya bisa dipakai ulang selama data sumber belum berubah. Tanpa cache,
#    seluruh pipeline ini dihitung ulang setiap kali dropdown diganti, karena
#    st.tabs() menjalankan ulang seluruh script di setiap interaksi widget. ──
#==========================#
@st.cache_data(show_spinner=False)
def build_employee_persona_clusters(df_behavior_raw):
    df_behavior = df_behavior_raw.copy()

    df_behavior["Issue Time"] = pd.to_datetime(df_behavior["Issue Time"], errors="coerce", dayfirst=True)
    df_behavior["Check in Date"] = pd.to_datetime(df_behavior["Check in Date"], errors="coerce", dayfirst=True)
    df_behavior["Check out Date"] = pd.to_datetime(df_behavior["Check out Date"], errors="coerce", dayfirst=True)

    df_behavior["Lead_Time"] = (df_behavior["Check in Date"] - df_behavior["Issue Time"]).dt.days
    df_behavior["Last_Minute"] = df_behavior["Lead_Time"].apply(lambda x: 1 if pd.notnull(x) and x <= 2 else 0)
    df_behavior["Weekend_Stay"] = df_behavior["Check in Date"].dt.weekday.apply(lambda x: 1 if pd.notnull(x) and x >= 5 else 0)

    employee_features = df_behavior.groupby("Employee Id").agg(
        Booking_Frequency=("Travel Request Number", "nunique"),
        Avg_Lead_Time=("Lead_Time", "mean"),
        Last_Minute_Ratio=("Last_Minute", "mean"),
        Avg_Stay=("Number of Rooms Night", "mean"),
        Weekend_Ratio=("Weekend_Stay", "mean")
    ).reset_index()

    employee_features = employee_features.fillna(0)

    feature_cols = ["Booking_Frequency","Avg_Lead_Time","Last_Minute_Ratio","Avg_Stay","Weekend_Ratio"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(employee_features[feature_cols])

    n_employee = len(employee_features)
    if n_employee >= 4: n_cluster = 4
    elif n_employee >= 2: n_cluster = 2
    else: n_cluster = 1

    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    employee_features["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_profile = employee_features.groupby("Cluster")[feature_cols].mean()

    persona_map = {}
    for cluster_id, row in cluster_profile.iterrows():
        if row["Last_Minute_Ratio"] > 0.5: persona = "Last Minute Traveler"
        elif row["Avg_Lead_Time"] > 14: persona = "Strategic Planner"
        elif row["Weekend_Ratio"] > 0.4: persona = "Weekend Traveler"
        elif row["Booking_Frequency"] > employee_features["Booking_Frequency"].median(): persona = "Frequent Traveler"
        else: persona = "Regular Business Traveler"
        persona_map[cluster_id] = persona

    employee_features["Persona"] = employee_features["Cluster"].map(persona_map)

    return employee_features, feature_cols


#==========================#
# FUNGSI AUTO-CANONICAL MAPPING
#==========================#
@st.cache_data(show_spinner=False)
def auto_canonical_hotel_mapping(
    df,
    hotel_col="Hotel Name",
    threshold=0.88
):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    hotel_series = (
        df[hotel_col]
        .dropna()
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.strip()
        .drop_duplicates()
    )

    hotel_names = hotel_series.tolist()

    if len(hotel_names) < 2:
        df["Canonical Hotel Name"] = df[hotel_col]
        return df, pd.DataFrame()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )

    tfidf = vectorizer.fit_transform(hotel_names)
    similarity = cosine_similarity(tfidf)

    clusters = {}
    visited = set()

    for i, name in enumerate(hotel_names):
        if i in visited:
            continue

        group = [name]
        visited.add(i)

        for j in range(i + 1, len(hotel_names)):
            if similarity[i, j] >= threshold:
                group.append(hotel_names[j])
                visited.add(j)

        # canonical = nama terpanjang
        canonical = max(group, key=len)
        for g in group:
            clusters[g] = canonical

    mapping_df = pd.DataFrame(
        clusters.items(),
        columns=["Hotel Name Clean", "Canonical Hotel Name"]
    )

    # merge ke df awal
    df_out = df.copy()
    df_out["_hotel_clean"] = (
        df_out[hotel_col]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.strip()
    )

    df_out = df_out.merge(
        mapping_df,
        left_on="_hotel_clean",
        right_on="Hotel Name Clean",
        how="left"
    )

    df_out["Canonical Hotel Name"] = (
        df_out["Canonical Hotel Name"]
        .fillna(df_out[hotel_col])
    )

    df_out.drop(columns=["_hotel_clean", "Hotel Name Clean"], inplace=True)

    return df_out, mapping_df



#==========================#    
# TEXT SIMILARITY
#==========================#
@st.cache_data(show_spinner=False)
def hotel_name_similarity(df, text_col="Hotel Name", threshold=0.75):
    df_text = (
        df[[text_col]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    df_text[text_col] = (
        df_text[text_col]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.strip()
    )

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )

    tfidf_matrix = vectorizer.fit_transform(df_text[text_col])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    results = []
    names = df_text[text_col].tolist()

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = similarity_matrix[i, j]
            if score >= threshold:
                results.append({
                    "Hotel Name A": names[i],
                    "Hotel Name B": names[j],
                    "Similarity Score": round(score, 3)
                })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df["Level"] = pd.cut(
            result_df["Similarity Score"],
            bins=[0.7, 0.8, 0.9, 1.0],
            labels=["Medium", "High", "Very High"]
        ).astype(str)

        result_df = result_df.sort_values(
            by="Similarity Score",
            ascending=False
        )

    return result_df

def trim_string_columns(df):
    """
    Membersihkan semua kolom bertipe object (string):
    - strip spasi depan & belakang
    - hapus spasi ganda di tengah
    """
    df_clean = df.copy()

    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        # kembalikan NaN asli (bukan string 'nan')
        df_clean[col] = df_clean[col].replace("nan", np.nan)

    return df_clean

# ===============================
# SESSION STATE INIT
# ===============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "df_all" not in st.session_state:
    st.session_state.df_all = pd.DataFrame()

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "data_period" not in st.session_state:
    st.session_state.data_period = ""

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

# ======================================
# USER LOGIN
# ======================================
def hash_password(password: str) -> str:
    """Hash password menggunakan SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "admin": {
        "password": st.secrets["auth"]["admin_password"],
        "role": "Admin"
    },
    "ssc": {
        "password": st.secrets["auth"]["ssc_password"],
        "role": "Analyst"
    },
    "dtm": {
        "password": st.secrets["auth"]["dtm_password"],
        "role": "Viewer"
    },
}

# ======================================
# BMKG FUNCTIONS
# ======================================
@st.cache_data(ttl=300, show_spinner=False)
def get_bmkg_realtime_quake():
    """Mengambil data gempa real-time dari BMKG (di-cache 5 menit — sebelumnya
    fetch HTTP baru dilakukan di SETIAP interaksi widget di seluruh aplikasi,
    karena fungsi ini dipanggil ulang tiap kali main_app() jalan)."""
    url = "https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "Infogempa" in data and "gempa" in data["Infogempa"]:
            return data["Infogempa"]["gempa"]

        return None

    except Exception:
        # Tidak lagi memanggil st.sidebar.error() di sini — pemanggilan elemen UI
        # dari dalam fungsi ber-cache bisa berperilaku tidak konsisten antara
        # cache-hit dan cache-miss. Kegagalan cukup direpresentasikan sebagai None;
        # tampilan sisi pemanggil sudah punya fallback "No recent data".
        return None

# ======================================
# NEWS TICKER
# ======================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss_news(limit=10):
    """Sebelumnya melakukan 3x HTTP request ke Google News di SETIAP interaksi
    widget di seluruh aplikasi (karena dipanggil ulang tiap main_app() jalan,
    dan main_app() jalan ulang di setiap klik). Sekarang di-cache 5 menit —
    berita tidak perlu ambil ulang secepat itu."""
    urls = [
        "https://news.google.com/rss/search?q=danantara&hl=id&gl=ID&ceid=ID:id",
        "https://news.google.com/rss/search?q=pertamina&hl=id&gl=ID&ceid=ID:id",
        "https://news.google.com/rss/search?q=bmkg&hl=id&gl=ID&ceid=ID:id"
    ]

    news = []

    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            root = ET.fromstring(r.content)

            for item in root.findall(".//item")[:limit]:
                title = item.find("title").text
                link = item.find("link").text
                news.append({"title": title, "link": link})
        except Exception:
            pass

    return news[:limit]

def render_news_ticker(news):
    items = "".join([
        f'<a class="news-item" href="{n["link"]}" target="_blank">{n["title"]}</a>'
        for n in news
    ])
    st.markdown(f"""
    <div class="news-ticker">
        <div class="ticker-content">{items}</div>
    </div>
    """, unsafe_allow_html=True)


# ======================================
# ── 2FA PAGES (DITAMBAHKAN) ──
# ======================================

def _2fa_css():
    return """
    <style>
    .fa-card {
        background: white;
        border-radius: 8px;
        padding: 36px 40px;
        max-width: 440px;
        margin: 3rem auto 0 auto;
        box-shadow: 0 2px 16px rgba(0,0,0,0.09);
        border-top: 3px solid #1BA0E2;
    }
    .fa-title {
        font-size: 1.25em;
        font-weight: 700;
        color: #1BA0E2;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }
    .fa-sub {
        font-size: 0.83em;
        color: #888;
        margin-bottom: 24px;
        line-height: 1.6;
    }
    .fa-qr-wrap {
        background: #f0f8ff;
        border: 1px solid #b8d9f0;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-bottom: 18px;
    }
    .fa-secret-box {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: 'Courier New', monospace;
        font-size: 1.05em;
        letter-spacing: 0.14em;
        color: #333;
        text-align: center;
        margin: 10px 0 18px 0;
        word-break: break-all;
    }
    .fa-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 12px;
    }
    .fa-step-num {
        background: #1BA0E2;
        color: white;
        border-radius: 50%;
        width: 22px; height: 22px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.75em; font-weight: 700;
        flex-shrink: 0; margin-top: 2px;
    }
    .fa-step-text { font-size: 0.85em; color: #444; line-height: 1.55; }
    .fa-timer {
        display: inline-flex; align-items: center; gap: 8px;
        background: #f0f8ff; border: 1px solid #b8d9f0;
        border-radius: 20px; padding: 5px 16px;
        font-size: 0.78em; color: #1BA0E2;
        margin-top: 16px;
    }
    </style>
    """

def twofa_setup_page(username: str):
    """Halaman setup 2FA — redesign v2: editorial split-panel."""
    import streamlit.components.v1 as _components
    st.set_page_config(page_title="Setup 2FA | MTRAX", layout="centered")

    totp_secrets = _load_totp_secrets()
    secret       = totp_secrets[username]
    uri          = _provisioning_uri(secret, "admin")   # akun selalu MTRAX:admin
    qr_b64       = _make_qr_b64(uri)
    fmt_secret   = " ".join([secret[i:i+4] for i in range(0, len(secret), 4)])

    if qr_b64:
        qr_html = f'<img id="qrimg" src="data:image/png;base64,{qr_b64}" alt="QR Code" />'
        qr_note = ''
    else:
        qr_html = ''
        qr_note = '<div class="qr-missing">📱 Install <code>qrcode[pil]</code> atau <code>segno</code> untuk QR code</div>'

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box;}}

:root {{
  --ink:    #0f0a12;
  --ink2:   #3a2a3a;
  --muted:  #8a7a8a;
  --line:   #e8dde8;
  --bg:     #f9f6fb;
  --white:  #ffffff;
  --accent: #1BA0E2;
  --acc-l:  #f3eaf1;
  --acc-d:  #6a1a5a;
  --green:  #22c55e;
  --mono:   'JetBrains Mono', monospace;
  --sans:   'Sora', sans-serif;
}}

body{{font-family:var(--sans);background:var(--bg);padding:20px 12px 28px;}}

/* ── wrapper ── */
.w{{max-width:500px;margin:0 auto;animation:up .5s cubic-bezier(.16,1,.3,1) both;}}
@keyframes up{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:none}}}}

/* ── topbar ── */
.topbar{{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;}}
.logo{{font-size:.72em;font-weight:700;letter-spacing:.28em;color:var(--accent);}}
.badge{{display:inline-flex;align-items:center;gap:5px;font-size:.65em;font-weight:600;
  letter-spacing:.1em;color:var(--accent);background:var(--acc-l);
  border:1px solid #ddc8d8;border-radius:100px;padding:3px 10px;}}
.badge-dot{{width:5px;height:5px;border-radius:50%;background:var(--green);
  box-shadow:0 0 5px var(--green);animation:blink 2s ease infinite;}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}

/* ── card ── */
.card{{background:var(--white);border-radius:18px;
  box-shadow:0 2px 24px rgba(60,10,55,.08),0 1px 3px rgba(60,10,55,.04);
  overflow:hidden;}}

/* ── panel split ── */
.split{{display:grid;grid-template-columns:1fr 1fr;}}

/* ── left: QR ── */
.left{{
  background:linear-gradient(160deg,#1a0820 0%,#3a1035 50%,#5c1a50 100%);
  padding:28px 22px 24px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:14px;position:relative;overflow:hidden;
}}
.left::before{{content:'';position:absolute;top:-60px;left:-60px;
  width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,rgba(156,87,137,.35),transparent 70%);}}
.left-tag{{font-size:.62em;font-weight:600;letter-spacing:.14em;
  color:rgba(255,255,255,.45);text-transform:uppercase;}}
#qrimg{{
  width:160px;height:160px;border-radius:10px;
  box-shadow:0 4px 20px rgba(0,0,0,.5),0 0 0 1px rgba(255,255,255,.08);
  display:block;
}}
.qr-missing{{
  width:160px;height:160px;border-radius:10px;
  background:rgba(255,255,255,.06);border:1px dashed rgba(255,255,255,.15);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  font-size:.75em;color:rgba(255,255,255,.45);text-align:center;line-height:1.6;gap:8px;
}}
.scan-label{{font-size:.62em;color:rgba(255,255,255,.38);text-align:center;line-height:1.5;}}

/* ── right: steps ── */
.right{{padding:24px 22px;display:flex;flex-direction:column;justify-content:center;gap:0;}}
.right-head{{margin-bottom:16px;}}
.right-title{{font-size:.98em;font-weight:700;color:var(--ink);line-height:1.3;}}
.right-sub{{font-size:.72em;color:var(--muted);margin-top:4px;line-height:1.5;}}
.right-sub strong{{color:var(--accent);font-weight:600;}}

.step{{display:flex;align-items:flex-start;gap:9px;padding:7px 0;
  border-bottom:1px solid var(--line);}}
.step:last-child{{border-bottom:none;}}
.sn{{width:20px;height:20px;border-radius:6px;background:var(--accent);
  color:#fff;font-size:.66em;font-weight:700;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;margin-top:1px;}}
.st{{font-size:.72em;color:var(--ink2);line-height:1.55;}}
.st strong{{color:var(--ink);font-weight:600;}}
.st em{{color:var(--accent);font-style:normal;font-weight:500;}}

/* ── bottom section ── */
.bottom{{padding:20px 24px;border-top:1px solid var(--line);}}

.key-label{{display:flex;align-items:center;gap:8px;
  font-size:.65em;font-weight:600;letter-spacing:.12em;
  color:var(--muted);text-transform:uppercase;margin-bottom:10px;}}
.key-label span{{flex:1;height:1px;background:var(--line);}}

.key-box{{
  background:#12071a;border-radius:10px;
  padding:13px 16px;
  font-family:var(--mono);font-size:.82em;
  letter-spacing:.15em;color:#c8a8d8;
  text-align:center;line-height:1.8;
  word-break:break-all;cursor:copy;user-select:all;
  border:1px solid rgba(156,87,137,.18);
  transition:background .2s,border-color .2s;
  position:relative;
}}
.key-box:hover{{background:#1a0a26;border-color:rgba(156,87,137,.38);}}
.key-box::after{{
  content:'COPIED!';position:absolute;inset:0;
  display:flex;align-items:center;justify-content:center;
  border-radius:10px;background:#1BA0E2;
  color:#fff;font-family:var(--sans);font-size:.8em;font-weight:700;letter-spacing:.12em;
  opacity:0;transition:opacity .15s;pointer-events:none;
}}
.key-box.copied::after{{opacity:1;}}

.key-hint{{text-align:center;font-size:.62em;color:var(--muted);margin-top:6px;}}

/* ── save-hint collapsible ── */
.save-details{{margin-top:14px;}}
.save-summary{{
  display:flex;align-items:center;gap:7px;
  font-size:.67em;font-weight:500;color:var(--muted);
  cursor:pointer;list-style:none;user-select:none;
  padding:6px 0;
}}
.save-summary::-webkit-details-marker{{display:none;}}
.save-summary::before{{
  content:'▶';font-size:.7em;color:var(--accent);
  transition:transform .2s;display:inline-block;
}}
details[open] .save-summary::before{{transform:rotate(90deg);}}
.save-body{{
  margin-top:8px;background:#0f0714;border-radius:8px;
  padding:12px 14px;font-family:var(--mono);font-size:.68em;
  color:#a890b8;line-height:1.8;border:1px solid rgba(156,87,137,.15);
  animation:fadeIn .2s ease;
}}
.save-body .cm{{color:#5a8a5a;}}
.save-body .key{{color:#1BA0E2;}}
.save-body .val{{color:#c8a8d8;}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(-4px)}}to{{opacity:1;transform:none}}}}

</style>
</head>
<body>
<div class="w">

  <!-- topbar -->
  <div class="topbar">
    <span class="logo">MTRAX</span>
    <span class="badge"><span class="badge-dot"></span>SETUP 2FA</span>
  </div>

  <div class="card">

    <!-- split panel -->
    <div class="split">

      <!-- LEFT — QR -->
      <div class="left">
        <span class="left-tag">Scan QR Code</span>
        {qr_html}
        {qr_note}
        <p class="scan-label">Google Authenticator<br>Authy · Microsoft Auth</p>
      </div>

      <!-- RIGHT — Steps -->
      <div class="right">
        <div class="right-head">
          <div class="right-title">Aktivasi 2FA</div>
          <div class="right-sub">Halo <strong>{username}</strong>,<br>ikuti langkah berikut.</div>
        </div>

        <div class="step">
          <div class="sn">1</div>
          <div class="st">Buka <strong>Google Authenticator</strong> atau Authy.</div>
        </div>
        <div class="step">
          <div class="sn">2</div>
          <div class="st">Ketuk <strong>"+"</strong> → <em>Scan QR</em> atau <em>Enter key</em>.</div>
        </div>
        <div class="step">
          <div class="sn">3</div>
          <div class="st">Nama akun: <strong>admin@MTRAX</strong>, tipe: <em>Time-based</em>.</div>
        </div>
        <div class="step">
          <div class="sn">4</div>
          <div class="st">Masukkan <strong>6 digit kode</strong> dari app di bawah.</div>
        </div>
      </div>

    </div>

    <!-- BOTTOM — secret key + hidden save hint -->
    <div class="bottom">

      <div class="key-label"><span></span>atau gunakan kunci manual<span></span></div>
      <div class="key-box" id="kbox" onclick="copyKey(this)" title="Klik untuk copy">{fmt_secret}</div>
      <div class="key-hint" id="khint">Klik untuk menyalin kunci</div>

      <!-- Hidden: cara simpan secret -->
      <details class="save-details">
        <summary class="save-summary">Cara menyimpan secret agar tidak reset saat restart</summary>
        <div class="save-body">
          <span class="cm"># Tambahkan ke .streamlit/secrets.toml:</span><br>
          <span class="key">[totp]</span><br>
          <span class="key">{username}</span> = <span class="val">"{secret}"</span>
        </div>
      </details>

    </div>

  </div>
</div>

<script>
function copyKey(el) {{
  const text = el.textContent.replace(/\\s+/g,'').trim();
  navigator.clipboard.writeText(text).then(function() {{
    el.classList.add('copied');
    document.getElementById('khint').textContent = '✓ Tersalin ke clipboard!';
    setTimeout(function() {{
      el.classList.remove('copied');
      document.getElementById('khint').textContent = 'Klik untuk menyalin kunci';
    }}, 1800);
  }}).catch(function() {{
    document.getElementById('khint').textContent = 'Pilih semua teks lalu Ctrl+C';
  }});
}}
</script>
</body>
</html>"""

    _components.html(html_content, height=540, scrolling=False)

    # ── OTP input styling ──────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    div[data-testid="stTextInput"] > label {
        font-family: 'Sora', sans-serif !important;
        font-size: 0.75em !important;
        font-weight: 600 !important;
        color: #8a7a8a !important;
        letter-spacing: 0.10em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border: 1.5px solid #cce4f4 !important;
        font-size: 1.4em !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.40em !important;
        text-align: center !important;
        padding: 14px 12px !important;
        background: #f0f8ff !important;
        color: #3a1a3a !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #1BA0E2 !important;
        box-shadow: 0 0 0 3px rgba(156,87,137,0.12) !important;
        background: #ffffff !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: #d0b8d0 !important; letter-spacing: 0.30em !important; }
    </style>
    """, unsafe_allow_html=True)

    otp_input = st.text_input(
        "Kode OTP dari Google Authenticator",
        placeholder="● ● ● ● ● ●",
        max_chars=6,
        key="setup_otp_field"
    )

    col_v, col_c = st.columns([3, 2])
    with col_v:
        if st.button("✅  Verifikasi & Masuk", use_container_width=True, type="primary"):
            if _totp_valid(secret, otp_input):
                st.session_state.totp_enrolled[username] = True
                _finish_login(username)
            else:
                st.error("❌ Kode salah atau sudah kedaluwarsa. Coba lagi.")
    with col_c:
        if st.button("← Kembali", use_container_width=True):
            st.session_state.pending_2fa  = False
            st.session_state.pending_user = ""
            st.rerun()


def twofa_verify_page(username: str):
    """Halaman verifikasi OTP untuk login berikutnya — premium redesign."""
    import streamlit.components.v1 as _components
    st.set_page_config(page_title="Verifikasi 2FA | MTRAX", layout="centered")

    totp_secrets = _load_totp_secrets()
    secret       = totp_secrets[username]
    remaining    = 30 - (int(time.time()) % 30)
    progress_pct = int((remaining / 30) * 100)

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: 'DM Sans', sans-serif;
  background: #eef6fc;
  min-height: 100vh;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 24px 16px;
}}

.shell {{
  width: 100%;
  max-width: 440px;
  animation: fadeUp 0.45s cubic-bezier(0.22,1,0.36,1) both;
}}

@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(18px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

.brand-bar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}}
.brand-name {{
  font-size: 0.78em; font-weight: 700;
  letter-spacing: 0.22em; text-transform: uppercase;
  color: #1BA0E2;
}}
.step-pill {{
  background: #e6f4fb; color: #1BA0E2;
  font-size: 0.70em; font-weight: 600;
  letter-spacing: 0.08em;
  padding: 4px 12px; border-radius: 100px;
}}

.card {{
  background: #ffffff;
  border-radius: 20px;
  box-shadow: 0 4px 32px rgba(100,40,90,0.10), 0 1px 4px rgba(100,40,90,0.06);
  overflow: hidden;
}}

.card-header {{
  background: linear-gradient(135deg, #062440 0%, #0D7FCC 55%, #1BA0E2 100%);
  padding: 28px 32px 24px;
  position: relative;
  overflow: hidden;
}}
.card-header::before {{
  content: '';
  position: absolute; top: -50px; right: -50px;
  width: 180px; height: 180px; border-radius: 50%;
  background: rgba(255,255,255,0.06);
}}
.header-badge {{
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 100px; padding: 5px 14px;
  font-size: 0.72em; font-weight: 600;
  color: rgba(255,255,255,0.88);
  letter-spacing: 0.06em; margin-bottom: 14px;
}}
.header-badge .dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: #a8ff78;
  box-shadow: 0 0 6px #a8ff78;
  animation: blink 2s ease infinite;
}}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.35}} }}

.header-title {{
  font-size: 1.25em; font-weight: 700; color: #fff;
  margin-bottom: 6px; position: relative; z-index: 1;
}}
.header-sub {{
  font-size: 0.80em; color: rgba(255,255,255,0.68);
  line-height: 1.6; position: relative; z-index: 1;
}}
.header-sub strong {{ color: rgba(255,255,255,0.95); }}
.header-sub em {{ color: #e0b8d8; font-style: normal; font-weight: 500; }}

/* ── Icon lock ─────────────────────────────── */
.lock-wrap {{
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 14px;
}}
.lock-circle {{
  width: 64px; height: 64px; border-radius: 20px;
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.25);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.8em;
}}

.card-body {{
  padding: 28px 32px;
}}

/* ── Timer ─────────────────────────────────── */
.timer-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #f0f8ff;
  border: 1px solid #cce4f4;
  border-radius: 12px;
  padding: 12px 18px;
  margin-bottom: 20px;
  gap: 12px;
}}
.timer-label {{
  font-size: 0.76em;
  color: #6a8fa0;
  font-weight: 500;
}}
.timer-val {{
  font-family: 'DM Mono', monospace;
  font-size: 1.0em;
  font-weight: 600;
  color: #1BA0E2;
  min-width: 28px;
  text-align: right;
}}
.timer-bar-wrap {{
  flex: 1;
  height: 4px;
  background: #cce4f4;
  border-radius: 2px;
  overflow: hidden;
}}
.timer-bar {{
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, #1BA0E2, #75caf0);
  width: {progress_pct}%;
  transition: width 1s linear;
}}

/* ── Hint ──────────────────────────────────── */
.hint-box {{
  display: flex;
  align-items: flex-start;
  gap: 10px;
  background: #eef6fc;
  border-radius: 10px;
  padding: 12px 16px;
}}
.hint-icon {{
  font-size: 1.1em;
  flex-shrink: 0;
  margin-top: 1px;
}}
.hint-text {{
  font-size: 0.78em;
  color: #6a5a6a;
  line-height: 1.6;
}}
.hint-text strong {{ color: #3a1a3a; }}
.hint-text em {{ color: #1BA0E2; font-style: normal; font-weight: 500; }}

</style>
</head>
<body>
<div class="shell">

  <div class="brand-bar">
    <span class="brand-name">MTRAX</span>
    <span class="step-pill">VERIFIKASI 2FA</span>
  </div>

  <div class="card">
    <div class="card-header">
      <div class="lock-wrap">
        <div class="lock-circle">🔐</div>
      </div>
      <div class="header-badge"><span class="dot"></span>Autentikasi Dua Langkah</div>
      <div class="header-title">Masukkan Kode OTP</div>
      <div class="header-sub">
        Halo <strong>{username}</strong>, password sudah benar!<br>
        Buka <strong>Google Authenticator</strong> dan masukkan kode 6 digit untuk akun <em>MTRAX:{username}</em>.
      </div>
    </div>

    <div class="card-body">
      <!-- Timer -->
      <div class="timer-row">
        <span class="timer-label">⏱ Kode berikutnya dalam</span>
        <div class="timer-bar-wrap"><div class="timer-bar"></div></div>
        <span class="timer-val">{remaining}s</span>
      </div>

      <!-- Hint -->
      <div class="hint-box">
        <span class="hint-icon">💡</span>
        <div class="hint-text">
          Buka <strong>Google Authenticator</strong> &rarr; cari akun <em>MTRAX:{username}</em> &rarr; masukkan <strong>6 digit kode</strong> yang ditampilkan.
        </div>
      </div>
    </div>
  </div>

</div>
</body>
</html>"""

    _components.html(html_content, height=420, scrolling=False)

    st.markdown("""
    <style>
    div[data-testid="stTextInput"] label {
        font-size: 0.82em !important;
        font-weight: 600 !important;
        color: #5a4a5a !important;
        letter-spacing: 0.04em;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border: 1.5px solid #cce4f4 !important;
        font-size: 1.3em !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.35em !important;
        text-align: center !important;
        padding: 14px !important;
        background: #f0f8ff !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #1BA0E2 !important;
        box-shadow: 0 0 0 3px rgba(156,87,137,0.12) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    otp_input = st.text_input(
        "Kode OTP (6 digit)",
        placeholder="● ● ● ● ● ●",
        max_chars=6,
        key="verify_otp_field"
    )

    col_v, col_c = st.columns([3, 2])
    with col_v:
        if st.button("✅  Verifikasi & Masuk", use_container_width=True, type="primary"):
            if _totp_valid(secret, otp_input):
                _finish_login(username)
            else:
                st.error("❌ Kode salah atau sudah kedaluwarsa. Coba lagi.")
                time.sleep(1)
    with col_c:
        if st.button("← Kembali", use_container_width=True):
            st.session_state.pending_2fa  = False
            st.session_state.pending_user = ""
            st.rerun()


def _finish_login(username: str):
    """Selesaikan proses login setelah OTP berhasil diverifikasi."""
    st.session_state.login_attempts = 0
    st.session_state.lockout_until  = 0
    st.session_state.authenticated  = True
    st.session_state.logged_in      = True
    st.session_state.username       = username
    st.session_state.role           = USERS[username]["role"]
    st.session_state.login_time     = time.time()
    st.session_state.pending_2fa    = False
    st.session_state.pending_user   = ""
    # Simpan token sesi supaya login tetap bertahan saat halaman di-reload (F5).
    _token = _create_session_token(username, USERS[username]["role"])
    st.session_state._session_token = _token
    # Jalur utama & reliable: simpan token di URL lewat st.query_params — ini
    # murni server-side (Python), tidak butuh JS/komponen sama sekali, dan
    # query string SELALU dipertahankan browser saat halaman di-reload (F5).
    try:
        st.query_params[_SESSION_QUERY_KEY] = _token
    except Exception:
        pass
    # Cookie tetap ditulis sebagai lapisan tambahan (mis. kalau URL berubah).
    _set_session_cookie(_token)
    st.success("✅ Login berhasil! Selamat datang.")
    # `st.session_state.authenticated` sudah True di atas, jadi `st.rerun()`
    # ini SUDAH CUKUP untuk langsung masuk ke dashboard dalam sesi yang sama —
    # tidak bergantung pada cookie/JS/query param sama sekali. Jeda singkat
    # sebelum rerun hanya untuk memberi waktu iframe cookie sempat dimuat.
    time.sleep(0.6)
    st.rerun()


# ======================================
# LOGIN FUNCTION (SECURE + ANTI BRUTE FORCE)
# ── DIMODIFIKASI: setelah password OK → redirect ke 2FA ──
# ======================================
def login_page():
    st.set_page_config(
        page_title="Login | MTRAX",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    current_time = time.time()
    if current_time < st.session_state.lockout_until:
        remaining_lock = int(st.session_state.lockout_until - current_time)
        st.error(f"Too many failed attempts. Try again in {remaining_lock} seconds.")
        st.stop()

    # Load logo
    def _b64_img(p):
        try:
            with open(p, "rb") as f: return base64.b64encode(f.read()).decode()
        except Exception: return ""

    logo_b64 = _b64_img(os.path.join("assets", "LOGO M FIX.png"))
    logo_tag = (
        f'<img src="data:image/png;base64,{logo_b64}" '
        f'style="width:44px;display:block;margin:0 auto 16px;" />'
        if logo_b64 else
        '<div style="width:44px;height:44px;border-radius:9px;'
        'background:linear-gradient(135deg,#1BA0E2,#0D7FCC);'
        'display:flex;align-items:center;justify-content:center;margin:0 auto 16px;'
        'font-size:0.9em;font-weight:800;color:#ffffff;letter-spacing:.04em;">MTX</div>'
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Roboto+Mono:wght@400;500&display=swap');

    /* ── RESET & BASE ── */
    *, *::before, *::after { font-family:'Inter',sans-serif !important; box-sizing:border-box; margin:0; padding:0; }
    html, body { height:100%; overflow:hidden; }

    /* ── FIX: kembalikan font Material Symbols untuk ikon bawaan Streamlit
         (mis. tombol show/hide password) — tanpa ini, wildcard font-family di atas
         merusak ligature ikon sehingga muncul teks literal "visibility" alih-alih ikon mata ── */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
        font-size: 1.2rem !important;
    }

    .stApp                              { background:#0D7FCC !important; overflow:hidden; }
    [data-testid="stHeader"]            { display:none !important; }
    [data-testid="stToolbar"]           { display:none !important; }
    [data-testid="stDecoration"]        { display:none !important; }
    [data-testid="stSidebar"]           { display:none !important; }
    [data-testid="stAppViewContainer"]  { padding:0 !important; overflow:hidden !important; }
    section[data-testid="stMain"]       { overflow:hidden !important; }
    #MainMenu, footer, header           { display:none !important; }
    .block-container                    { padding:0 !important; max-width:100% !important; overflow:hidden !important; }
    ::-webkit-scrollbar                 { display:none !important; }

    /* ── HIDE DUMMY LEFT COL ── */
    [data-testid="stHorizontalBlock"]                        { gap:0 !important; overflow:hidden !important; }
    [data-testid="stHorizontalBlock"] > div                  { padding:0 !important; }
    [data-testid="stHorizontalBlock"] > div:first-child      { display:none !important; }

    /* ══════════════════════════════════════════════════════
       CANVAS — full screen gradient background
    ══════════════════════════════════════════════════════ */
    .lp-canvas {
        position: fixed; top:0; left:0;
        width: 100%; height: 100vh; z-index: 1;
        background: linear-gradient(135deg,
            #e2f871 0%,
            #c8ec50 12%,
            #8dd4a8 26%,
            #4ab8d4 38%,
            #1BA0E2 50%,
            #1494C6 65%,
            #0D7FCC 80%,
            #07395f 100%
        );
        overflow: hidden;
    }

    /* animated mesh radial blobs */
    .lp-canvas::before {
        content:'';
        position:absolute; inset:0;
        background:
            radial-gradient(ellipse at 8% 10%,  rgba(226,248,113,0.55) 0%, transparent 38%),
            radial-gradient(ellipse at 90% 5%,  rgba(13,127,204,0.45)  0%, transparent 40%),
            radial-gradient(ellipse at 75% 90%, rgba(20,148,198,0.40)  0%, transparent 45%),
            radial-gradient(ellipse at 15% 88%, rgba(226,248,113,0.30) 0%, transparent 35%);
        animation: lp-meshShift 12s ease-in-out infinite alternate;
    }
    @keyframes lp-meshShift {
        0%   { opacity:1; transform:scale(1) translateX(0); }
        100% { opacity:0.85; transform:scale(1.04) translateX(12px); }
    }

    /* dot grid overlay */
    .lp-canvas::after {
        content:''; position:absolute; inset:0; pointer-events:none;
        background-image: radial-gradient(circle, rgba(255,255,255,0.18) 1px, transparent 1px);
        background-size: 30px 30px;
        -webkit-mask-image: linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.1) 60%, transparent 100%);
        mask-image: linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.1) 60%, transparent 100%);
    }

    /* ── BLOBS ── */
    .lp-blob {
        position:absolute; border-radius:50%; pointer-events:none;
    }
    .lp-blob-1 {
        width:520px; height:520px; top:-160px; left:-140px;
        background: radial-gradient(circle, rgba(226,248,113,0.40) 0%, transparent 68%);
        animation: lp-floatA 8s ease-in-out infinite alternate;
    }
    .lp-blob-2 {
        width:480px; height:480px; top:30%; right:-100px;
        background: radial-gradient(circle, rgba(13,127,204,0.32) 0%, transparent 65%);
        animation: lp-floatB 10s ease-in-out infinite alternate;
    }
    .lp-blob-3 {
        width:380px; height:380px; bottom:-120px; left:20%;
        background: radial-gradient(circle, rgba(7,57,95,0.28) 0%, transparent 60%);
        animation: lp-floatA 9s ease-in-out infinite alternate-reverse;
    }
    .lp-blob-4 {
        width:260px; height:260px; top:-60px; right:18%;
        background: radial-gradient(circle, rgba(200,236,80,0.28) 0%, transparent 65%);
        animation: lp-floatB 7s ease-in-out infinite alternate;
    }
    @keyframes lp-floatA {
        0%   { transform:translate(0,0) scale(1); }
        100% { transform:translate(20px,30px) scale(1.06); }
    }
    @keyframes lp-floatB {
        0%   { transform:translate(0,0) scale(1); }
        100% { transform:translate(-15px,20px) scale(0.95); }
    }

    /* ── RINGS ── */
    .lp-ring {
        position:absolute; border-radius:50%; pointer-events:none;
    }
    .lp-ring-1 { width:600px; height:600px; top:-180px; left:-180px; border:1.5px solid rgba(255,255,255,0.12); }
    .lp-ring-2 { width:380px; height:380px; top:60px; left:60px; border:1px solid rgba(255,255,255,0.07); }
    .lp-ring-3 { width:320px; height:320px; bottom:-80px; right:10%; border:1px solid rgba(255,255,255,0.08); }
    .lp-ring-4 { width:180px; height:180px; top:15%; right:28%; border:1px solid rgba(226,248,113,0.25); }

    /* ── PARTICLES ── */
    .lp-particle { position:absolute; border-radius:50%; pointer-events:none; }
    .lp-p1 { width:5px; height:5px; background:#e2f871; top:12%; left:8%;
              box-shadow:0 0 14px rgba(226,248,113,0.8); animation:lp-pulse 3s ease-in-out infinite; }
    .lp-p2 { width:4px; height:4px; background:#fff; top:6%; right:12%;
              opacity:0.6; animation:lp-pulse 4s ease-in-out infinite 1s; }
    .lp-p3 { width:3px; height:3px; background:#e2f871; bottom:22%; left:18%;
              box-shadow:0 0 10px rgba(226,248,113,0.6); animation:lp-pulse 5s ease-in-out infinite 0.5s; }
    .lp-p4 { width:6px; height:6px; background:rgba(255,255,255,0.5); bottom:35%; right:18%;
              animation:lp-pulse 3.5s ease-in-out infinite 2s; }
    .lp-p5 { width:3px; height:3px; background:#1BA0E2; top:45%; left:38%;
              box-shadow:0 0 10px rgba(27,160,226,0.7); animation:lp-pulse 4s ease-in-out infinite 1.5s; }
    @keyframes lp-pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50%     { opacity:0.35; transform:scale(1.5); }
    }

    /* ── LEFT CONTENT: brand + hero ── */
    .lp-inner {
        position:fixed; z-index:10;
        top:0; left:0; bottom:0; width:50%;
        display:flex; flex-direction:column;
        justify-content:space-between;
        padding:40px 52px 48px;
    }
    .lp-brand { display:flex; align-items:center; gap:12px; }
    .lp-brand-icon {
        width:36px; height:36px; border-radius:8px;
        background:rgba(255,255,255,0.18);
        border:1px solid rgba(255,255,255,0.28);
        display:flex; align-items:center; justify-content:center;
    }
    .lp-brand-icon-sq {
        display:block; width:10px; height:10px;
        background:white; border-radius:1px; opacity:0.95;
    }
    .lp-brand-name { font-size:0.80em; font-weight:800; color:#fff; letter-spacing:0.20em;
                     text-shadow:0 1px 8px rgba(0,0,0,0.18); }
    .lp-divider {
        width:34px; height:3px;
        background:linear-gradient(90deg, #e2f871 0%, rgba(226,248,113,0.30) 100%);
        border-radius:2px; margin-bottom:18px;
        box-shadow:0 0 12px rgba(226,248,113,0.55);
    }
    .lp-headline {
        font-size:2.40em; font-weight:900; color:#fff;
        line-height:1.15; letter-spacing:-0.02em; margin-bottom:16px;
        text-shadow:0 2px 20px rgba(0,0,0,0.20);
    }
    .lp-sub { font-size:0.80em; color:rgba(255,255,255,0.60); font-weight:400; line-height:1.7; }

    /* ── COPYRIGHT ── */
    .lp-copy {
        position:fixed; bottom:20px; right:28px; z-index:10;
        font-size:0.60em; color:rgba(255,255,255,0.40);
        font-weight:500; letter-spacing:0.04em;
    }
    .lp-copy a { color:inherit; text-decoration:none; }
    .lp-copy a:hover { color:rgba(226,248,113,0.70); }

    /* ── FLOATING LOGIN CARD — glass white ── */
    [data-testid="stHorizontalBlock"] > div:last-child {
        position: fixed !important;
        top: 50% !important;
        left: 72% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 1000 !important;
        width: 340px !important;
        max-width: 340px !important;
        background: rgba(255,255,255,0.96) !important;
        border-radius: 20px !important;
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.30),
            0 8px 32px rgba(7,57,95,0.22),
            0 32px 80px rgba(7,57,95,0.16) !important;
        padding: 40px 36px 36px !important;
        border: none !important;
        overflow: visible !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
    }

    /* shimmer top border on card */
    [data-testid="stHorizontalBlock"] > div:last-child::before {
        content:'';
        position:absolute; top:0; left:20px; right:20px; height:1px;
        background:linear-gradient(90deg,
            transparent 0%, rgba(226,248,113,0.60) 30%,
            rgba(27,160,226,0.40) 70%, transparent 100%);
        border-radius:1px;
    }

    /* ── INPUT FIELDS ── */
    .stTextInput label { display:none !important; }
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1.5px solid #dde5ee !important;
        background: #f7fafd !important;
        padding: 12px 14px !important;
        font-size: 0.86em !important;
        color: #1a2a3a !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    .stTextInput > div > div > input::placeholder { color: #b0bec8 !important; }
    .stTextInput > div > div > input:focus {
        border-color: #1BA0E2 !important;
        box-shadow: 0 0 0 3px rgba(27,160,226,0.13) !important;
        background: #ffffff !important;
        outline: none !important;
    }
    .stTextInput > div > div > div > button {
        background: transparent !important; border: none !important; color: #b0bec8 !important;
        display: inline-flex !important; align-items: center !important; justify-content: center !important;
        min-width: 32px !important; height: 32px !important; padding: 0 !important;
    }
    .stTextInput > div > div > div > button:hover { color: #0D7FCC !important; }

    /* ── LOG IN BUTTON — lime gradient ── */
    /* (mencakup stButton dan stFormSubmitButton — form dipakai agar tombol Enter bisa submit login) */
    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #e2f871 0%, #cce84a 50%, #b8d930 100%) !important;
        color: #07395f !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 13px !important;
        font-weight: 900 !important;
        font-size: 0.78em !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all .18s ease !important;
        margin-top: 6px !important;
        box-shadow: 0 4px 18px rgba(226,248,113,0.55), 0 1px 4px rgba(7,57,95,0.12) !important;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #d4e860 0%, #bdd938 50%, #a8c820 100%) !important;
        box-shadow: 0 6px 24px rgba(226,248,113,0.70), 0 2px 8px rgba(7,57,95,0.15) !important;
        transform: translateY(-1.5px) !important;
    }
    div[data-testid="stButton"] > button:active,
    div[data-testid="stFormSubmitButton"] > button:active { transform: translateY(0) !important; }

    /* Streamlit's st.form menambahkan border bawaan di sekeliling form — dihilangkan
       supaya tetap menyatu dengan desain kartu login yang sudah ada */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    .stAlert { border-radius: 8px !important; margin-top: 8px !important; font-size: 0.78em !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── RENDER BACKGROUND CANVAS
    st.markdown("""
    <div class="lp-canvas">
        <div class="lp-blob lp-blob-1"></div>
        <div class="lp-blob lp-blob-2"></div>
        <div class="lp-blob lp-blob-3"></div>
        <div class="lp-blob lp-blob-4"></div>
        <div class="lp-ring lp-ring-1"></div>
        <div class="lp-ring lp-ring-2"></div>
        <div class="lp-ring lp-ring-3"></div>
        <div class="lp-ring lp-ring-4"></div>
        <div class="lp-particle lp-p1"></div>
        <div class="lp-particle lp-p2"></div>
        <div class="lp-particle lp-p3"></div>
        <div class="lp-particle lp-p4"></div>
        <div class="lp-particle lp-p5"></div>
    </div>
    <div class="lp-inner">
        <div class="lp-brand">
            <div class="lp-brand-icon"><span class="lp-brand-icon-sq"></span></div>
            <span class="lp-brand-name">MTRAX</span>
        </div>
        <div>
            <div class="lp-divider"></div>
            <div class="lp-headline">Welcome to<br><b>MTRAX</b><br>Analytics</div>
            <div class="lp-sub">Corporate Travel Analytics Platform<br>for Pertamina Group</div>
        </div>
    </div>
    <div class="lp-copy"><a href="https://www.linkedin.com/in/rifyalt/" target="_blank">© 2025 MTRAX.</a> All rights reserved.</div>
    """, unsafe_allow_html=True)

    # ── FLOATING CARD: dummy col (hidden) + form col
    _dummy, col_form = st.columns(2)
    with col_form:
        st.markdown(f"""
        {logo_tag}
        <div style="text-align:center;font-size:1.15em;font-weight:800;color:#1a1a1a;
             letter-spacing:0.01em;margin-bottom:5px;">Welcome</div>
        <div style="text-align:center;font-size:0.70em;color:#666;margin-bottom:28px;
             font-weight:400;letter-spacing:0.02em;">Log in to your account</div>
        <div style="font-size:0.70em;font-weight:700;color:#07395f;
             letter-spacing:0.10em;text-transform:uppercase;margin-bottom:6px;">Username</div>
        """, unsafe_allow_html=True)

        # Dibungkus st.form supaya menekan Enter di kolom Username/Password
        # langsung men-submit login (form_submit_button otomatis ter-trigger oleh Enter),
        # tanpa harus mengklik tombol LOG IN secara manual.
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("_u", placeholder="Enter Username ...",
                                     label_visibility="collapsed", key="login_username")

            st.markdown("""
            <div style="font-size:0.70em;font-weight:700;color:#07395f;
                 letter-spacing:0.10em;text-transform:uppercase;margin:14px 0 6px;">Password</div>
            """, unsafe_allow_html=True)

            password = st.text_input("_p", type="password", placeholder="Enter Password ...",
                                     label_visibility="collapsed", key="login_password")

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            submitted = st.form_submit_button("LOG IN", use_container_width=True)

        if submitted:
            if not validate_username(username):
                st.error("Invalid username format.")
                return
            if username in USERS:
                if secure_compare(password, USERS[username]["password"]):
                    st.session_state.login_attempts = 0
                    st.session_state.lockout_until  = 0
                    st.session_state.pending_2fa    = True
                    st.session_state.pending_user   = username
                    st.session_state.pending_role   = USERS[username]["role"]
                    st.rerun()
                else:
                    st.session_state.login_attempts += 1
            else:
                st.session_state.login_attempts += 1

            if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
                st.session_state.lockout_until  = time.time() + LOCKOUT_SECONDS
                st.session_state.login_attempts = 0
                st.error("Too many failed attempts. Account temporarily locked.")
            else:
                remaining_attempts = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                st.error(f"Login failed. {remaining_attempts} attempt(s) remaining.")

# ======================================
# MAIN APP
# ======================================
def main_app():
    """Main application"""
    st.set_page_config(
        page_title="MTRAX | Travel Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── BUG FIX: sliding expiry di server (_validate_session_token) memperpanjang
    # masa berlaku token, tapi cookie di BROWSER tetap memakai max-age lama sejak
    # login pertama. Kalau tidak disegarkan, cookie browser akan tetap kedaluwarsa
    # sesuai waktu login awal walau token server masih valid -> user bisa
    # ter-logout otomatis meski masih aktif memakai aplikasi. Segarkan cookie
    # (tanpa reload) setiap kali main_app dimuat, selama masih ada token sesi.
    if st.session_state.get("_session_token"):
        _set_session_cookie(st.session_state._session_token)

    # Custom CSS - MTRAX Theme v2 (Blue/Grey Palette)
    st.markdown("""
        <style>
        /* ============================================================
           MTRAX DESIGN SYSTEM
           Primary:  #1BA0E2  #1494C6  #0D7FCC
           Neutral:  #f0f0f0  #dedede  #434343
           Accent:   #98ea16  #d61b54  #ff5e1f  #ffdc00  #e2f871  #e13230
        ============================================================ */

        :root {
            --clr-primary:      #1BA0E2;
            --clr-primary-mid:  #1494C6;
            --clr-primary-dark: #0D7FCC;
            --clr-bg:           #f0f0f0;
            --clr-surface:      #ffffff;
            --clr-border:       #dedede;
            --clr-text-main:    #434343;
            --clr-text-muted:   #7a8a96;
            --clr-accent-green: #98ea16;
            --clr-accent-pink:  #d61b54;
            --clr-accent-orange:#ff5e1f;
            --clr-accent-yellow:#ffdc00;
            --clr-accent-lime:  #e2f871;
            --clr-accent-red:   #e13230;
        }

        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica', sans-serif;
        }

        /* ── Proteksi tambahan: pastikan ikon Material Symbols bawaan Streamlit
             (selectbox, multiselect, date picker, expander, dsb.) tidak ikut
             ter-override font-family-nya oleh rule '*' di atas ── */
        [data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
            font-weight: normal !important;
            font-style: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-feature-settings: 'liga' !important;
            font-feature-settings: 'liga' !important;
            -webkit-font-smoothing: antialiased !important;
        }

        .stApp {
            background: var(--clr-bg);
        }

        #MainMenu, footer, header { visibility: hidden; }

        /* ── NEWS TICKER ─────────────────────────────── */
        .news-ticker {
            background: linear-gradient(90deg, #d71c58 0%, #d71c58 100%);
            color: white;
            padding: 8px 0;
            font-size: 0.85em;
            overflow: hidden;
            white-space: nowrap;
            position: relative;
            margin: -60px -60px 0 -60px;
        }

        .news-ticker:hover .ticker-content {
            animation-play-state: paused;
        }

        .ticker-content {
            display: inline-block;
            padding-left: 100%;
            animation: ticker 70s linear infinite;
        }

        .news-item {
            color: #ffffff !important;
            text-decoration: none;
            margin-right: 40px;
            cursor: pointer;
            transition: opacity 0.2s ease;
        }

        .news-item:hover {
            opacity: 0.8;
            text-decoration: underline;
        }

        @keyframes ticker {
            0%   { transform: translateX(0); }
            100% { transform: translateX(-100%); }
        }

        /* ── HEADER ──────────────────────────────────── */
        .header {
            background: var(--clr-surface);
            padding: 25px 40px;
            border-bottom: 2px solid var(--clr-primary);
            margin: 0 -60px 30px -60px;
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title {
            font-size: 1.8em;
            font-weight: 700;
            color: var(--clr-primary-dark);
            letter-spacing: -0.02em;
        }

        .header-subtitle {
            font-size: 0.9em;
            color: var(--clr-text-muted);
            margin-top: 4px;
        }

        .header-user { text-align: right; }

        .user-name {
            font-size: 0.95em;
            color: var(--clr-text-main);
            font-weight: 500;
        }

        .user-role {
            font-size: 0.85em;
            color: var(--clr-text-muted);
            margin-top: 2px;
        }

        /* ── GLOBAL FILTER BAR ───────────────────────── */
        .global-filter-bar {
            background: var(--clr-surface);
            border: 1px solid var(--clr-border);
            border-left: 4px solid var(--clr-primary);
            border-radius: 6px;
            padding: 14px 20px 10px 20px;
            margin-bottom: 20px;
        }
        .global-filter-label {
            font-size: 0.72em;
            font-weight: 700;
            color: var(--clr-primary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }

        /* ── METRIC BOX ──────────────────────────────── */
        .metric-box {
            background: var(--clr-surface);
            padding: 20px;
            border-radius: 6px;
            border-left: 3px solid var(--clr-primary);
        }

        .metric-label {
            font-size: 0.8em;
            color: var(--clr-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 2em;
            font-weight: 600;
            color: var(--clr-text-main);
        }

        /* ── STATS CARDS ─────────────────────────────── */
        .stats-card {
            background: var(--clr-surface);
            padding: 25px;
            border-radius: 6px;
            text-align: center;
            height: 100%;
        }

        .stats-number {
            font-size: 2.5em;
            font-weight: 700;
            color: var(--clr-primary);
            margin: 15px 0;
        }

        .stats-label {
            font-size: 0.85em;
            color: var(--clr-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stats-detail {
            font-size: 0.9em;
            color: var(--clr-text-muted);
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid var(--clr-bg);
        }

        /* ── SECTION TITLE ───────────────────────────── */
        .section-title {
            font-size: 1.2em;
            font-weight: 600;
            color: var(--clr-text-main);
            margin: 30px 0 20px 0;
        }

        /* ── TABS ────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: var(--clr-surface);
            border-bottom: 2px solid var(--clr-border);
        }

        .stTabs [data-baseweb="tab"] {
            padding: 12px 20px;
            font-weight: 500;
            color: var(--clr-text-muted);
            border-bottom: 2px solid transparent;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--clr-primary);
        }

        .stTabs [aria-selected="true"] {
            color: var(--clr-primary);
            border-bottom-color: var(--clr-primary);
            background: transparent;
        }

        /* ── BUTTONS ─────────────────────────────────── */
        .stButton > button {
            border-radius: 4px;
            font-weight: 500;
            padding: 10px 20px;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1BA0E2 0%, #0D7FCC 100%);
            color: white;
            border: none;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #a6ce39 !important;  # ← ganti warna sesuai keinginan
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1494C6 0%, #0D7FCC 100%);
        }

        /* ── SIDEBAR ─────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: var(--clr-surface);
            border-right: 1px solid var(--clr-border);
        }

        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: var(--clr-primary-dark);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ── STREAMLIT METRIC ────────────────────────── */
        .stMetric {
            background: var(--clr-surface);
            padding: 18px;
            border-radius: 6px;
            border-left: 3px solid var(--clr-primary);
        }

        .stMetric label {
            color: var(--clr-text-muted) !important;
            font-size: 0.8em !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stMetric [data-testid="stMetricValue"] {
            color: var(--clr-text-main) !important;
            font-size: 1.8em !important;
        }

        /* ── PROGRESS ────────────────────────────────── */
        .stProgress > div > div {
            background: linear-gradient(90deg, #1BA0E2, #0D7FCC);
        }

        /* ── DIVIDER ─────────────────────────────────── */
        .divider {
            height: 1px;
            background: var(--clr-border);
            margin: 30px 0;
        }

        /* ── SELECTBOX ───────────────────────────────── */
        .stSelectbox > div > div {
            border-radius: 4px;
        }

        /* ── ACCENT BADGE HELPERS ────────────────────── */
        .badge-green  { background:#98ea16; color:#1e4000; padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-red    { background:#e13230; color:#fff;    padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-orange { background:#ff5e1f; color:#fff;    padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-yellow { background:#ffdc00; color:#433000; padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-pink   { background:#d61b54; color:#fff;    padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-blue   { background:#1BA0E2; color:#fff;    padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        </style>
    """, unsafe_allow_html=True)

    # NEWS TICKER
    news_data = fetch_rss_news()
    render_news_ticker(news_data)

    # HEADER
    greet, now = get_greeting()
    role_display = st.session_state.get("role", "User")
    username_display = st.session_state.get("username", "User")

    st.markdown(f"""
        <div class='header'>
            <div class='header-content'>
                <div>
                    <div class='header-title'>MTRAX</div>
                    <div class='header-subtitle'>Corporate Travel Analytics</div>
                </div>
                <div class='header-user'>
                    <div class='user-name'>{greet}, {username_display.title()}</div>
                    <div class='user-role'>{role_display} · {now.strftime('%d %b %Y')}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ======================================
    # SIDEBAR MODE STATE
    # ======================================
    if "sidebar_mode" not in st.session_state:
        st.session_state.sidebar_mode = "expanded"

    # ======================================
    # SIDEBAR
    # ======================================
    def load_logo_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    logo_base64 = load_logo_base64("assets/LOGO M FIX.png")
    # ===== LOGO =====
#    if logo_base64:
#        logo_size = 60 if st.session_state.sidebar_mode == "expanded" else 35

#        st.markdown(f"""
#            <div style='text-align:center;padding:10px 0 15px 0;'>
#                <img src='data:image/png;base64,{logo_base64}' width='{logo_size}'/>
#            </div>
#        """, unsafe_allow_html=True)

    # ======================================
    # SIDEBAR WIDTH CONTROL (DYNAMIC)
    # ======================================

    sidebar_width = 260 if st.session_state.sidebar_mode == "expanded" else 90

    st.markdown(f"""
    <style>

    /* Sidebar width */
    section[data-testid="stSidebar"] {{
        width: {sidebar_width}px !important;
        min-width: {sidebar_width}px !important;
        max-width: {sidebar_width}px !important;
        transition: all 0.25s ease-in-out;
    }}

    /* Hide labels in compact mode */
    {"section[data-testid='stSidebar'] label { display:none !important; }" if st.session_state.sidebar_mode == "compact" else ""}

    /* Hide select text in compact mode */
    {"section[data-testid='stSidebar'] .stSelectbox div div div { font-size:0 !important; }" if st.session_state.sidebar_mode == "compact" else ""}

    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # ===============================
        # TOGGLE BUTTON
        # ===============================
#        col1, col2 = st.columns([1,1])

#        with col1:
#            if st.button("☰", use_container_width=True):
#                if st.session_state.sidebar_mode == "expanded":
#                    st.session_state.sidebar_mode = "compact"
#                else:
#                    st.session_state.sidebar_mode = "expanded"
#                st.rerun()

#        with col2:
#            if st.session_state.sidebar_mode == "expanded":
#                st.markdown("<div style='font-size:12px;color:gray;padding-top:8px;'>Menu</div>", unsafe_allow_html=True)

#        st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)


        st.markdown(f"""
            <div style='text-align:center;padding:20px 0;'>
                <img src='data:image/png;base64,{logo_base64}' width='60'/>
            </div>
        """, unsafe_allow_html=True)

        # ── 2FA Status Badge di Sidebar ──────────────────────────────
        _uname_side   = st.session_state.get("username", "")
        _enrolled     = st.session_state.totp_enrolled.get(_uname_side, False)
        # BUG FIX: secret TOTP bersifat shared milik 'admin' (lihat _load_totp_secrets),
        # bukan per-username. Sebelumnya kode di sini mengecek
        # st.secrets["totp"][_uname_side] — hampir selalu False untuk user selain
        # 'admin' walau 2FA sebenarnya sudah aktif lewat secret bersama, sehingga
        # badge salah menampilkan "Setup Required" padahal 2FA sudah aktif.
        try:
            _from_toml = bool(st.secrets["totp"]["admin"])
        except Exception:
            _from_toml = False
        _2fa_active = _enrolled or _from_toml
        _badge_color = "#3dab7a" if _2fa_active else "#e05a2b"
        _badge_text  = "Active ✅" if _2fa_active else "Setup Required"
        st.markdown(f"""
        <div style='background:white;border-radius:6px;padding:10px 14px;
                    border-left:3px solid {_badge_color};margin-bottom:4px;
                    font-size:0.78em;'>
            <div style='color:#888;font-size:0.85em;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:3px;'>🔐 2FA Status</div>
            <div style='color:{_badge_color};font-weight:600;'>{_badge_text}</div>
            <div style='color:#bbb;font-size:0.88em;'>Google Authenticator · TOTP</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Drive options
        drive_options = {
            "2023": "1xDFRdGLDiiScIwW9gTucRyeFCmuqNyq_",
            "2024": "16ZMZ42BLN4GPbYKAd5h75ocbxFuyc85V",
            "2025": "1chxbGHfk9hHNPZ8vlU6AqRVUKH1jEnxF",
            "2026": "14CbafYeVrKUXWBE1LPUFlRXHeXGXAaO4",
        }
        
        selected_period = st.selectbox(
            "Select Data Period",
            options=list(drive_options.keys()),
            help="Choose which year's data to load"
        )
        
        if st.button("Get Data", use_container_width=True, type="primary"):
            with st.spinner(f"Loading {selected_period} data..."):
                progress_bar = st.progress(0)
                try:
                    folder_id = drive_options[selected_period]
                    drop_cols = ["Unnamed: 0", "Travel Request Number.1"]
                    
                    progress_bar.progress(20)
                    loaded_data = load_drive_data(folder_id, drop_cols)
                    
                    progress_bar.progress(100)
                    st.session_state.df_all = loaded_data
                    st.session_state.data_loaded = True
                    st.session_state.data_period = selected_period
                    
                    st.success(f"✅ Loaded {len(loaded_data):,} records")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        uploaded_files = st.file_uploader(
            "Upload Excel Files",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            help="Select multiple Excel files"
        )

        if uploaded_files:
            if st.button("Process Uploaded Files", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        dfs = []
                        drop_cols = ["Unnamed: 0", "Travel Request Number.1"]
                        
                        for file in uploaded_files:
                            df = pd.read_excel(file)
                            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                            dfs.append(df)
                        
                        st.session_state.df_all = pd.concat(dfs, ignore_index=True)
                        st.session_state.data_loaded = True
                        st.session_state.data_period = "Manual Upload"
                        
                        st.success(f"✅ Processed {len(uploaded_files)} files")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Data status
        if st.session_state.data_loaded and not st.session_state.df_all.empty:
            period_info = f" ({st.session_state.data_period})" if st.session_state.data_period else ""
            st.success(f"✅ {len(st.session_state.df_all):,} records{period_info}")
        else:
            st.info("ℹ️ No data loaded")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # BMKG Info
        st.markdown("### 🌏 BMKG INFO")
        
        quake_data = get_bmkg_realtime_quake()
        
        if quake_data:
            st.markdown(f"""
                <div style='background:white;padding:12px;border-radius:4px;border-left:3px solid #1BA0E2;'>
                    <div style='font-weight:600;color:#1494C6;font-size:0.85em;'>Latest Earthquake</div>
                    <div style='font-size:0.8em;color:#666;margin-top:8px;'>
                        📍 {quake_data.get('Wilayah', 'N/A')}<br>
                        📊 M {quake_data.get('Magnitude', 'N/A')}<br>
                        🕐 {quake_data.get('Tanggal', 'N/A')} {quake_data.get('Jam', 'N/A')}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No recent data")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Logout
        if st.button("Logout", use_container_width=True, type="primary"):
            _revoke_session_token(st.session_state.get("_session_token"))
            _clear_session_cookie()
            try:
                if _SESSION_QUERY_KEY in st.query_params:
                    del st.query_params[_SESSION_QUERY_KEY]
            except Exception:
                pass
            time.sleep(0.5)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ======================================
    # MAIN CONTENT
    # ======================================
    df_all = st.session_state.df_all

    if not df_all.empty:
        # Helper functions
        def prepare_monthly_trend(df, date_col, value_col, agg="count"):
            if date_col not in df.columns:
                return pd.DataFrame()
            
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce", dayfirst=True)
            df_copy = df_copy.dropna(subset=[date_col])
            df_copy["YearMonth"] = df_copy[date_col].dt.to_period("M").astype(str)

            if agg == "count":
                trend = df_copy.groupby("YearMonth")[value_col].count().reset_index(name="Value")
            elif agg == "nunique":
                trend = df_copy.groupby("YearMonth")[value_col].nunique().reset_index(name="Value")
            elif agg == "sum":
                trend = df_copy.groupby("YearMonth")[value_col].sum().reset_index(name="Value")
            else:
                trend = df_copy.groupby("YearMonth").size().reset_index(name="Value")

            return trend

    # ======================================
    # DATA CLEANING
    # ======================================
    if not df_all.empty:
        # Bersihkan Invoice Amount
        if "Invoice Amount" in df_all.columns:
            df_all["Invoice Amount"] = pd.to_numeric(
                df_all["Invoice Amount"].astype(str).str.replace("$", "").str.replace(",", ""),
                errors="coerce"
            )
        
        # Bersihkan Number of Rooms Night
        if "Number of Rooms Night" in df_all.columns:
            df_all["Number of Rooms Night"] = pd.to_numeric(df_all["Number of Rooms Night"], errors="coerce")
        
        # Convert dates
        if "Check in Date" in df_all.columns:
            df_all["Check in Date"] = pd.to_datetime(df_all["Check in Date"], errors="coerce", dayfirst=True)
        
        if "Check out Date" in df_all.columns:
            df_all["Check out Date"] = pd.to_datetime(df_all["Check out Date"], errors="coerce", dayfirst=True)
        
        if "Issue Time" in df_all.columns:
            df_all["Issue Time"] = pd.to_datetime(df_all["Issue Time"], errors="coerce", dayfirst=True)

            # Mapping Company Code -> Nama Perusahaan
        company_map = {
            "1010": "PT Pertamina (Persero)",
            "2022": "PT Pertamina Geothermal Energy",
            "2033": "PT Pertamina Trans Kontinental",
            "2034": "PT Pelita Air Service",
            "2042": "PT Pertamina Retail",
            "2059": "PT Pertamina Port And Logistics",
            "2061": "PT Pertamina Energy Terminal",
            "2119": "PT Nusantara Regas",
            "2138": "PT Pertamina Lubricants",
            "2147": "PT Pertamina International Shipping",
            "2151": "PT Pertamina International EP",
            "2183": "PT Pertamina Power Indonesia",
            "2186": "PT Kilang Pertamina International",
            "2205": "PT Kilang Pertamina Balikpapan",
            "2222": "PT Pertamina Patra Niaga",
            "5000": "PT Pertamina Hulu Energi",
            "2060": "PT Pertamina Marine Solutions",
            "2062": "PT Pertamina Marine Engineering",
            "2110": "PT Pertamina Drilling Services Indonesia ",
            "2052": "PT Pertamina Maintenance and Construction",
            "2040": "PT Pertamina Pedeve Indonesia",
            "2050": "PT Patra Logistik",
        }

        if "Company Code" in df_all.columns:
            df_all["Company Code"] = df_all["Company Code"].astype(str).str.strip()
            df_all["Nama Perusahaan"] = df_all["Company Code"].map(company_map).fillna("Lainnya / Unknown")

        # ======================================
        # AUTO-NORMALISASI
        # ======================================
        hotel_mapping = pd.DataFrame()  # default agar tidak NameError jika kolom "Hotel Name" tidak ada
        if "Hotel Name" in df_all.columns:
            df_all, hotel_mapping = auto_canonical_hotel_mapping(
                df_all,
                hotel_col="Hotel Name",
                threshold=0.88
            )

        # ======================================
        # GLOBAL CITY STANDARDIZATION
        # ======================================

        if "City" in df_all.columns:
            df_all["City"] = (
                df_all["City"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower()
                .str.title()
            )
            df_all.loc[df_all["City"] == "Nan", "City"] = np.nan

        if "City Destination" in df_all.columns:
            df_all["City Destination"] = (
                df_all["City Destination"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower()
                .str.title()
            )
            df_all.loc[df_all["City Destination"] == "Nan", "City Destination"] = np.nan

        # ======================================
        # GLOBAL FILTERS
        # ======================================
        st.markdown("""
            <div class='global-filter-label'>🔍 Global Filters — berlaku di seluruh tab</div>
        """, unsafe_allow_html=True)

        gcol1, gcol2, gcol3 = st.columns([2, 1.5, 1.5])

        with gcol1:
            # Filter Nama Perusahaan
            if "Nama Perusahaan" in df_all.columns:
                company_options = sorted(df_all["Nama Perusahaan"].dropna().unique().tolist())
                selected_companies = st.multiselect(
                    "🏢 Nama Perusahaan",
                    options=company_options,
                    default=[],
                    placeholder="Semua perusahaan",
                    key="gf_company"
                )
            else:
                selected_companies = []
                st.info("Kolom 'Nama Perusahaan' tidak ditemukan.")

        # Tentukan kolom tanggal untuk filter (From / To)
        _date_col_for_filter = None
        for _candidate_col in ["Issue Time", "Check in Date", "Check out Date"]:
            if _candidate_col in df_all.columns:
                _date_col_for_filter = _candidate_col
                break

        if _date_col_for_filter:
            _valid_dates = pd.to_datetime(df_all[_date_col_for_filter], errors="coerce", dayfirst=True).dropna()
            _min_date = _valid_dates.min().date() if not _valid_dates.empty else None
            _max_date = _valid_dates.max().date() if not _valid_dates.empty else None
        else:
            _min_date = _max_date = None

        # Inisialisasi nilai default di session_state agar tidak reset tiap rerun
        if _min_date and "gf_date_from_init" not in st.session_state:
            st.session_state["gf_date_from_init"] = _min_date
        if _max_date and "gf_date_to_init" not in st.session_state:
            st.session_state["gf_date_to_init"] = _max_date

        # Reset init jika data baru dimuat (periode berubah)
        _current_period = st.session_state.get("data_period", "")
        if st.session_state.get("_last_data_period") != _current_period:
            st.session_state["_last_data_period"] = _current_period
            if _min_date:
                st.session_state["gf_date_from_init"] = _min_date
            if _max_date:
                st.session_state["gf_date_to_init"] = _max_date

        with gcol2:
            if _min_date:
                # Gunakan nilai dari session_state sebagai default, bukan langsung _min_date
                _default_from = st.session_state.get("gf_date_from_init", _min_date)
                # Pastikan default_from masih dalam range yang valid
                if _default_from < _min_date:
                    _default_from = _min_date
                if _default_from > _max_date:
                    _default_from = _min_date
                filter_date_from = st.date_input(
                    "📅 From",
                    value=_default_from,
                    min_value=_min_date,
                    max_value=_max_date,
                    key="gf_date_from",
                    format="YYYY/MM/DD"
                )
                # Simpan pilihan user ke session_state agar persisten
                st.session_state["gf_date_from_init"] = filter_date_from
            else:
                filter_date_from = None
                st.info("Kolom tanggal tidak ditemukan.")

        with gcol3:
            if _max_date:
                _default_to = st.session_state.get("gf_date_to_init", _max_date)
                # Pastikan default_to masih dalam range yang valid
                if _default_to > _max_date:
                    _default_to = _max_date
                if _default_to < _min_date:
                    _default_to = _max_date
                filter_date_to = st.date_input(
                    "📅 To",
                    value=_default_to,
                    min_value=_min_date,
                    max_value=_max_date,
                    key="gf_date_to",
                    format="YYYY/MM/DD"
                )
                # Simpan pilihan user ke session_state agar persisten
                st.session_state["gf_date_to_init"] = filter_date_to
            else:
                filter_date_to = None

        # Validasi: pastikan from <= to
        # (Sama seperti badge filter di bawah, elemen ini SELALU dirender dengan
        # konten kosong/hidden saat tidak perlu, supaya jumlah elemen sebelum
        # st.tabs() tetap konsisten dan tab yang aktif tidak ke-reset — lihat
        # penjelasan lengkap di komentar BUG FIX pada badge "Filter aktif".)
        if filter_date_from and filter_date_to and filter_date_from > filter_date_to:
            st.markdown(
                "<div style='background:#fff8e1;border:1px solid #ffe0a3;border-left:4px solid #f0ad4e;"
                "border-radius:6px;padding:8px 16px;margin:8px 0;font-size:0.83em;color:#8a6100;'>"
                "⚠️ Tanggal 'From' tidak boleh lebih besar dari 'To'. Filter tanggal diabaikan."
                "</div>", unsafe_allow_html=True
            )
            filter_date_from = _min_date
            filter_date_to = _max_date
        else:
            st.markdown("<div style='display:none;height:0;margin:0;padding:0;'></div>", unsafe_allow_html=True)

        # Terapkan filter ke df_all
        df_filtered = df_all.copy()

        if selected_companies:
            df_filtered = df_filtered[df_filtered["Nama Perusahaan"].isin(selected_companies)]

        if _date_col_for_filter and filter_date_from and filter_date_to:
            _date_series = pd.to_datetime(df_filtered[_date_col_for_filter], errors="coerce", dayfirst=True)
            df_filtered = df_filtered[
                (_date_series.dt.date >= filter_date_from) &
                (_date_series.dt.date <= filter_date_to)
            ]

        # Tampilkan info jumlah data setelah filter
        _total = len(df_all)
        _filtered = len(df_filtered)
        _pct = (_filtered / _total * 100) if _total > 0 else 0
        _filter_active = bool(selected_companies) or (filter_date_from != _min_date or filter_date_to != _max_date) if _min_date else bool(selected_companies)

        # ── BUG FIX: tab dashboard selalu balik ke "Value Creation" saat filter
        # tanggal diubah ─────────────────────────────────────────────────────
        # Sebelumnya badge "Filter aktif" ini hanya dirender kalau `_filter_active`
        # True (`if _filter_active: st.markdown(...)`). st.tabs() TIDAK punya
        # parameter `key`, dan tab yang sedang aktif hanya disimpan di state
        # lokal browser (bukan di session_state) — Streamlit mengenali widget
        # ini berdasarkan POSISINYA di antara elemen-elemen lain. Begitu badge
        # ini muncul/hilang (karena filter tanggal berubah), jumlah elemen SEBELUM
        # st.tabs() ikut berubah, posisi st.tabs() jadi "beda" di mata Streamlit,
        # dan tab yang sedang aktif otomatis di-reset ke tab pertama.
        #
        # Perbaikan: elemen ini SELALU dirender (jumlah elemen sebelum st.tabs()
        # konsisten di setiap rerun) — kontennya saja yang berubah (kosong/hidden
        # saat filter tidak aktif). Ini menjaga posisi st.tabs() tetap stabil.
        if _filter_active:
            _filter_badge_html = f"""
            <div style='background:#e6f4fb;border:1px solid #b8d9f0;border-left:4px solid #1BA0E2;
                        border-radius:6px;padding:8px 16px;margin:8px 0 18px 0;font-size:0.83em;color:#0D7FCC;
                        display:flex;gap:20px;align-items:center;'>
                🔎 <b>Filter aktif</b> &nbsp;|&nbsp; Menampilkan 
                <b>{_filtered:,}</b> dari <b>{_total:,}</b> records 
                <span style='background:#1BA0E2;color:white;border-radius:20px;padding:1px 10px;font-size:0.9em;'>{_pct:.1f}%</span>
            </div>"""
        else:
            _filter_badge_html = "<div style='display:none;height:0;margin:0;padding:0;'></div>"
        st.markdown(_filter_badge_html, unsafe_allow_html=True)
        
        # Gunakan df_filtered di seluruh tab (menggantikan df_all)
        df_all = df_filtered

        st.markdown("<div class='divider' style='margin: 4px 0 18px 0;'></div>", unsafe_allow_html=True)

        # ======================================
        # TAB SEMUA
        # ======================================

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            "Value Creation",
            "Dashboard",
            "Explorer",
            "CRM",
            "Network",
            "Price Intelligence",
            "Sankey Flow",
            "Top Hotel/City",
            "Dendrogram",
            "Export",
            "Patra Jasa"
        ])

                # TAB 1: STRATEGIC VALUE CREATION
        # ======================================
        with tab1:

            import streamlit.components.v1 as components

            components.html("""
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    background: #fafafa;
    color: #1a1a1a;
    padding: 8px 4px 40px;
    -webkit-font-smoothing: antialiased;
  }

  :root {
    --purple : #1BA0E2;
    --purp-l : rgba(156,87,137,0.08);
    --purp-b : rgba(156,87,137,0.22);
    --black  : #1a1a1a;
    --grey   : #888888;
    --grey-l : #e0e0e0;
    --white  : #fafafa;
    --surf   : #ffffff;
    --bdr    : #e0e0e0;
  }

  /* ── Page Header ── */
  .hdr {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 18px; border-bottom: 1px solid var(--bdr);
    margin-bottom: 28px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .mark {
    width: 34px; height: 34px; background: var(--purple);
    border-radius: 6px; display: inline-flex;
    align-items: center; justify-content: center;
    font-family: 'Geist Mono', monospace;
    font-size: 11px; font-weight: 600; color: #fff; letter-spacing: .04em;
    flex-shrink: 0;
  }
  .brand-title {
    font-size: 17px; font-weight: 600; color: var(--black);
    letter-spacing: -.02em; line-height: 1.2; margin: 0;
  }
  .brand-sub { font-size: 12px; color: var(--grey); margin: 2px 0 0; }
  .hdr-right { display: flex; align-items: center; gap: 8px; }
  .mod-tag {
    font-family: 'Geist Mono', monospace; font-size: 10px;
    color: var(--grey); background: var(--grey-l);
    padding: 4px 10px; border-radius: 4px; letter-spacing: .08em;
  }
  .live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    border: 1px solid var(--bdr); border-radius: 100px;
    padding: 4px 12px; font-size: 11px; color: var(--grey);
    background: var(--surf);
  }
  .live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--purple); display: inline-block;
    animation: blink 2.4s ease infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

  /* ── Eyebrow ── */
  .eyebrow {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Geist Mono', monospace; font-size: 10px; font-weight: 500;
    color: var(--grey); letter-spacing: .18em;
    text-transform: uppercase; margin-bottom: 14px;
  }
  .eyebrow-ln { flex: 1; height: 1px; background: var(--bdr); }

  /* ── 2×2 Pillar Grid ── */
  .pgrid {
    display: grid; grid-template-columns: 1fr 1fr;
    border: 1px solid var(--bdr); border-radius: 10px;
    overflow: hidden; background: var(--bdr); gap: 1px;
    margin-bottom: 1px;
  }
  .pcard {
    background: var(--surf); padding: 26px 24px;
    position: relative; transition: background .18s ease;
  }
  .pcard:hover { background: #fdfdfd; }
  .pcard::before {
    content: ''; position: absolute;
    top: 26px; bottom: 26px; left: 0; width: 2px;
    background: var(--purple); opacity: 0; transition: opacity .2s ease;
  }
  .pcard:hover::before { opacity: 1; }
  .card-top {
    display: flex; align-items: flex-start;
    justify-content: space-between; margin-bottom: 14px;
  }
  .card-icon { font-size: 18px; line-height: 1; }
  .badge {
    font-family: 'Geist Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: .12em;
    color: var(--purple); background: var(--purp-l);
    border: 1px solid var(--purp-b);
    padding: 3px 8px; border-radius: 3px;
  }
  .pcard h3 {
    font-size: 14px; font-weight: 600; color: var(--black);
    letter-spacing: -.015em; margin-bottom: 4px; line-height: 1.3;
  }
  .obj { font-size: 12px; color: var(--grey); margin-bottom: 18px; line-height: 1.55; }
  .fl {
    font-family: 'Geist Mono', monospace; font-size: 9px; font-weight: 500;
    letter-spacing: .15em; text-transform: uppercase;
    color: var(--purple); margin-bottom: 8px;
  }
  .dlist { display: flex; flex-direction: column; gap: 5px; margin-bottom: 14px; }
  .drow  { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--black); }
  .ddot  {
    width: 4px; height: 4px; border-radius: 50%;
    background: var(--purple); flex-shrink: 0; display: inline-block;
  }
  .sep  { height: 1px; background: var(--grey-l); margin: 12px 0; }
  .ilist { display: flex; flex-direction: column; gap: 5px; }
  .irow {
    display: flex; align-items: flex-start; gap: 8px;
    font-size: 11.5px; color: var(--grey); line-height: 1.5;
  }
  .iarr { font-size: 8px; margin-top: 5px; flex-shrink: 0; color: var(--purple); opacity: .7; }

  /* ── Card 5 ── */
  .c5wrap {
    border: 1px solid var(--bdr); border-top: none;
    border-radius: 0 0 10px 10px; overflow: hidden; margin-bottom: 32px;
  }
  .c5hd {
    background: #f5f5f5; border-bottom: 1px solid var(--bdr);
    padding: 16px 24px; display: flex;
    align-items: center; justify-content: space-between;
  }
  .c5hd-l { display: flex; align-items: center; gap: 10px; }
  .c5hd h3 { font-size: 13px; font-weight: 600; color: var(--black); margin: 0; }
  .c5hd .c5sub { font-size: 11px; color: var(--grey); }
  .c5body {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    background: var(--bdr); gap: 1px;
  }
  .c5col { background: var(--surf); padding: 20px 24px; }

  /* ── Executive Summary ── */
  .exec { border: 1px solid var(--bdr); border-radius: 10px; overflow: hidden; }
  .exec-hd {
    background: var(--black); padding: 22px 26px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .exec-hd h2 {
    font-size: 16px; font-weight: 600; color: var(--white);
    letter-spacing: -.02em; margin: 0;
  }
  .exec-hd .exec-sub { font-size: 11.5px; color: #888; margin: 3px 0 0; }
  .exec-tag {
    font-family: 'Geist Mono', monospace; font-size: 9.5px;
    letter-spacing: .12em; color: var(--purple);
    background: rgba(156,87,137,.15); border: 1px solid rgba(156,87,137,.3);
    padding: 4px 11px; border-radius: 100px; text-transform: uppercase;
    white-space: nowrap;
  }
  .pillars {
    display: grid; grid-template-columns: repeat(4,1fr);
    background: var(--bdr); gap: 1px;
    border-bottom: 1px solid var(--bdr);
  }
  .pillar {
    background: var(--surf); padding: 18px 18px 16px; position: relative;
  }
  .pillar::after {
    content: ''; position: absolute;
    bottom: 0; left: 18px; right: 18px; height: 1px;
    background: var(--purple); opacity: 0; transition: opacity .2s;
  }
  .pillar:hover::after { opacity: .4; }
  .p-icon { font-size: 16px; margin-bottom: 8px; display: block; }
  .pillar p { font-size: 12px; color: var(--black); font-weight: 500; line-height: 1.45; margin: 0; }
  .flow-bar {
    background: var(--surf); padding: 18px 24px;
    display: flex; align-items: center; justify-content: center;
    border-top: 1px solid var(--bdr);
  }
  .fn { display: flex; flex-direction: column; align-items: center; gap: 2px; }
  .fn-lbl { font-size: 12px; font-weight: 600; color: var(--black); letter-spacing: -.01em; }
  .fn-lbl.hi { color: var(--purple); }
  .fn-sub {
    font-family: 'Geist Mono', monospace; font-size: 9px;
    color: var(--grey); letter-spacing: .06em;
  }
  .fsep { display: flex; align-items: center; margin: 0 18px; }
  .fsep-ln { width: 28px; height: 1px; background: var(--grey-l); }
  .fsep-arr { font-size: 9px; color: var(--purple); opacity: .6; }
</style>
</head>
<body>

  <!-- ── Page Header ── -->
  <div class="hdr">
    <div class="brand">
      <div class="mark">MTX</div>
      <div>
        <p class="brand-title">Strategic Value Creation Framework</p>
        <p class="brand-sub">Travel Analytics &amp; Procurement Intelligence</p>
      </div>
    </div>
    <div class="hdr-right">
      <span class="mod-tag">MODULE 08</span>
      <span class="live-badge"><span class="live-dot"></span>5 Value Pillars</span>
    </div>
  </div>

  <!-- ── Eyebrow ── -->
  <div class="eyebrow"><span>Value Pillars</span><div class="eyebrow-ln"></div></div>

  <!-- ── 2×2 Grid ── -->
  <div class="pgrid">

    <!-- 01 Financial -->
    <div class="pcard">
      <div class="card-top">
        <span class="card-icon">💰</span>
        <span class="badge">01 · FINANCIAL</span>
      </div>
      <h3>Financial Optimization</h3>
      <p class="obj">Mengurangi total travel spend dan meningkatkan efisiensi biaya operasional secara terukur.</p>
      <div class="fl">Value Drivers</div>
      <div class="dlist">
        <div class="drow"><span class="ddot"></span>Rate benchmarking antar hotel</div>
        <div class="drow"><span class="ddot"></span>Price per night analysis</div>
        <div class="drow"><span class="ddot"></span>Negotiation leverage berbasis volume room nights</div>
        <div class="drow"><span class="ddot"></span>Last-minute booking cost impact</div>
      </div>
      <div class="sep"></div>
      <div class="fl">Business Impact</div>
      <div class="ilist">
        <div class="irow"><span class="iarr">▶</span>Estimasi saving 5–15% dari negotiated rate</div>
        <div class="irow"><span class="iarr">▶</span>Pengurangan overpricing hotel tidak terstandarisasi</div>
        <div class="irow"><span class="iarr">▶</span>Kontrol budget lintas perusahaan</div>
      </div>
    </div>

    <!-- 02 Operational -->
    <div class="pcard">
      <div class="card-top">
        <span class="card-icon">⚙️</span>
        <span class="badge">02 · OPERATIONAL</span>
      </div>
      <h3>Operational Efficiency</h3>
      <p class="obj">Meningkatkan kecepatan dan kualitas proses booking secara end-to-end.</p>
      <div class="fl">Value Drivers</div>
      <div class="dlist">
        <div class="drow"><span class="ddot"></span>Lead time monitoring</div>
        <div class="drow"><span class="ddot"></span>Multi-booking behavior analysis</div>
        <div class="drow"><span class="ddot"></span>Travel request pattern heatmap</div>
        <div class="drow"><span class="ddot"></span>Automation &amp; canonical hotel mapping</div>
      </div>
      <div class="sep"></div>
      <div class="fl">Business Impact</div>
      <div class="ilist">
        <div class="irow"><span class="iarr">▶</span>Mengurangi booking mendadak (≤2 hari)</div>
        <div class="irow"><span class="iarr">▶</span>Mengurangi duplikasi nama hotel</div>
        <div class="irow"><span class="iarr">▶</span>Meningkatkan data reliability untuk reporting</div>
      </div>
    </div>

    <!-- 03 Procurement -->
    <div class="pcard">
      <div class="card-top">
        <span class="card-icon">🎯</span>
        <span class="badge">03 · PROCUREMENT</span>
      </div>
      <h3>Strategic Procurement Intelligence</h3>
      <p class="obj">Meningkatkan posisi tawar terhadap hotel dan vendor strategis.</p>
      <div class="fl">Value Drivers</div>
      <div class="dlist">
        <div class="drow"><span class="ddot"></span>Top 10 hotel concentration</div>
        <div class="drow"><span class="ddot"></span>Volume aggregation per city</div>
        <div class="drow"><span class="ddot"></span>Corporate usage clustering</div>
        <div class="drow"><span class="ddot"></span>Canonical hotel normalization</div>
      </div>
      <div class="sep"></div>
      <div class="fl">Business Impact</div>
      <div class="ilist">
        <div class="irow"><span class="iarr">▶</span>Centralized negotiation strategy</div>
        <div class="irow"><span class="iarr">▶</span>Volume-based discount leverage</div>
        <div class="irow"><span class="iarr">▶</span>Preferred hotel program optimization</div>
      </div>
    </div>

    <!-- 04 Governance -->
    <div class="pcard">
      <div class="card-top">
        <span class="card-icon">🛡️</span>
        <span class="badge">04 · GOVERNANCE</span>
      </div>
      <h3>Risk &amp; Governance Control</h3>
      <p class="obj">Menjamin kontrol dan keamanan data travel perusahaan secara sistemik.</p>
      <div class="fl">Value Drivers</div>
      <div class="dlist">
        <div class="drow"><span class="ddot"></span>Role-based download restriction</div>
        <div class="drow"><span class="ddot"></span>Admin-only data export</div>
        <div class="drow"><span class="ddot"></span>Real-time monitoring dashboard</div>
        <div class="drow"><span class="ddot"></span>2FA Google Authenticator (TOTP)</div>
      </div>
      <div class="sep"></div>
      <div class="fl">Business Impact</div>
      <div class="ilist">
        <div class="irow"><span class="iarr">▶</span>Mencegah data leakage</div>
        <div class="irow"><span class="iarr">▶</span>Meningkatkan compliance standar</div>
        <div class="irow"><span class="iarr">▶</span>Governance berbasis sistem</div>
      </div>
    </div>

  </div>

  <!-- ── Card 5 — Predictive ── -->
  <div class="c5wrap">
    <div class="c5hd">
      <div class="c5hd-l">
        <span style="font-size:17px;">🔮</span>
        <div>
          <h3>Predictive &amp; Future Intelligence</h3>
          <span class="c5sub">Next Phase Development</span>
        </div>
      </div>
      <span class="badge">05 · PREDICTIVE</span>
    </div>
    <div class="c5body">
      <div class="c5col">
        <div class="fl">Potential Development</div>
        <div class="dlist" style="margin-top:10px;">
          <div class="drow"><span class="ddot"></span>LSTM-based demand forecasting</div>
          <div class="drow"><span class="ddot"></span>Hotel price anomaly detection</div>
          <div class="drow"><span class="ddot"></span>Traveler segmentation (KMeans)</div>
          <div class="drow"><span class="ddot"></span>Automated negotiation simulator</div>
        </div>
      </div>
      <div class="c5col" style="border-left:1px solid #e0e0e0;border-right:1px solid #e0e0e0;">
        <div class="fl">Future Business Value</div>
        <div class="ilist" style="margin-top:10px;">
          <div class="irow"><span class="iarr">▶</span>Predictive budget planning yang akurat</div>
          <div class="irow"><span class="iarr">▶</span>Early warning overpricing otomatis</div>
          <div class="irow"><span class="iarr">▶</span>Smart hotel contract recommendation</div>
        </div>
      </div>
      <div class="c5col">
        <div class="fl">Technology Stack</div>
        <div class="dlist" style="margin-top:10px;">
          <div class="drow"><span class="ddot"></span>Deep Learning / LSTM</div>
          <div class="drow"><span class="ddot"></span>Unsupervised Clustering</div>
          <div class="drow"><span class="ddot"></span>Anomaly Detection Models</div>
          <div class="drow"><span class="ddot"></span>Simulation Engine</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Executive Summary eyebrow ── -->
  <div class="eyebrow" style="margin-top:4px;">
    <span>Executive Summary</span><div class="eyebrow-ln"></div>
  </div>

  <!-- ── Executive Summary ── -->
  <div class="exec">
    <div class="exec-hd">
      <div>
        <h2>MTRAX Platform Overview</h2>
        <p class="exec-sub">Lebih dari sekadar dashboard — sistem intelijen strategis untuk travel spend.</p>
      </div>
      <span class="exec-tag">Strategic Intelligence</span>
    </div>

    <div class="pillars">
      <div class="pillar">
        <span class="p-icon">🧠</span>
        <p>Strategic Decision<br>Support System</p>
      </div>
      <div class="pillar" style="border-left:1px solid #e0e0e0;">
        <span class="p-icon">⚡</span>
        <p>Negotiation<br>Intelligence Engine</p>
      </div>
      <div class="pillar" style="border-left:1px solid #e0e0e0;">
        <span class="p-icon">💡</span>
        <p>Corporate Cost<br>Optimization Platform</p>
      </div>
      <div class="pillar" style="border-left:1px solid #e0e0e0;">
        <span class="p-icon">🔐</span>
        <p>Governance-Controlled<br>Analytics Ecosystem</p>
      </div>
    </div>

    <div class="flow-bar">
      <div class="fn">
        <span class="fn-lbl">Insight</span>
        <span class="fn-sub">Data → Analytics</span>
      </div>
      <div class="fsep"><div class="fsep-ln"></div><span class="fsep-arr">›</span></div>
      <div class="fn">
        <span class="fn-lbl">Strategy</span>
        <span class="fn-sub">Pattern → Direction</span>
      </div>
      <div class="fsep"><div class="fsep-ln"></div><span class="fsep-arr">›</span></div>
      <div class="fn">
        <span class="fn-lbl">Negotiation Leverage</span>
        <span class="fn-sub">Volume → Power</span>
      </div>
      <div class="fsep"><div class="fsep-ln"></div><span class="fsep-arr">›</span></div>
      <div class="fn">
        <span class="fn-lbl hi">Financial Impact</span>
        <span class="fn-sub">Cost → Savings</span>
      </div>
    </div>
  </div>

</body>
</html>
""", height=1600, scrolling=True)


        # ======================================
        # TAB 2: DASHBOARD
        # ======================================
        with tab2:

        # ======================================
        # FILTER OVERVIEW — NAMA PERUSAHAAN
        # ======================================
            company_col = "Nama Perusahaan"

            if company_col in df_all.columns:
                company_list = (
                    df_all[company_col]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                )

                st.markdown("""
                <style>
                div[data-testid="stMultiSelect"] > label {
                    font-size: 0.85em !important;
                    color: #555 !important;
                    font-weight: 400 !important;
                    margin-bottom: 4px !important;
                }
                div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
                    border: 1px solid #e0e0e0 !important;
                    border-radius: 6px !important;
                    background: white !important;
                    min-height: 38px !important;
                    font-size: 0.875em !important;
                }
                div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
                    border-color: #1BA0E2 !important;
                    box-shadow: 0 0 0 2px rgba(156,87,137,0.15) !important;
                }
                div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
                    background: #1BA0E2 !important;
                    border-radius: 50px !important;
                    padding: 2px 10px !important;
                    font-size: 0.78em !important;
                    font-weight: 500 !important;
                    color: white !important;
                    border: none !important;
                }
                div[data-testid="stMultiSelect"] [data-baseweb="tag"] span[role="presentation"] {
                    color: rgba(255,255,255,0.75) !important;
                    font-size: 1.1em !important;
                }
                div[data-testid="stMultiSelect"] [role="option"]:hover {
                    background: rgba(156,87,137,0.08) !important;
                }
                div[data-testid="stMultiSelect"] [aria-selected="true"] {
                    background: rgba(156,87,137,0.12) !important;
                    color: #1BA0E2 !important;
                    font-weight: 500 !important;
                }
                </style>
                """, unsafe_allow_html=True)

                # BUG FIX: widget multiselect ini sebelumnya di-comment-out, sehingga
                # baris `if selected_companies:` di bawah diam-diam memakai variabel
                # `selected_companies` milik filter TAB 1 (key="gf_company") alih-alih
                # filter miliknya sendiri. Akibatnya, dropdown filter perusahaan khusus
                # Tab 2 (Dashboard) tidak pernah muncul, dan Tab 2 selalu ikut ter-filter
                # oleh pilihan perusahaan dari Tab 1 tanpa bisa diatur independen di sini.
                selected_companies_ov = st.multiselect(
                    "Filter Overview berdasarkan Nama Perusahaan",
                    options=company_list,
                    default=[],
                    placeholder="Semua perusahaan (pilih untuk filter spesifik)…",
                    key="ov_company_tab2"
                )

                if selected_companies_ov:
                    df_overview = df_all[df_all[company_col].isin(selected_companies_ov)]
                else:
                    df_overview = df_all.copy()
            else:
                st.warning("Kolom 'Nama Perusahaan' tidak ditemukan.")
                df_overview = df_all.copy()

            # ======================================
            # FILTER DOMESTIK & INTERNASIONAL
            # ======================================
            if "Country" in df_overview.columns:
                _country_upper = df_overview["Country"].astype(str).str.strip().str.upper()
                _has_domestic   = (_country_upper == "INDONESIA").any()
                _has_intl       = (_country_upper != "INDONESIA").any()

                st.markdown("""
                <style>
                div[data-testid="stRadio"] > label {
                    font-size: 0.85em !important;
                    color: #555 !important;
                    font-weight: 500 !important;
                }
                div[data-testid="stRadio"] [data-baseweb="radio"] label {
                    font-size: 0.875em !important;
                }
                </style>
                """, unsafe_allow_html=True)

                _market_options = ["Semua", "Domestik 🇮🇩", "Internasional 🌍"]
                _market_filter = st.radio(
                    "🌐 Market",
                    options=_market_options,
                    index=0,
                    horizontal=True,
                    key="db_market_filter"
                )

                if _market_filter == "Domestik 🇮🇩":
                    df_overview = df_overview[
                        df_overview["Country"].astype(str).str.strip().str.upper() == "INDONESIA"
                    ]
                    _market_badge = (
                        "<span style='background:#e6f4fb;color:#0D7FCC;border:1px solid #b8d9f0;"
                        "border-radius:20px;padding:2px 12px;font-size:0.8em;font-weight:600;"
                        "margin-left:8px;'>🇮🇩 Domestik</span>"
                    )
                elif _market_filter == "Internasional 🌍":
                    df_overview = df_overview[
                        df_overview["Country"].astype(str).str.strip().str.upper() != "INDONESIA"
                    ]
                    _market_badge = (
                        "<span style='background:#fff4e6;color:#b05a00;border:1px solid #f5c891;"
                        "border-radius:20px;padding:2px 12px;font-size:0.8em;font-weight:600;"
                        "margin-left:8px;'>🌍 Internasional</span>"
                    )
                else:
                    _market_badge = (
                        "<span style='background:#f0f5e9;color:#3a6b00;border:1px solid #bdd99a;"
                        "border-radius:20px;padding:2px 12px;font-size:0.8em;font-weight:600;"
                        "margin-left:8px;'>🌐 Semua Market</span>"
                    )

                # Info badge jumlah data setelah filter market
                _n_market = len(df_overview)
                st.markdown(
                    f"<div style='margin:4px 0 14px 0;font-size:0.82em;color:#666;'>"
                    f"Menampilkan <b>{_n_market:,}</b> records{_market_badge}</div>",
                    unsafe_allow_html=True
                )


            st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
            
            # Primary Metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                total_rows = len(df_overview)
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>Bookings</div>
                        <div class='metric-value'>{total_rows:,}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                if "Travel Request Number" in df_overview.columns:
                    unique_tr = df_overview["Travel Request Number"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Trvl Requests</div>
                            <div class='metric-value'>{unique_tr:,}</div>
                        </div>
                    """, unsafe_allow_html=True)

            with col3:
                if "Employee Id" in df_overview.columns:
                    unique_travelers = df_overview["Employee Id"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Travelers</div>
                            <div class='metric-value'>{unique_travelers:,}</div>
                        </div>
                    """, unsafe_allow_html=True)

            with col4:
                if "Hotel Name" in df_overview.columns:
                    unique_hotels = df_overview["Hotel Name"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Hotels</div>
                            <div class='metric-value'>{unique_hotels:,}</div>
                        </div>
                    """, unsafe_allow_html=True)

            with col5:
                if "Number of Rooms Night" in df_overview.columns:
                    total_nights = df_overview["Number of Rooms Night"].sum()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Room Nights</div>
                            <div class='metric-value'>{total_nights:,.0f}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Secondary Metrics
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if "Company Code" in df_overview.columns:
                    unique_company = df_overview["Company Code"].nunique()
                    st.metric("Companies", f"{unique_company:,}")

            with col2:
                if "Cost Center Pekerja" in df_overview.columns:
                    unique_cc = df_overview["Cost Center Pekerja"].nunique()
                    st.metric("Cost Centers", f"{unique_cc:,}")

            with col3:
                if "City" in df_overview.columns:
                    unique_cities = df_overview["City"].nunique()
                    st.metric("Cities", f"{unique_cities:,}")

            with col4:
                if "Country" in df_overview.columns:
                    unique_countries = df_overview["Country"].nunique()
                    st.metric("Countries", f"{unique_countries:,}")

            # Travel Request Analysis
            if "Travel Request Number" in df_overview.columns:
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Travel Request Analysis</div>", unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    tr_bookings = df_overview.groupby("Travel Request Number").size()
                    avg_booking = tr_bookings.mean()
                    max_booking = tr_bookings.max()
                    
                    st.markdown(f"""
                        <div class='stats-card'>
                            <div class='stats-label'>Avg Bookings</div>
                            <div class='stats-number'>{avg_booking:.1f}</div>
                            <div class='stats-detail'>Max: {max_booking}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if "Number of Rooms Night" in df_overview.columns:
                        tr_nights = df_overview.groupby("Travel Request Number")["Number of Rooms Night"].sum()
                        avg_nights = tr_nights.mean()
                        max_nights = tr_nights.max()
                        
                        st.markdown(f"""
                            <div class='stats-card'>
                                <div class='stats-label'>Avg Nights</div>
                                <div class='stats-number'>{avg_nights:.1f}</div>
                                <div class='stats-detail'>Max: {max_nights:.0f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    tr_with_multiple = (tr_bookings > 1).sum()
                    tr_single = (tr_bookings == 1).sum()
                    multi_percentage = (tr_with_multiple / len(tr_bookings) * 100)
                    
                    st.markdown(f"""
                        <div class='stats-card'>
                            <div class='stats-label'>Multi-Booking</div>
                            <div class='stats-number'>{multi_percentage:.0f}%</div>
                            <div class='stats-detail'>{tr_with_multiple:,} / {tr_single:,}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    if "Invoice Amount" in df_overview.columns and "Number of Rooms Night" in df_overview.columns:
                        valid_rows = (df_overview["Invoice Amount"].notna()) & \
                                    (df_overview["Number of Rooms Night"].notna()) & \
                                    (df_overview["Number of Rooms Night"] > 0)
                        df_valid = df_overview[valid_rows].copy()
                        df_valid["Price Per Night"] = df_valid["Invoice Amount"] / df_valid["Number of Rooms Night"]
                        
                        avg_price = df_valid["Price Per Night"].mean()
                        
                        st.markdown(f"""
                            <div class='stats-card'>
                                <div class='stats-label'>Avg Rate</div>
                                <div class='stats-number'>{avg_price/1000:.0f}k</div>
                                <div class='stats-detail'>Per Night</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col5:
                    date_cols = ["Issue Time", "Check in Date"]
                    for col in date_cols:
                        if col in df_overview.columns:
                            df_overview[col] = pd.to_datetime(df_overview[col], errors="coerce", dayfirst=True)

                    if "Issue Time" in df_all.columns and "Check in Date" in df_all.columns:
                        df_lead = df_overview[
                            df_overview["Issue Time"].notna() &
                            df_overview["Check in Date"].notna()
                        ].copy()

                        df_lead["Lead Time (Days)"] = (
                            df_lead["Check in Date"].dt.normalize() -
                            df_lead["Issue Time"].dt.normalize()
                        ).dt.days

                        lead_valid = df_lead[df_lead["Lead Time (Days)"] >= 0]
                        avg_lead = lead_valid["Lead Time (Days)"].mean()
                        last_minute_pct = (
                            (lead_valid["Lead Time (Days)"] <= 2).sum() / len(lead_valid) * 100
                            if len(lead_valid) > 0 else 0
                        )

                        st.markdown(f"""
                            <div class='stats-card'>
                                <div class='stats-label'>Lead Time</div>
                                <div class='stats-number'>{avg_lead:.0f}</div>
                                <div class='stats-detail'>{last_minute_pct:.0f}% last-minute</div>
                            </div>
                        """, unsafe_allow_html=True)

            if "Issue Time" in df_overview.columns and "Travel Request Number" in df_overview.columns:
                df_heat = df_overview[
                    df_overview["Issue Time"].notna() &
                    df_overview["Travel Request Number"].notna()
                ].copy()

                df_heat["Issue Hour"] = df_heat["Issue Time"].dt.hour
                df_heat["Issue Day"] = df_heat["Issue Time"].dt.day_name()

                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                pivot_core = (
                    df_heat
                    .groupby(["Issue Day", "Issue Hour"])["Travel Request Number"]
                    .nunique()
                    .reset_index(name="TR_Count")
                    .pivot(index="Issue Day", columns="Issue Hour", values="TR_Count")
                    .reindex(day_order)
                    .fillna(0)
                )

                total_day = pivot_core.sum(axis=1)
                total_hour = pivot_core.sum(axis=0)
                grand_total = total_day.sum()

                fig = px.imshow(
                    pivot_core,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=["#ffffff", "#ddd", "#1BA0E2"]
                )

                annotations = []

                for i, day in enumerate(pivot_core.index):
                    annotations.append(dict(
                        x=len(pivot_core.columns),
                        y=i,
                        text=f"<b>{int(total_day.loc[day])}</b>",
                        showarrow=False,
                        font=dict(color="black", size=12)
                    ))

                for j, hour in enumerate(pivot_core.columns):
                    annotations.append(dict(
                        x=j,
                        y=len(pivot_core.index),
                        text=f"<b>{int(total_hour.loc[hour])}</b>",
                        showarrow=False,
                        font=dict(color="black", size=12)
                    ))

                annotations.append(dict(
                    x=len(pivot_core.columns),
                    y=len(pivot_core.index),
                    text=f"<b>{int(grand_total)}</b>",
                    showarrow=False,
                    font=dict(color="black", size=13)
                ))

                fig.update_layout(
                    title="Travel Request Heatmap (Issue Time)",
                    xaxis_title="Issue Hour",
                    yaxis_title="Issue Day",
                    annotations=annotations,
                    height=420,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=40, r=60, t=60, b=60),
                    font=dict(size=11)
                )

                fig.update_xaxes(range=[-0.5, len(pivot_core.columns) + 0.5])
                fig.update_yaxes(range=[len(pivot_core.index) + 0.5, -0.5])

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                if "City Destination" in df_overview.columns:
                    top_cities = df_overview["City Destination"].value_counts().head(10)

                    fig_cities = px.bar(
                        x=top_cities.values,
                        y=top_cities.index,
                        orientation="h",
                        text=top_cities.values
                    )

                    fig_cities.update_traces(
                        marker_color="#1BA0E2",
                        texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=11)
                    )

                    fig_cities.update_layout(
                        height=380,
                        title="Top 10 Cities",
                        xaxis_title="",
                        yaxis_title="",
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=40, t=50, b=10),
                        showlegend=False
                    )

                    st.plotly_chart(fig_cities, use_container_width=True)

            with col2:
                if "Country Destination" in df_overview.columns:
                    top_countries = df_overview["Country Destination"].value_counts().head(10)

                    fig_countries = px.pie(
                        values=top_countries.values,
                        names=top_countries.index,
                        hole=0.4,
                        color_discrete_sequence=["#1BA0E2", "#47b5e8", "#75caf0", "#a3dcf6", "#d1eefb"]
                    )

                    fig_countries.update_layout(
                        height=380,
                        title="Top 10 Countries",
                        plot_bgcolor="white",
                        paper_bgcolor="white"
                    )

                    st.plotly_chart(fig_countries, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                if "Nama Perusahaan" in df_overview.columns and "Travel Request Number" in df_overview.columns:
                    top_dir = (
                        df_overview.groupby("Nama Perusahaan")["Travel Request Number"]
                        .nunique()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index(name="Travel Requests")
                    )

                    fig_dir = px.bar(
                        top_dir,
                        x="Travel Requests",
                        y="Nama Perusahaan",
                        orientation="h",
                        text="Travel Requests"
                    )

                    fig_dir.update_traces(
                        marker_color="#1BA0E2",
                        texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=11)
                    )

                    fig_dir.update_layout(
                        height=380,
                        title="Top 10 Directorates by Travel Requests",
                        xaxis_title="Number of Travel Requests",
                        yaxis_title="",
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=40, t=50, b=10),
                        showlegend=False
                    )

                    st.plotly_chart(fig_dir, use_container_width=True)

                st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

                if "Issue Time" in df_overview.columns and "Travel Request Number" in df_overview.columns:

                    trend_df = prepare_monthly_trend(
                        df_overview,
                        date_col="Issue Time",
                        value_col="Travel Request Number",
                        agg="nunique"
                    )

                    fig_trend = px.line(
                        trend_df,
                        x="YearMonth",
                        y="Value",
                        markers=True,
                        labels={"Value": "Unique Travel Requests", "YearMonth": "Month"}
                    )

                    fig_trend.update_traces(
                        line=dict(color="#1BA0E2", width=3), 
                        marker=dict(size=8, color="#1BA0E2")
                    )
                    
                    fig_trend.update_layout(
                        height=380,
                        title="Monthly Travel Request Trend",
                        xaxis_title="",
                        yaxis_title="Travel Requests",
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=40, r=20, t=50, b=40),
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    output_tr_trend = BytesIO()
                    trend_df.to_excel(
                        output_tr_trend,
                        index=False,
                        sheet_name="Monthly Travel Request Trend"
                    )
                    output_tr_trend.seek(0)

                    if st.session_state.get('role') == 'Admin':
                        st.download_button(
                            label="⬇️ Download Data",
                            data=output_tr_trend,
                            file_name="monthly_travel_request_trend.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_monthly_tr_trend")
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border:1px solid #b8d9f0;border-left:3px solid #1BA0E2;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;
                        display:flex;align-items:center;gap:8px;'>
                            <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                        </div>""", unsafe_allow_html=True)

            with col2:
                if "Country" in df_overview.columns:
                    df_all["Country"] = df_overview["Country"].astype(str).str.strip().str.upper()

                    indo = df_overview[df_overview["Country"] == "INDONESIA"].shape[0]
                    intl = df_overview[df_overview["Country"] != "INDONESIA"].shape[0]

                    pie_df = pd.DataFrame({
                        "Market": ["Domestic", "International"],
                        "Bookings": [indo, intl]
                    })

                    fig_pie = px.pie(
                        pie_df,
                        names="Market",
                        values="Bookings",
                        hole=0.5,
                        color_discrete_sequence=["#1BA0E2", "#cccccc"]
                    )

                    fig_pie.update_traces(
                        textinfo="percent+label", 
                        textfont=dict(size=12),
                        marker=dict(line=dict(color='white', width=2))
                    )

                    total = pie_df["Bookings"].sum()

                    fig_pie.update_layout(
                        height=380,
                        title="Market Distribution (Dom vs Int)",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                        margin=dict(l=10, r=30, t=50, b=10),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        annotations=[dict(
                            text=f"<b>{total:,}</b><br>Total", 
                            x=0.5, y=0.5, font=dict(size=16), showarrow=False
                        )]
                    )

                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

                if "Issue Time" in df_overview.columns and "Number of Rooms Night" in df_overview.columns:

                    trend_df = prepare_monthly_trend(
                        df_overview,
                        date_col="Issue Time",
                        value_col="Number of Rooms Night",
                        agg="sum"
                    )

                    fig_trend = px.line(
                        trend_df,
                        x="YearMonth",
                        y="Value",
                        markers=True,
                        labels={"Value": "Total Room Nights", "YearMonth": "Month"}
                    )

                    fig_trend.update_traces(
                        line=dict(color="#1BA0E2", width=3), 
                        marker=dict(size=8, color="#1BA0E2")
                    )
                    
                    fig_trend.update_layout(
                        height=380,
                        title="Monthly Room Nights Trend",
                        xaxis_title="",
                        yaxis_title="Room Nights",
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=40, r=20, t=50, b=40),
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_trend, use_container_width=True)

                    output_rn_trend = BytesIO()
                    trend_df.to_excel(
                        output_rn_trend,
                        index=False,
                        sheet_name="Monthly Room Nights Trend"
                    )
                    output_rn_trend.seek(0)

                    if st.session_state.get('role') == 'Admin':
                        st.download_button(
                            label="⬇️ Download Data",
                            data=output_rn_trend,
                            file_name="monthly_room_nights_trend.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_monthly_rn_trend")
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border:1px solid #b8d9f0;border-left:3px solid #1BA0E2;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;
                        display:flex;align-items:center;gap:8px;'>
                            <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("Column 'Issue Time' or 'Number of Rooms Night' not available.")

            for city_col in ["City", "City Destination"]:
                if city_col in df_overview.columns:
                    df_overview[city_col] = (
                        df_overview[city_col]
                        .astype(str).str.strip().str.lower().str.title()
                    )
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ======================================
        # TAB 3: EXPLORER
        # ======================================
        with tab3:
            st.markdown("<div class='section-title'>Data Explorer</div>", unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                search_term = st.text_input("🔍 Search", placeholder="Search in any column...")

            with col2:
                if "City Destination" in df_all.columns:
                    cities = ["All"] + sorted(df_all["City Destination"].dropna().unique().tolist())
                    selected_city = st.selectbox("Filter by City", cities)
                else:
                    selected_city = "All"

            df_filtered = df_all.copy()

            if search_term:
                mask = df_filtered.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                df_filtered = df_filtered[mask]

            if selected_city != "All" and "City Destination" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["City Destination"] == selected_city]

            st.markdown(f"**Showing {len(df_filtered):,} of {len(df_all):,} records**")
            st.dataframe(df_filtered, use_container_width=True, height=500)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Dataset Information</div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Columns", len(df_all.columns))

            with col2:
                st.metric("Total Rows", f"{len(df_all):,}")

            with col3:
                missing_pct = (df_all.isnull().sum().sum() / (len(df_all) * len(df_all.columns))) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")

            st.markdown("<div class='section-title'>Hotel Name Text Similarity Analysis</div>", unsafe_allow_html=True)

            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.70,
                max_value=0.95,
                value=0.85,
                step=0.01,
                help="Semakin tinggi, semakin ketat kemiripan"
            )

            if "Hotel Name" in df_all.columns:
                with st.spinner("Analyzing hotel name similarity..."):
                    sim_df = hotel_name_similarity(
                        df_all,
                        text_col="Hotel Name",
                        threshold=threshold
                    )

                if not sim_df.empty:
                    st.dataframe(sim_df, use_container_width=True, height=450)
                    st.caption("🔎 Digunakan untuk mendeteksi potensi duplikasi nama hotel akibat perbedaan penulisan.")

                    output = BytesIO()
                    sim_df.to_excel(output, index=False, sheet_name="Hotel Name Similarity")
                    output.seek(0)

                    if st.session_state.get('role') == 'Admin':
                        st.download_button(
                            label="⬇️ Download Similarity Result (Excel)",
                            data=output,
                            file_name="hotel_name_similarity.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_hotel_similarity")
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border:1px solid #b8d9f0;border-left:3px solid #1BA0E2;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;
                        display:flex;align-items:center;gap:8px;'>
                            <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("Tidak ditemukan hotel dengan tingkat kemiripan sesuai threshold.")
            else:
                st.warning("Kolom 'Hotel Name' tidak tersedia.")

            st.markdown("### Canonical Hotel Mapping")

            if not hotel_mapping.empty:
                st.dataframe(
                    hotel_mapping.sort_values("Canonical Hotel Name"),
                    use_container_width=True,
                    height=350
                )

                output = BytesIO()
                hotel_mapping.to_excel(output, index=False, sheet_name="Hotel Canonical Mapping")
                output.seek(0)

                if st.session_state.get('role') == 'Admin':
                    st.download_button(
                        "⬇️ Download Canonical Mapping (Excel)",
                        data=output,
                        file_name="hotel_canonical_mapping.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_hotel_canonical_mapping")
                else:
                    st.markdown("""
                    <div style='background:#f9f9f9;border:1px solid #b8d9f0;border-left:3px solid #1BA0E2;
                    border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;
                    display:flex;align-items:center;gap:8px;'>
                        <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Tidak ada mapping canonical yang terbentuk.")

        # ======================================
        # TAB 3: ANALYTICS — CRM
        # ======================================
        with tab4:

            st.markdown("<div class='section-title'>CRM Analytics</div>", unsafe_allow_html=True)

            df_crm = df_all.copy()

            required_cols = ["Employee Id", "Travel Request Number", "Issue Time"]
            if not all(col in df_crm.columns for col in required_cols):
                st.warning("Data belum cukup untuk analisa CRM")
            else:
                df_crm["Issue Time"] = pd.to_datetime(df_crm["Issue Time"], errors="coerce", dayfirst=True)
                df_crm = df_crm.dropna(subset=["Employee Id", "Issue Time"])

                traveler_stats = (
                    df_crm
                    .groupby("Employee Id")
                    .agg(
                        total_tr=("Travel Request Number", "nunique"),
                        total_booking=("Travel Request Number", "count"),
                        last_booking=("Issue Time", "max"),
                        first_booking=("Issue Time", "min")
                    )
                    .reset_index()
                )

                total_travelers = len(traveler_stats)
                repeat_travelers = (traveler_stats["total_tr"] > 1).sum()
                repeat_rate = repeat_travelers / total_travelers * 100
                avg_booking = traveler_stats["total_booking"].mean()

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Active Travelers", f"{total_travelers:,}")
                col2.metric("Repeat Traveler Rate", f"{repeat_rate:.1f}%")
                col3.metric("Avg Booking / Traveler", f"{avg_booking:.1f}")
                col4.metric("Repeat Travelers", f"{repeat_travelers:,}")

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        with tab4:
            st.markdown("<div class='section-title'>Employee Booking Cohort Analysis</div>", unsafe_allow_html=True)

            cohort_df = build_employee_cohort(df_all)

            if cohort_df.empty:
                st.warning("Data tidak cukup untuk Cohort Analysis (butuh Employee Id & Issue Time).")
            else:
                cohort_pct = cohort_df.copy()
                if 0 in cohort_pct.columns:
                    base = cohort_pct[0].replace(0, np.nan)
                    cohort_pct = cohort_pct.div(base, axis=0) * 100
                else:
                    base = cohort_pct.iloc[:, 0].replace(0, np.nan)
                    cohort_pct = cohort_pct.div(base, axis=0) * 100

                cohort_pct = cohort_pct.round(1)

                text_matrix = cohort_pct.map(
                    lambda v: f"{v:.1f}%" if not np.isnan(v) and v > 0 else ""
                )

                fig = px.imshow(
                    cohort_pct,
                    text_auto=False,
                    aspect="auto",
                    color_continuous_scale=["#ffffff", "#e0c7d8", "#1BA0E2"],
                    zmin=0, zmax=100
                )

                fig.update_traces(
                    text=text_matrix.values,
                    texttemplate="%{text}",
                    textfont=dict(size=10)
                )

                fig.update_layout(
                    title=dict(
                        text="Employee Booking Cohort Heatmap  "
                             "<span style='color:#6a8fa0;font-size:11px;'>"
                             "Retensi relatif terhadap bulan pertama booking (Bulan ke-0 = 100%)"
                             "</span>",
                        font=dict(size=13, color="#2a1a2a"),
                        x=0, xanchor="left"
                    ),
                    xaxis_title="Bulan ke-n sejak booking pertama",
                    yaxis_title="Cohort (Bulan Pertama Booking)",
                    coloraxis_colorbar=dict(
                        title=dict(text="%", font=dict(size=10, color="#6a8fa0")),
                        ticksuffix="%",
                        tickfont=dict(size=9, color="#6a8fa0"),
                        len=0.8
                    ),
                    height=max(400, len(cohort_pct) * 36 + 120),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=60, r=40, t=70, b=60),
                    font=dict(size=11, color="#2a1a2a")
                )

                st.plotly_chart(fig, use_container_width=True)

                output = BytesIO()
                cohort_df.reset_index().to_excel(output, index=False, sheet_name="Employee Cohort")
                output.seek(0)

                if st.session_state.get('role') == 'Admin':
                    st.download_button(
                        label="⬇️ Download Data",
                        data=output,
                        file_name="employee_booking_cohort.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_employee_cohort")
                else:
                    st.markdown("""
                    <div style='background:#f9f9f9;border:1px solid #b8d9f0;border-left:3px solid #1BA0E2;
                    border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;
                    display:flex;align-items:center;gap:8px;'>
                        <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                    </div>""", unsafe_allow_html=True)

                with st.expander("📖 Panduan Membaca Cohort Heatmap", expanded=False):
                    _cohort_narasi = (
                        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:4px;'>"
                        "<div style='background:#f0f8ff;border-radius:9px;padding:14px 16px;border:1px solid #b8d9f0;'>"
                        "<div style='font-weight:700;color:#6a1a5a;font-size:0.83em;margin-bottom:8px;'>&#128269; Apa itu Cohort Heatmap?</div>"
                        "<div style='font-size:0.78em;color:#4a3a4a;line-height:1.7;'>Cohort Heatmap mengelompokkan karyawan berdasarkan <b>bulan pertama kali mereka melakukan booking</b> (cohort). Nilai pada setiap sel adalah <b>persentase retensi</b>.</div></div>"
                        "<div style='background:#f0f8ff;border-radius:9px;padding:14px 16px;border:1px solid #b8d9f0;'>"
                        "<div style='font-weight:700;color:#6a1a5a;font-size:0.83em;margin-bottom:8px;'>&#127919; Kegunaan Analisis Ini</div>"
                        "<div style='font-size:0.78em;color:#4a3a4a;line-height:1.8;'>&#8226; <b>Pantau loyalitas traveler</b><br>&#8226; <b>Deteksi penurunan aktivitas</b><br>&#8226; <b>Evaluasi kebijakan travel</b><br>&#8226; <b>Benchmark antar periode</b></div></div>"
                        "<div style='background:#f0f8ff;border-radius:9px;padding:14px 16px;border:1px solid #b8d9f0;'>"
                        "<div style='font-weight:700;color:#6a1a5a;font-size:0.83em;margin-bottom:8px;'>&#128202; Cara Membaca</div>"
                        "<div style='font-size:0.78em;color:#4a3a4a;line-height:1.8;'>&#8226; <b>Kolom 0</b> = bulan pertama &#8594; selalu <b>100%</b><br>&#8226; <b>Warna gelap</b> = retensi tinggi &#9989;<br>&#8226; <b>Warna terang</b> = retensi rendah &#9888;&#65039;</div></div>"
                        "</div>"
                    )
                    st.markdown(_cohort_narasi, unsafe_allow_html=True)

                today = df_crm["Issue Time"].max()

                traveler_stats["Recency (Days)"] = (today - traveler_stats["last_booking"]).dt.days

                if "Invoice Amount" in df_crm.columns:
                    spend = (df_crm.groupby("Employee Id")["Invoice Amount"].sum().reset_index(name="Total Spend"))
                    traveler_stats = traveler_stats.merge(spend, on="Employee Id", how="left")
                else:
                    traveler_stats["Total Spend"] = 0

                def segment(row):
                    if row["total_tr"] >= 10: return "High Value"
                    elif row["total_tr"] >= 3: return "Medium Value"
                    else: return "Low Value"

                traveler_stats["Segment"] = traveler_stats.apply(segment, axis=1)

                st.markdown("<div class='section-title'>Top Valuable Travelers</div>", unsafe_allow_html=True)

                top_travelers = traveler_stats.sort_values(by=["total_tr", "Total Spend"], ascending=False).head(10)
                display_cols = ["Employee Id", "total_tr", "total_booking", "Total Spend", "Segment"]
                numeric_cols = ["total_tr", "total_booking", "Total Spend"]

                st.dataframe(
                    top_travelers[display_cols]
                        .style
                        .format({"Total Spend": lambda x: f"Rp{x:,.0f}" if pd.notnull(x) else "Rp 0"})
                        .set_properties(subset=numeric_cols, **{"text-align": "right"}),
                    use_container_width=True
                )

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                if True:  # (sebelumnya "with tab4:" redundan — sudah berada di dalam tab4, cukup dihilangkan tanpa mengubah indentasi di bawahnya)
                    st.markdown("### Behavioral Persona Clustering")

                    df_behavior = df_all.copy()
                    selected_data = pd.DataFrame()       # default agar aman jika kolom tidak lengkap
                    selected_cluster = None              # default — dipakai di blok radar chart di bawah
                    employee_features = pd.DataFrame()   # default — dipakai di blok radar/insight di bawah
                    feature_cols = []                    # default — dipakai di blok radar/insight di bawah

                    required_cols = ["Travel Request Number","Employee Id","Issue Time","Check in Date","Check out Date","Number of Rooms Night"]

                    if all(col in df_behavior.columns for col in required_cols):

                        # PERBAIKAN PERFORMA: pipeline clustering (parsing tanggal, groupby,
                        # scaling, fit KMeans) dipindah ke fungsi ber-cache — hasilnya sama
                        # persis berapa pun kali dropdown Employee Id di bawah ini diganti,
                        # jadi tidak perlu dihitung ulang setiap kali (sebelumnya inilah yang
                        # membuat mengganti Employee Id terasa memicu render ulang seluruh halaman).
                        _df_behavior_slim = df_behavior[required_cols].copy()
                        employee_features, feature_cols = build_employee_persona_clusters(_df_behavior_slim)

                        # PERBAIKAN: dropdown Employee Id diganti free-text search sesuai permintaan —
                        # user mengetik (sebagian/seluruh) Employee Id, bukan memilih dari daftar dropdown.
                        _emp_search = st.text_input(
                            "🔍 Cari Employee Id",
                            value="",
                            placeholder="Ketik Employee ID (boleh sebagian)...",
                            key="persona_employee_search"
                        )

                        selected_data = pd.DataFrame()

                        if not _emp_search.strip():
                            st.info("ℹ️ Ketik Employee ID di atas untuk melihat profil persona-nya.")
                        else:
                            _emp_matches = employee_features[
                                employee_features["Employee Id"].astype(str)
                                .str.contains(_emp_search.strip(), case=False, na=False, regex=False)
                            ]

                            if _emp_matches.empty:
                                st.warning(f"⚠️ Employee Id yang mengandung \"{_emp_search.strip()}\" tidak ditemukan.")
                            elif len(_emp_matches) == 1:
                                selected_data = _emp_matches
                                selected_employee = selected_data["Employee Id"].values[0]
                                selected_cluster = selected_data["Cluster"].values[0]
                                selected_persona = selected_data["Persona"].values[0]
                                st.success(f"Employee Id: {selected_employee} — Persona: {selected_persona}")
                            else:
                                st.info(
                                    f"🔎 Ditemukan {len(_emp_matches)} Employee Id yang cocok — "
                                    f"perjelas ketikan Anda untuk mempersempit ke satu hasil."
                                )
                                _preview_cols = [c for c in ["Employee Id", "Persona", "Booking_Frequency"] if c in _emp_matches.columns]
                                st.dataframe(
                                    _emp_matches[_preview_cols].head(15),
                                    use_container_width=True, hide_index=True
                                )
                                if len(_emp_matches) > 15:
                                    st.caption(f"Menampilkan 15 dari {len(_emp_matches)} hasil.")

                if not selected_data.empty:
                    st.markdown("""
                    <style>
                    .metric-card{background:white;padding:20px 16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);text-align:center;transition:all 0.3s ease;margin-bottom:12px;border:1px solid #f0f0f0;border-top:3px solid #1BA0E2;}
                    .metric-card:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(156,87,137,0.12);}
                    .metric-value{font-size:32px;font-weight:700;margin:8px 0;color:#1BA0E2;}
                    .metric-label{font-size:11px;color:#888888;text-transform:uppercase;letter-spacing:1px;font-weight:500;}
                    .insight-card{background:white;border-radius:6px;padding:16px;margin-bottom:10px;border-left:4px solid;box-shadow:0 1px 4px rgba(0,0,0,0.05);font-size:14px;line-height:1.6;}
                    .insight-success{border-left-color:#10b981;background:linear-gradient(to right,#ecfdf5,white);}
                    .insight-warning{border-left-color:#f59e0b;background:linear-gradient(to right,#fffbeb,white);}
                    .insight-info{border-left-color:#1BA0E2;background:linear-gradient(to right,#f8f4f7,white);}
                    .insight-error{border-left-color:#ef4444;background:linear-gradient(to right,#fef2f2,white);}
                    .persona-badge{background:linear-gradient(135deg,#1BA0E2 0%,#75caf0 100%);padding:24px;border-radius:8px;text-align:center;color:white;font-size:19px;font-weight:600;box-shadow:0 4px 16px rgba(156,87,137,0.2);margin:15px 0;}
                    .section-header{font-size:18px;font-weight:600;color:#1a1a1a;margin:30px 0 18px 0;padding-bottom:8px;border-bottom:2px solid #1BA0E2;}
                    .progress-container{background:#f0f0f0;height:6px;border-radius:3px;overflow:hidden;margin-top:6px;}
                    .progress-bar{height:100%;border-radius:3px;transition:width 0.4s ease;}
                    .breakdown-card{background:white;padding:12px 14px;border-radius:6px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,0.04);border:1px solid #f0f0f0;}
                    </style>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 1.4], gap="large")

                    with col1:
                        st.markdown('<div class="section-header">📊 Behavioral Overview</div>', unsafe_allow_html=True)
                        bf = selected_data["Booking_Frequency"].values[0]
                        lead = selected_data["Avg_Lead_Time"].values[0]
                        lf = selected_data["Last_Minute_Ratio"].values[0]
                        weekend = selected_data["Weekend_Ratio"].values[0]
                        stay = selected_data["Avg_Stay"].values[0]

                        st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        with m1:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Booking Frequency</div><div class="metric-value">{round(bf,1)}</div></div>', unsafe_allow_html=True)
                        with m2:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Lead Time (Days)</div><div class="metric-value">{round(lead,1)}</div></div>', unsafe_allow_html=True)
                        m3, m4 = st.columns(2)
                        with m3:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Last Minute Ratio</div><div class="metric-value">{round(lf*100,1)}%</div></div>', unsafe_allow_html=True)
                        with m4:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Weekend Ratio</div><div class="metric-value">{round(weekend*100,1)}%</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Average Stay Duration</div><div class="metric-value">{round(stay,1)} <span style="font-size:18px;font-weight:500;">nights</span></div></div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="section-header">🎯 Behavioral Radar Profile</div>', unsafe_allow_html=True)

                        from sklearn.preprocessing import MinMaxScaler
                        cluster_profile = employee_features.groupby("Cluster")[feature_cols].mean()
                        minmax_scaler = MinMaxScaler()
                        cluster_scaled = minmax_scaler.fit_transform(cluster_profile)

                        profile_row = cluster_scaled[selected_cluster]
                        radar_values = list(profile_row) + [profile_row[0]]
                        radar_labels = feature_cols + [feature_cols[0]]

                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(r=radar_values, theta=radar_labels, fill='toself',
                            line=dict(width=4, color="#1BA0E2"), fillcolor="rgba(156,87,137,0.25)", name='Profile'))
                        fig.add_trace(go.Scatterpolar(r=[0.5]*len(radar_labels), theta=radar_labels,
                            line=dict(width=2, color="rgba(138,77,120,0.4)", dash='dash'), name='Benchmark'))

                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                            showlegend=True, height=580,
                            margin=dict(l=50,r=50,t=50,b=50),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                if not selected_data.empty:
                    col1, col2 = st.columns([1, 1.4], gap="large")

                    with col1:
                        st.markdown('<div class="section-header">💡 Behavioral Insights</div>', unsafe_allow_html=True)
                        if lf > 0.5:
                            st.markdown('<div class="insight-card insight-error"><strong>⚠️ Reactive Traveler</strong><br>High last-minute booking ratio detected.</div>', unsafe_allow_html=True)
                        elif lead > 7:
                            st.markdown('<div class="insight-card insight-success"><strong>✅ Strategic Planner</strong><br>Excellent advance planning.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="insight-card insight-info"><strong>ℹ️ Balanced Approach</strong><br>Shows balanced booking behavior.</div>', unsafe_allow_html=True)

                        avg_bf = employee_features["Booking_Frequency"].mean()
                        if bf > avg_bf:
                            intensity_pct = ((bf - avg_bf) / avg_bf * 100)
                            st.markdown(f'<div class="insight-card insight-warning"><strong>📊 High Activity</strong><br>Travel intensity <strong>{round(intensity_pct,1)}%</strong> above peer average.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="insight-card insight-success"><strong>📊 Normal Activity</strong><br>Travel intensity aligns with baseline.</div>', unsafe_allow_html=True)

                        if weekend > 0.4:
                            st.markdown('<div class="insight-card insight-info"><strong>🌅 Weekend Preference</strong><br>Strong weekend tendency.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="insight-card insight-info"><strong>💼 Weekday Focus</strong><br>Primarily weekday travel.</div>', unsafe_allow_html=True)

                        avg_stay_val = employee_features["Avg_Stay"].mean()
                        if stay > avg_stay_val:
                            st.markdown(f'<div class="insight-card insight-warning"><strong>🏨 Extended Stays</strong><br><strong>{round(stay-avg_stay_val,1)}</strong> nights above average.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="insight-card insight-success"><strong>🏨 Quick Visits</strong><br>Efficient short trips.</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="section-header">📋 Metric Breakdown</div>', unsafe_allow_html=True)
                        for i, feature in enumerate(feature_cols):
                            score = profile_row[i]
                            if score > 0.7: status, status_icon, color = "High", "🔴", "#1BA0E2"
                            elif score > 0.4: status, status_icon, color = "Medium", "🟡", "#75caf0"
                            else: status, status_icon, color = "Low", "🟢", "#e7c3d9"
                            progress_width = int(score * 100)
                            st.markdown(f"""
                            <div class="breakdown-card">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                    <strong style="color:#1a1a1a;font-size:13px;">{feature.replace("_"," ")}</strong>
                                    <span style="color:#888;font-weight:500;font-size:11px;margin:0 10px;">{score:.2f}</span>
                                    <span style="color:{color};font-weight:600;font-size:11px;">{status_icon} {status}</span>
                                </div>
                                <div class="progress-container">
                                    <div class="progress-bar" style="background:{color};width:{progress_width}%;"></div>
                                </div>
                            </div>""", unsafe_allow_html=True)

        # ======================================
        # TAB 5: SOCIAL NETWORK ANALYSIS
        # ======================================
        # ======================================
        # TAB 5: SOCIAL NETWORK ANALYSIS — MTRAX Blue Theme
        # ======================================
        with tab5:

            # ── Header Banner ──────────────────────────────────────────────
            st.markdown("""
            <style>
            .sna-header {
                background: linear-gradient(135deg, #062440 0%, #0D7FCC 55%, #1BA0E2 100%);
                border-radius: 14px;
                padding: 30px 36px;
                margin-bottom: 24px;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(13,127,204,0.28);
            }
            .sna-header::before {
                content: '';
                position: absolute; top: -60px; right: -60px;
                width: 260px; height: 260px; border-radius: 50%;
                background: rgba(255,255,255,0.05);
            }
            .sna-header::after {
                content: '';
                position: absolute; bottom: -40px; left: 30%;
                width: 180px; height: 180px; border-radius: 50%;
                background: rgba(226,248,113,0.07);
            }
            .sna-header-inner {
                display: flex; align-items: center; gap: 16px;
                position: relative; z-index: 1;
            }
            .sna-icon-wrap {
                width: 52px; height: 52px; border-radius: 12px;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.22);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.6em; flex-shrink: 0;
            }
            .sna-title { color: #fff; font-size: 1.45em; font-weight: 700; margin: 0; }
            .sna-sub { color: rgba(255,255,255,0.60); font-size: 0.82em; margin-top: 4px; }
            .sna-pill {
                margin-left: auto;
                display: inline-flex; align-items: center; gap: 7px;
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(255,255,255,0.22);
                border-radius: 100px; padding: 6px 16px;
                font-size: 0.72em; font-weight: 600;
                color: rgba(255,255,255,0.85); letter-spacing: 0.06em;
            }
            .sna-pill-dot {
                width: 6px; height: 6px; border-radius: 50%;
                background: #e2f871;
                box-shadow: 0 0 6px #e2f871;
                animation: sna-blink 2s ease infinite;
            }
            @keyframes sna-blink { 0%,100%{opacity:1} 50%{opacity:.3} }

            .sna-stat {
                background: #fff;
                border-radius: 12px;
                padding: 20px 18px 16px;
                border-top: 3px solid var(--sna-accent, #1BA0E2);
                box-shadow: 0 2px 12px rgba(13,127,204,0.08);
                text-align: center;
                transition: transform .18s ease, box-shadow .18s ease;
                margin-bottom: 16px;
            }
            .sna-stat:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(13,127,204,0.16);
            }
            .sna-stat-icon { font-size: 1.5em; margin-bottom: 8px; display: block; }
            .sna-stat-label {
                font-size: 0.68em; font-weight: 600; color: #8a9aaa;
                text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 6px;
            }
            .sna-stat-value { font-size: 1.45em; font-weight: 700; }

            .sna-filter-bar {
                background: #f0f8ff;
                border: 1px solid #cce4f4;
                border-left: 4px solid #1BA0E2;
                border-radius: 8px;
                padding: 12px 18px 8px;
                margin-bottom: 14px;
            }
            .sna-filter-label {
                font-size: 0.70em; font-weight: 700;
                color: #1BA0E2; text-transform: uppercase;
                letter-spacing: 0.10em; margin-bottom: 2px;
            }

            .sna-rank-head {
                background: linear-gradient(135deg, #f0f8ff 0%, #ffffff 100%);
                border-radius: 10px; padding: 14px 18px 4px;
                border-left: 3px solid #1BA0E2; margin-bottom: 8px;
                box-shadow: 0 1px 6px rgba(13,127,204,0.07);
            }
            .sna-rank-title {
                font-size: 0.78em; font-weight: 700; color: #1BA0E2;
                text-transform: uppercase; letter-spacing: 0.07em;
            }
            .sna-rank-head-hotel {
                background: linear-gradient(135deg, #eef3fb 0%, #ffffff 100%);
                border-radius: 10px; padding: 14px 18px 4px;
                border-left: 3px solid #0D7FCC; margin-bottom: 8px;
                box-shadow: 0 1px 6px rgba(13,127,204,0.07);
            }
            .sna-rank-title-hotel {
                font-size: 0.78em; font-weight: 700; color: #0D7FCC;
                text-transform: uppercase; letter-spacing: 0.07em;
            }
            .sna-empty-warn {
                background: #f0f8ff; border: 1px solid #cce4f4;
                border-left: 4px solid #1BA0E2; border-radius: 8px;
                padding: 16px 20px; font-size: 0.88em; color: #0D7FCC;
            }
            </style>

            <div class="sna-header">
                <div class="sna-header-inner">
                    <div class="sna-icon-wrap">🕸️</div>
                    <div>
                        <div class="sna-title">Social Network Analysis</div>
                        <div class="sna-sub">Employee ↔ Hotel Interaction Network · Centrality &amp; Dependency Mapping</div>
                    </div>
                    <div class="sna-pill">
                        <span class="sna-pill-dot"></span>NETWORK GRAPH
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            required_cols = ["Employee Id", "Hotel Name"]

            if all(col in df_all.columns for col in required_cols):

                df_sna = (df_all.dropna(subset=required_cols)
                          .groupby(required_cols).size().reset_index(name="weight"))

                total_emp_count = df_sna["Employee Id"].nunique()
                total_htl_count = df_sna["Hotel Name"].nunique()
                total_edges     = len(df_sna)
                avg_connections = df_sna.groupby("Employee Id")["weight"].sum().mean()

                stat_cols = st.columns(4)
                stat_data = [
                    ("👤", "Total Employees",    f"{total_emp_count:,}",   "#1BA0E2"),
                    ("🏨", "Total Hotels",       f"{total_htl_count:,}",   "#1494C6"),
                    ("🔗", "Total Interactions", f"{total_edges:,}",        "#0D7FCC"),
                    ("📊", "Avg Trips / Emp",    f"{avg_connections:.1f}", "#062440"),
                ]
                for col, (icon, label, val, color) in zip(stat_cols, stat_data):
                    with col:
                        st.markdown(f"""
                        <div class="sna-stat" style="--sna-accent:{color};">
                            <span class="sna-stat-icon">{icon}</span>
                            <div class="sna-stat-label">{label}</div>
                            <div class="sna-stat-value" style="color:{color};">{val}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("""
                <div class="sna-filter-bar">
                    <div class="sna-filter-label">⚙ Network Parameters</div>
                </div>""", unsafe_allow_html=True)

                fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
                with fcol1:
                    top_emp = st.slider("👤 Top Employees", 5, min(100, total_emp_count), min(50, total_emp_count))
                with fcol2:
                    top_htl = st.slider("🏨 Top Hotels", 5, min(50, total_htl_count), min(15, total_htl_count))
                with fcol3:
                    layout_algo = st.selectbox("📐 Layout", ["Spring", "Kamada-Kawai", "Circular"])

                top_employees = (df_sna.groupby("Employee Id")["weight"].sum()
                                 .sort_values(ascending=False).head(top_emp).index)
                top_hotels = (df_sna.groupby("Hotel Name")["weight"].sum()
                              .sort_values(ascending=False).head(top_htl).index)
                df_filtered_sna = df_sna[
                    df_sna["Employee Id"].isin(top_employees) &
                    df_sna["Hotel Name"].isin(top_hotels)
                ]

                G = nx.Graph()
                for _, row in df_filtered_sna.iterrows():
                    G.add_edge(row["Employee Id"], row["Hotel Name"], weight=row["weight"])

                seed = 42
                if layout_algo == "Spring":
                    pos = nx.spring_layout(G, seed=seed, k=0.7)
                elif layout_algo == "Kamada-Kawai":
                    try:
                        pos = nx.kamada_kawai_layout(G)
                    except:
                        pos = nx.spring_layout(G, seed=seed)
                else:
                    pos = nx.circular_layout(G)

                degree      = dict(G.degree())
                betweenness = nx.betweenness_centrality(G)
                max_weight  = max((G[u][v]["weight"] for u, v in G.edges()), default=1)
                max_degree  = max(degree.values(), default=1)

                # Edge traces — blue tones
                edge_traces = []
                for u, v in G.edges():
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    w = G[u][v]["weight"]
                    opacity = 0.12 + 0.55 * (w / max_weight)
                    width   = 0.5  + 3.0  * (w / max_weight)
                    edge_traces.append(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                        line=dict(width=width, color=f"rgba(27,160,226,{opacity:.2f})"),
                        hoverinfo="none", showlegend=False
                    ))

                # Employee nodes — blue gradient
                emp_x, emp_y, emp_text, emp_size, emp_mc = [], [], [], [], []
                for node in G.nodes():
                    if node not in top_employees:
                        continue
                    x, y = pos[node]
                    deg  = degree[node]
                    bet  = betweenness.get(node, 0)
                    size = 14 + (deg / max_degree) * 30
                    total_trips    = df_filtered_sna[df_filtered_sna["Employee Id"] == node]["weight"].sum()
                    hotels_visited = df_filtered_sna[df_filtered_sna["Employee Id"] == node]["Hotel Name"].nunique()
                    emp_x.append(x); emp_y.append(y)
                    emp_size.append(size); emp_mc.append(deg)
                    emp_text.append(
                        f"<b>👤 {node}</b><br>"
                        f"Hotel Connections: <b>{deg}</b><br>"
                        f"Total Stays: <b>{int(total_trips):,}</b><br>"
                        f"Unique Hotels: <b>{hotels_visited}</b><br>"
                        f"Betweenness: <b>{bet:.3f}</b>"
                    )

                employee_trace = go.Scatter(
                    x=emp_x, y=emp_y, mode="markers", name="Employee",
                    hoverinfo="text", text=emp_text,
                    marker=dict(
                        size=emp_size, color=emp_mc,
                        colorscale=[
                            [0.0, "#cce4f4"],
                            [0.4, "#1BA0E2"],
                            [0.7, "#1494C6"],
                            [1.0, "#062440"]
                        ],
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Degree<br>(Employee)",
                                       font=dict(size=10, color="#6a8fa0")),
                            thickness=10, len=0.45, y=0.75, x=1.01,
                            tickfont=dict(size=9, color="#6a8fa0")
                        ),
                        line=dict(width=2, color="white"),
                        symbol="circle"
                    )
                )

                # Hotel nodes — lime accent + dark blue
                htl_x, htl_y, htl_text, htl_size, htl_mc = [], [], [], [], []
                for node in G.nodes():
                    if node not in top_hotels:
                        continue
                    x, y = pos[node]
                    deg  = degree[node]
                    bet  = betweenness.get(node, 0)
                    size = 18 + (deg / max_degree) * 28
                    total_stays = df_filtered_sna[df_filtered_sna["Hotel Name"] == node]["weight"].sum()
                    unique_emps = df_filtered_sna[df_filtered_sna["Hotel Name"] == node]["Employee Id"].nunique()
                    htl_x.append(x); htl_y.append(y)
                    htl_size.append(size); htl_mc.append(deg)
                    htl_text.append(
                        f"<b>🏨 {node}</b><br>"
                        f"Employee Connections: <b>{deg}</b><br>"
                        f"Total Stays: <b>{int(total_stays):,}</b><br>"
                        f"Unique Travelers: <b>{unique_emps}</b><br>"
                        f"Betweenness: <b>{bet:.3f}</b>"
                    )

                hotel_trace = go.Scatter(
                    x=htl_x, y=htl_y, mode="markers", name="Hotel",
                    hoverinfo="text", text=htl_text,
                    marker=dict(
                        size=htl_size, color=htl_mc,
                        colorscale=[
                            [0.0, "#d8f0b0"],
                            [0.3, "#98ea16"],
                            [0.6, "#0D7FCC"],
                            [1.0, "#062440"]
                        ],
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Degree<br>(Hotel)",
                                       font=dict(size=10, color="#6a8fa0")),
                            thickness=10, len=0.45, y=0.28, x=1.01,
                            tickfont=dict(size=9, color="#6a8fa0")
                        ),
                        line=dict(width=2, color="white"),
                        symbol="diamond"
                    )
                )

                fig = go.Figure(
                    data=edge_traces + [employee_trace, hotel_trace],
                    layout=go.Layout(
                        title=dict(
                            text=(
                                f"<b>Employee ↔ Hotel Network</b>"
                                f"<span style='font-size:11px;color:#8a9aaa;'>"
                                f"  ·  Top {top_emp} Employees"
                                f"  ·  Top {top_htl} Hotels"
                                f"  ·  {layout_algo} Layout</span>"
                            ),
                            font=dict(size=15, color="#1a2a3a"),
                            x=0.0, xanchor="left"
                        ),
                        showlegend=True,
                        legend=dict(
                            bgcolor="rgba(240,248,255,0.95)",
                            bordercolor="#cce4f4", borderwidth=1,
                            font=dict(size=11, color="#1a2a3a")
                        ),
                        hovermode="closest",
                        margin=dict(b=60, l=10, r=90, t=70),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor="#f0f8ff",
                        paper_bgcolor="white",
                        height=660
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                # ── Rank Tables ─────────────────────────────────────────────
                rank_col1, rank_col2 = st.columns(2)

                emp_rank = []
                for node in top_employees:
                    if node not in G.nodes():
                        continue
                    deg = degree.get(node, 0)
                    bet = betweenness.get(node, 0)
                    total_stays = df_filtered_sna[df_filtered_sna["Employee Id"] == node]["weight"].sum()
                    emp_rank.append({
                        "Employee ID": str(node),
                        "Connections": deg,
                        "Total Stays": int(total_stays),
                        "Centrality":  round(bet, 4)
                    })
                emp_rank_df = (pd.DataFrame(emp_rank)
                               .sort_values("Connections", ascending=False)
                               .head(10).reset_index(drop=True))
                emp_rank_df.index = emp_rank_df.index + 1
                emp_rank_df.index.name = "Rank"

                htl_rank = []
                for node in top_hotels:
                    if node not in G.nodes():
                        continue
                    deg = degree.get(node, 0)
                    bet = betweenness.get(node, 0)
                    total_stays = df_filtered_sna[df_filtered_sna["Hotel Name"] == node]["weight"].sum()
                    unique_emps = df_filtered_sna[df_filtered_sna["Hotel Name"] == node]["Employee Id"].nunique()
                    htl_rank.append({
                        "Hotel Name":  str(node),
                        "Travelers":   deg,
                        "Total Stays": int(total_stays),
                        "Centrality":  round(bet, 4)
                    })
                htl_rank_df = (pd.DataFrame(htl_rank)
                               .sort_values("Travelers", ascending=False)
                               .head(10).reset_index(drop=True))
                htl_rank_df.index = htl_rank_df.index + 1
                htl_rank_df.index.name = "Rank"

                with rank_col1:
                    st.markdown("""
                    <div class="sna-rank-head">
                        <div class="sna-rank-title">👤 Top Employees by Connectivity</div>
                    </div>""", unsafe_allow_html=True)
                    st.dataframe(
                        emp_rank_df.style.background_gradient(
                            subset=["Connections", "Total Stays"], cmap="Blues"
                        ),
                        use_container_width=True
                    )

                with rank_col2:
                    st.markdown("""
                    <div class="sna-rank-head-hotel">
                        <div class="sna-rank-title-hotel">🏨 Top Hotels by Dependency Risk</div>
                    </div>""", unsafe_allow_html=True)
                    st.dataframe(
                        htl_rank_df.style.background_gradient(
                            subset=["Travelers", "Total Stays"], cmap="PuBu"
                        ),
                        use_container_width=True
                    )

            else:
                st.markdown("""
                <div class="sna-empty-warn">
                    ⚠️ Kolom <strong>Employee Id</strong> atau
                    <strong>Hotel Name</strong> tidak tersedia dalam dataset.
                </div>""", unsafe_allow_html=True)

        # ======================================
        # TAB 6: PRICE INTELLIGENCE — Global Filters
        # ======================================
        with tab6:
            def _render_tab6():

                st.markdown("""
                <style>
                :root {
                    --pi-blue:      #1BA0E2;
                    --pi-blue-mid:  #1494C6;
                    --pi-blue-dark: #0D7FCC;
                    --pi-navy:      #062440;
                    --pi-lime:      #e2f871;
                    --pi-bg:        #f0f8ff;
                    --pi-border:    #cce4f4;
                    --pi-surface:   #ffffff;
                    --pi-muted:     #6a8fa0;
                }

                .pi-header {
                    background: linear-gradient(135deg, #062440 0%, #0D7FCC 50%, #1BA0E2 100%);
                    border-radius: 14px; padding: 28px 36px; margin-bottom: 24px;
                    position: relative; overflow: hidden;
                    box-shadow: 0 8px 32px rgba(13,127,204,0.26);
                }
                .pi-header::before {
                    content: ''; position: absolute; top: -70px; right: -50px;
                    width: 240px; height: 240px; border-radius: 50%;
                    background: rgba(255,255,255,0.05); pointer-events: none;
                }
                .pi-header::after {
                    content: ''; position: absolute; bottom: -50px; left: 25%;
                    width: 200px; height: 200px; border-radius: 50%;
                    background: rgba(226,248,113,0.07); pointer-events: none;
                }
                .pi-header-inner {
                    display: flex; align-items: center; gap: 16px;
                    position: relative; z-index: 1;
                }
                .pi-header-icon {
                    width: 52px; height: 52px; border-radius: 12px;
                    background: rgba(255,255,255,0.14);
                    border: 1px solid rgba(255,255,255,0.22);
                    display: flex; align-items: center; justify-content: center;
                    font-size: 1.6em; flex-shrink: 0;
                }
                .pi-header-title { color: #fff; font-size: 1.45em; font-weight: 700; margin: 0; }
                .pi-header-sub   { color: rgba(255,255,255,0.58); font-size: 0.82em; margin-top: 4px; }
                .pi-header-badge {
                    margin-left: auto;
                    display: inline-flex; align-items: center; gap: 7px;
                    background: rgba(255,255,255,0.12);
                    border: 1px solid rgba(255,255,255,0.22);
                    border-radius: 100px; padding: 6px 16px;
                    font-size: 0.72em; font-weight: 600;
                    color: rgba(255,255,255,0.85); letter-spacing: 0.06em;
                    white-space: nowrap;
                }
                .pi-badge-dot {
                    width: 6px; height: 6px; border-radius: 50%;
                    background: var(--pi-lime); box-shadow: 0 0 6px var(--pi-lime);
                    animation: pi-blink 2s ease infinite;
                }
                @keyframes pi-blink { 0%,100%{opacity:1} 50%{opacity:.3} }

                /* ── Global Filter Bar ── */
                .pi-global-filter {
                    background: var(--pi-surface);
                    border: 1px solid var(--pi-border);
                    border-left: 4px solid var(--pi-blue);
                    border-radius: 10px; padding: 16px 22px 14px;
                    margin-bottom: 24px;
                    box-shadow: 0 2px 10px rgba(13,127,204,0.07);
                }
                .pi-global-filter-title {
                    font-size: 0.70em; font-weight: 700; color: var(--pi-blue);
                    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px;
                    display: flex; align-items: center; gap: 8px;
                }
                .pi-filter-active-badge {
                    display: inline-flex; align-items: center; gap: 5px;
                    background: var(--pi-blue); color: white;
                    border-radius: 100px; padding: 2px 10px;
                    font-size: 0.85em; font-weight: 600; letter-spacing: 0.04em;
                }
                .pi-filter-dot {
                    width: 5px; height: 5px; border-radius: 50%;
                    background: var(--pi-lime); display: inline-block;
                }

                /* ── Section Divider ── */
                .pi-section-divider {
                    display: flex; align-items: center; gap: 14px;
                    margin: 28px 0 20px;
                    font-size: 0.70em; font-weight: 700;
                    color: var(--pi-muted); letter-spacing: 0.14em;
                    text-transform: uppercase;
                }
                .pi-section-divider::before,
                .pi-section-divider::after {
                    content: ''; flex: 1; height: 1px; background: var(--pi-border);
                }

                /* ── KPI Cards ── */
                .pi-kpi-grid {
                    display: grid; grid-template-columns: repeat(4,1fr);
                    gap: 14px; margin-bottom: 20px;
                }
                .pi-kpi {
                    background: var(--pi-surface); border-radius: 10px;
                    padding: 18px 16px 14px;
                    border-top: 3px solid var(--pi-blue);
                    box-shadow: 0 2px 10px rgba(13,127,204,0.08);
                    transition: transform .18s, box-shadow .18s;
                }
                .pi-kpi:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(13,127,204,0.14); }
                .pi-kpi-label {
                    font-size: 0.68em; font-weight: 600; color: var(--pi-muted);
                    text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 8px;
                }
                .pi-kpi-value { font-size: 1.30em; font-weight: 700; color: var(--pi-navy); line-height: 1.2; }
                .pi-kpi-sub   { font-size: 0.72em; color: var(--pi-muted); margin-top: 4px; }

                /* ── Hotel KPI Cards ── */
                .pi-hotel-kpi-grid {
                    display: grid; grid-template-columns: repeat(5,1fr);
                    gap: 12px; margin-bottom: 22px;
                }
                .pi-hotel-kpi {
                    background: var(--pi-surface); border-radius: 10px;
                    padding: 16px 14px 12px;
                    border-top: 3px solid var(--pi-color, #1BA0E2);
                    box-shadow: 0 2px 8px rgba(13,127,204,0.07);
                    text-align: center;
                    transition: transform .18s, box-shadow .18s;
                }
                .pi-hotel-kpi:hover { transform: translateY(-2px); box-shadow: 0 5px 16px rgba(13,127,204,0.13); }
                .pi-hotel-kpi-icon  { font-size: 1.35em; margin-bottom: 6px; display: block; }
                .pi-hotel-kpi-label {
                    font-size: 0.65em; font-weight: 600; color: var(--pi-muted);
                    text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 6px;
                }
                .pi-hotel-kpi-value {
                    font-size: 1.05em; font-weight: 700;
                    color: var(--pi-color, #1BA0E2); line-height: 1.25;
                }

                /* ── Saving Sim Cards ── */
                .pi-sim-grid {
                    display: grid; grid-template-columns: repeat(3,1fr);
                    gap: 14px; margin: 16px 0;
                }
                .pi-sim-card {
                    border-radius: 10px; padding: 20px 18px;
                    border: 1px solid var(--pi-border);
                }
                .pi-sim-card-label {
                    font-size: 0.68em; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 10px;
                }
                .pi-sim-card-value { font-size: 1.30em; font-weight: 700; line-height: 1.2; }
                .pi-sim-card-sub   { font-size: 0.74em; margin-top: 5px; }

                .pi-sim-current { background: var(--pi-bg); border-color: var(--pi-border); }
                .pi-sim-current .pi-sim-card-label { color: var(--pi-blue); }
                .pi-sim-current .pi-sim-card-value { color: var(--pi-navy); }
                .pi-sim-current .pi-sim-card-sub   { color: var(--pi-muted); }

                .pi-sim-neg { background: #f0fbf4; border-color: #b8e8cc; }
                .pi-sim-neg .pi-sim-card-label { color: #2e8a57; }
                .pi-sim-neg .pi-sim-card-value { color: #1a5c38; }
                .pi-sim-neg .pi-sim-card-sub   { color: #5aaa7e; }

                .pi-sim-saving {
                    background: linear-gradient(135deg, #062440 0%, #0D7FCC 55%, #1BA0E2 100%);
                    border-color: transparent; position: relative; overflow: hidden;
                }
                .pi-sim-saving::after {
                    content: ''; position: absolute; top: -30px; right: -30px;
                    width: 120px; height: 120px; border-radius: 50%;
                    background: rgba(226,248,113,0.12);
                }
                .pi-sim-saving .pi-sim-card-label { color: rgba(255,255,255,0.65); position: relative; z-index: 1; }
                .pi-sim-saving .pi-sim-card-value { color: #fff; font-size: 1.45em; position: relative; z-index: 1; }
                .pi-sim-saving .pi-sim-card-sub   { color: rgba(255,255,255,0.60); position: relative; z-index: 1; }
                .pi-sim-saving-icon { display: block; font-size: 1.4em; margin-bottom: 6px; position: relative; z-index: 1; }

                /* ── Insight Box ── */
                .pi-insight {
                    background: var(--pi-bg); border: 1px solid var(--pi-border);
                    border-left: 4px solid var(--pi-blue); border-radius: 8px;
                    padding: 16px 20px; font-size: 0.88em;
                    color: #2a3a4a; line-height: 1.7;
                }
                .pi-insight strong { color: var(--pi-navy); }
                .pi-insight em     { color: var(--pi-blue); font-style: normal; font-weight: 600; }

                .pi-locked {
                    background: var(--pi-bg); border: 1px solid var(--pi-border);
                    border-left: 3px solid var(--pi-blue); border-radius: 6px;
                    padding: 10px 16px; font-size: 0.82em; color: var(--pi-blue);
                    display: flex; align-items: center; gap: 8px; margin-top: 8px;
                }

                /* ── Empty state ── */
                .pi-empty {
                    background: var(--pi-bg); border: 1px solid var(--pi-border);
                    border-left: 4px solid var(--pi-blue); border-radius: 8px;
                    padding: 20px 24px; font-size: 0.88em; color: var(--pi-navy);
                    text-align: center;
                }

                /* ── Radio pill override ── */
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] {
                    display:inline-flex!important; background:#1BA0E2;
                    border-radius:50px; padding:3px; gap:0;
                }
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] > label {
                    cursor:pointer; padding:5px 18px!important; border-radius:50px!important;
                    font-size:0.78em!important; font-weight:500!important;
                    color:rgba(255,255,255,0.80)!important; margin:0!important;
                }
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] > label > div:first-child { display:none!important; }
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) { background:white!important; }
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] > label:has(input:checked) > div:last-child p { color:#1BA0E2!important; font-weight:700!important; }
                div[data-testid="stRadio"].st-key-pi_country_radio > label { display:none!important; }
                </style>

                <!-- Header -->
                <div class="pi-header">
                    <div class="pi-header-inner">
                        <div class="pi-header-icon">💹</div>
                        <div>
                            <div class="pi-header-title">Price Intelligence</div>
                            <div class="pi-header-sub">Pareto 80/20 Spend Concentration · Hotel Rate Benchmarking · Negotiation Simulator</div>
                        </div>
                        <div class="pi-header-badge">
                            <span class="pi-badge-dot"></span>LIVE ANALYSIS
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ══════════════════════════════════════════════════════
                #  GLOBAL FILTERS — berlaku untuk SEMUA section di tab6
                # ══════════════════════════════════════════════════════
                st.markdown("""
                <div class="pi-global-filter">
                    <div class="pi-global-filter-title">
                        🔍 Global Filters
                        <span style="color:var(--pi-muted);font-weight:400;letter-spacing:0;text-transform:none;font-size:1.1em;">
                            — berlaku untuk seluruh analisis di tab ini
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                gf_col1, gf_col2, gf_col3 = st.columns([1.2, 2, 2])

                # ── Filter 1: Wilayah ────────────────────────────────────────
                with gf_col1:
                    st.markdown("<div style='font-size:0.72em;font-weight:600;color:#6a8fa0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Wilayah</div>", unsafe_allow_html=True)
                    if "Country" in df_all.columns:
                        pi_country_filter = st.radio(
                            label="pi_wilayah",
                            options=["Indonesia", "Internasional"],
                            index=0, horizontal=True,
                            label_visibility="collapsed",
                            key="pi_country_radio"
                        )
                    else:
                        pi_country_filter = "Semua"
                        st.info("Kolom Country tidak tersedia")

                # ── Apply wilayah filter ke base dataframe ───────────────────
                df_pi_base = df_all.copy()

                if "Country" in df_pi_base.columns and pi_country_filter in ["Indonesia", "Internasional"]:
                    df_pi_base["_cu"] = df_pi_base["Country"].astype(str).str.strip().str.upper()
                    domestic_alias    = ["INDONESIA", "ID", "IDN"]
                    if pi_country_filter == "Indonesia":
                        df_pi_base    = df_pi_base[df_pi_base["_cu"].isin(domestic_alias)]
                        filter_label  = "🇮🇩 Indonesia"
                    else:
                        df_pi_base    = df_pi_base[~df_pi_base["_cu"].isin(domestic_alias)]
                        filter_label  = "🌐 Internasional"
                    df_pi_base.drop(columns=["_cu"], inplace=True)
                else:
                    filter_label = "🌏 Semua Wilayah"

                # ── Filter 2: Nama Hotel (multiselect) ───────────────────────
                with gf_col2:
                    st.markdown("<div style='font-size:0.72em;font-weight:600;color:#6a8fa0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Filter Hotel (opsional)</div>", unsafe_allow_html=True)
                    if "Hotel Name" in df_pi_base.columns:
                        hotel_options = sorted(df_pi_base["Hotel Name"].dropna().unique().tolist())
                        pi_hotel_filter = st.multiselect(
                            label="pi_hotel_ms",
                            options=hotel_options,
                            default=[],
                            placeholder="Semua hotel (pilih untuk filter spesifik)…",
                            label_visibility="collapsed",
                            key="pi_hotel_multiselect"
                        )
                    else:
                        pi_hotel_filter = []

                # ── Filter 3: Dimensi Pareto ─────────────────────────────────
                with gf_col3:
                    st.markdown("<div style='font-size:0.72em;font-weight:600;color:#6a8fa0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Dimensi Pareto</div>", unsafe_allow_html=True)
                    dimension_options = [c for c in ["Hotel Name", "City", "Supplier Name"] if c in df_pi_base.columns]
                    if dimension_options:
                        pi_dimension = st.selectbox(
                            label="pi_dim",
                            options=dimension_options,
                            label_visibility="collapsed",
                            key="pi_dimension_select"
                        )
                    else:
                        pi_dimension = None

                # ── Apply hotel filter ───────────────────────────────────────
                if pi_hotel_filter:
                    df_pi_base = df_pi_base[df_pi_base["Hotel Name"].isin(pi_hotel_filter)]

                # ── Filter summary badge ─────────────────────────────────────
                total_records  = len(df_all)
                filtered_records = len(df_pi_base)
                filter_pct     = (filtered_records / total_records * 100) if total_records > 0 else 0
                hotel_label    = f"{len(pi_hotel_filter)} hotel dipilih" if pi_hotel_filter else "Semua hotel"

                st.markdown(f"""
                <div style='background:#e6f4fb;border:1px solid #cce4f4;border-left:4px solid #1BA0E2;
                            border-radius:8px;padding:10px 18px;margin:4px 0 22px 0;
                            font-size:0.82em;color:#0D7FCC;display:flex;gap:20px;align-items:center;flex-wrap:wrap;'>
                    <span>🔎 <b>Filter aktif</b></span>
                    <span>Wilayah: <b>{filter_label}</b></span>
                    <span>Hotel: <b>{hotel_label}</b></span>
                    <span>Menampilkan <b>{filtered_records:,}</b> dari <b>{total_records:,}</b> records
                        <span style='background:#1BA0E2;color:white;border-radius:20px;
                                     padding:1px 10px;font-size:0.88em;margin-left:4px;'>{filter_pct:.1f}%</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

                if df_pi_base.empty:
                    st.markdown('<div class="pi-empty">⚠️ Tidak ada data yang sesuai dengan filter yang dipilih.</div>',
                                unsafe_allow_html=True)
                    return  # keluar dari tab ini saja, tidak menghentikan tab lain

                # ══════════════════════════════════════════════════════
                #  SECTION 1 — PARETO SPEND CONCENTRATION
                # ══════════════════════════════════════════════════════
                st.markdown('<div class="pi-section-divider">Spend Concentration — Pareto 80/20</div>',
                            unsafe_allow_html=True)

                if "Invoice Amount" not in df_pi_base.columns or pi_dimension is None:
                    st.markdown('<div class="pi-insight">⚠️ Kolom <strong>Invoice Amount</strong> atau dimensi tidak tersedia.</div>',
                                unsafe_allow_html=True)
                else:
                    df_sc = df_pi_base.dropna(subset=["Invoice Amount"]).copy()

                    if df_sc.empty:
                        st.markdown('<div class="pi-insight">⚠️ Tidak ada data setelah filter.</div>',
                                    unsafe_allow_html=True)
                    else:
                        pareto_df = (df_sc.groupby(pi_dimension)["Invoice Amount"]
                                     .sum().reset_index()
                                     .sort_values("Invoice Amount", ascending=False))
                        total_spend           = pareto_df["Invoice Amount"].sum()
                        pareto_df["Spend %"]  = pareto_df["Invoice Amount"] / total_spend * 100
                        pareto_df["Cumulative %"] = pareto_df["Spend %"].cumsum()
                        pareto_df["Rank"]     = range(1, len(pareto_df) + 1)

                        top_20_pct_count  = max(1, int(len(pareto_df) * 0.2))
                        top_contributors  = pareto_df.head(top_20_pct_count)
                        top_spend         = top_contributors["Invoice Amount"].sum()
                        top_spend_pct     = top_spend / total_spend * 100

                        # KPI Cards
                        st.markdown(f"""
                        <div class="pi-kpi-grid">
                            <div class="pi-kpi" style="border-top-color:#1BA0E2;">
                                <div class="pi-kpi-label">Total Spend · {filter_label}</div>
                                <div class="pi-kpi-value">Rp{total_spend:,.0f}</div>
                                <div class="pi-kpi-sub">keseluruhan periode</div>
                            </div>
                            <div class="pi-kpi" style="border-top-color:#1494C6;">
                                <div class="pi-kpi-label">Top 20% Count</div>
                                <div class="pi-kpi-value">{top_20_pct_count}</div>
                                <div class="pi-kpi-sub">{pi_dimension} teratas</div>
                            </div>
                            <div class="pi-kpi" style="border-top-color:#0D7FCC;">
                                <div class="pi-kpi-label">Top 20% Contribution</div>
                                <div class="pi-kpi-value" style="color:#0D7FCC;">{top_spend_pct:.1f}%</div>
                                <div class="pi-kpi-sub">dari total spend</div>
                            </div>
                            <div class="pi-kpi" style="border-top-color:#062440;">
                                <div class="pi-kpi-label">Bottom 80% Spend</div>
                                <div class="pi-kpi-value">Rp{(total_spend - top_spend):,.0f}</div>
                                <div class="pi-kpi-sub">sisa {len(pareto_df) - top_20_pct_count} entitas</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Pareto Chart
                        colors = ["#1BA0E2" if i < top_20_pct_count else "#d4e8f8"
                                  for i in range(len(pareto_df))]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=pareto_df[pi_dimension], y=pareto_df["Invoice Amount"],
                            name="Spend", marker=dict(color=colors),
                            hovertemplate="<b>%{x}</b><br>Rp%{y:,.0f}<extra></extra>"
                        ))
                        fig.add_trace(go.Scatter(
                            x=pareto_df[pi_dimension], y=pareto_df["Cumulative %"],
                            name="Cumulative %", yaxis="y2", mode="lines+markers",
                            line=dict(color="#062440", width=2.5),
                            marker=dict(size=5, color="#062440")
                        ))
                        fig.add_hline(y=80, yref="y2", line_dash="dash",
                                      line_color="#1BA0E2", opacity=0.5,
                                      annotation_text="80%", annotation_position="right",
                                      annotation_font_color="#1BA0E2")
                        fig.update_layout(
                            template="plotly_white",
                            yaxis=dict(title="Spend (Rp)", gridcolor="#e8f4fd"),
                            yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                                        range=[0, 100], showgrid=False, ticksuffix="%"),
                            height=460, plot_bgcolor="white", paper_bgcolor="white",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="right", x=1,
                                        bgcolor="rgba(240,248,255,0.9)",
                                        bordercolor="#cce4f4", borderwidth=1),
                            margin=dict(l=60, r=80, t=40, b=100),
                            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Sub-tabs
                        tab_sim, tab_detail, tab_insight = st.tabs([
                            "💰  Saving Simulation", "📋  Detail Data", "💡  Insight"
                        ])

                        with tab_sim:
                            renegotiation_rate = st.slider(
                                "Target diskon pada Top 20% contributors (%)", 0, 25, 5, 1,
                                key="pi_pareto_disc"
                            )
                            potential_saving = top_spend * (renegotiation_rate / 100)
                            st.markdown(f"""
                            <div class="pi-sim-grid">
                                <div class="pi-sim-card pi-sim-current">
                                    <div class="pi-sim-card-label">Top 20% Spend</div>
                                    <div class="pi-sim-card-value">Rp{top_spend:,.0f}</div>
                                    <div class="pi-sim-card-sub">{top_20_pct_count} {pi_dimension} · {top_spend_pct:.1f}% of all</div>
                                </div>
                                <div class="pi-sim-card pi-sim-neg">
                                    <div class="pi-sim-card-label">Setelah Diskon {renegotiation_rate}%</div>
                                    <div class="pi-sim-card-value">Rp{top_spend*(1-renegotiation_rate/100):,.0f}</div>
                                    <div class="pi-sim-card-sub">spend setelah renegosiasi</div>
                                </div>
                                <div class="pi-sim-card pi-sim-saving">
                                    <span class="pi-sim-saving-icon">💰</span>
                                    <div class="pi-sim-card-label">Potensi Saving</div>
                                    <div class="pi-sim-card-value">Rp{potential_saving:,.0f}</div>
                                    <div class="pi-sim-card-sub">dengan target diskon {renegotiation_rate}%</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with tab_detail:
                            display_cols = [pi_dimension, "Invoice Amount", "Spend %", "Cumulative %", "Rank"]
                            st.dataframe(
                                top_contributors[display_cols].style
                                .format({"Invoice Amount": "Rp{:,.0f}",
                                         "Spend %": "{:.2f}%",
                                         "Cumulative %": "{:.2f}%"})
                                .background_gradient(subset=["Spend %"], cmap="Blues"),
                                use_container_width=True
                            )
                            output_excel = BytesIO()
                            top_contributors.to_excel(output_excel, index=False, sheet_name="Top Contributors")
                            output_excel.seek(0)
                            if st.session_state.get("role") == "Admin":
                                st.download_button(
                                    label="⬇️ Download Excel", data=output_excel,
                                    file_name=f"pareto_{pi_dimension.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="dl_pareto_contributors")
                            else:
                                st.markdown('<div class="pi-locked">🔒 Download hanya tersedia untuk <strong>Admin</strong></div>',
                                            unsafe_allow_html=True)

                        with tab_insight:
                            st.markdown(f"""
                            <div class="pi-insight">
                                <strong>Konsentrasi Spending · {filter_label} · {hotel_label}</strong><br>
                                Top 20% (<em>{top_20_pct_count} {pi_dimension}</em>) menyumbang
                                <em>{top_spend_pct:.1f}%</em> dari total pengeluaran
                                <strong>Rp{total_spend:,.0f}</strong>.<br><br>
                                Fokus renegosiasi pada kelompok ini memberikan dampak finansial terbesar.
                                Dengan target diskon 10%, estimasi saving yang dapat dicapai adalah
                                <em>Rp{top_spend * 0.10:,.0f}</em>.
                            </div>
                            """, unsafe_allow_html=True)

                # ══════════════════════════════════════════════════════
                #  SECTION 2 — HOTEL PRICE INTELLIGENCE
                # ══════════════════════════════════════════════════════
                st.markdown('<div class="pi-section-divider">Hotel Price Intelligence</div>',
                            unsafe_allow_html=True)

                # Build hotel display name dari df_pi_base (sudah terfilter wilayah + hotel)
                df_pi_hotel = df_pi_base.copy()

                if "Canonical Hotel Name" in df_pi_hotel.columns:
                    df_pi_hotel["Hotel Display"] = (df_pi_hotel["Canonical Hotel Name"].astype(str)
                                                    .str.strip().str.replace(r"\s+", " ", regex=True).str.title())
                elif "Hotel Name" in df_pi_hotel.columns:
                    df_pi_hotel["Hotel Display"] = (df_pi_hotel["Hotel Name"].astype(str)
                                                    .str.strip().str.replace(r"\s+", " ", regex=True).str.title())
                else:
                    st.markdown('<div class="pi-insight">⚠️ Kolom Hotel Name tidak tersedia.</div>',
                                unsafe_allow_html=True)
                    return  # keluar dari tab ini saja, tidak menghentikan tab lain

                hotel_list_intel = sorted(df_pi_hotel["Hotel Display"].dropna().unique())

                if not hotel_list_intel:
                    st.markdown(f"""
                    <div class="pi-insight">
                        ⚠️ Tidak ada data hotel untuk filter:
                        <strong>{filter_label}</strong>
                        {f"· <strong>{hotel_label}</strong>" if pi_hotel_filter else ""}.
                        Coba ubah filter di atas.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Tampilkan info berapa hotel tersedia
                    st.markdown(f"""
                    <div style='background:#f0f8ff;border:1px solid #cce4f4;border-left:3px solid #1BA0E2;
                                border-radius:6px;padding:10px 16px;margin-bottom:14px;
                                font-size:0.82em;color:#0D7FCC;'>
                        🏨 <b>{len(hotel_list_intel):,} hotel</b> tersedia untuk
                        <b>{filter_label}</b>
                        {f"· filter: <b>{hotel_label}</b>" if pi_hotel_filter else ""}
                    </div>
                    """, unsafe_allow_html=True)

                    selected_hotel = st.selectbox(
                        "Pilih hotel untuk analisis detail",
                        hotel_list_intel,
                        label_visibility="collapsed",
                        key="pi_hotel_detail_select"
                    )

                    df_hotel = df_pi_hotel[df_pi_hotel["Hotel Display"] == selected_hotel].copy()

                    if not df_hotel.empty and "Invoice Amount" in df_hotel.columns and "Number of Rooms Night" in df_hotel.columns:
                        df_valid = df_hotel[
                            df_hotel["Invoice Amount"].notna() &
                            (df_hotel["Number of Rooms Night"] > 0)
                        ].copy()
                        df_valid["Price Per Night"] = df_valid["Invoice Amount"] / df_valid["Number of Rooms Night"]

                        if df_valid.empty:
                            st.markdown('<div class="pi-insight">⚠️ Tidak ada data harga yang valid untuk hotel ini.</div>',
                                        unsafe_allow_html=True)
                        else:
                            avg_rate          = df_valid["Price Per Night"].mean()
                            median_rate       = df_valid["Price Per Night"].median()
                            max_rate          = df_valid["Price Per Night"].max()
                            min_rate          = df_valid["Price Per Night"].min()
                            total_room_nights = df_valid["Number of Rooms Night"].sum()

                            # Hotel KPI Cards
                            st.markdown(f"""
                            <div class="pi-hotel-kpi-grid">
                                <div class="pi-hotel-kpi" style="--pi-color:#1BA0E2;">
                                    <span class="pi-hotel-kpi-icon">📊</span>
                                    <div class="pi-hotel-kpi-label">Avg Rate</div>
                                    <div class="pi-hotel-kpi-value">Rp {avg_rate:,.0f}</div>
                                </div>
                                <div class="pi-hotel-kpi" style="--pi-color:#1494C6;">
                                    <span class="pi-hotel-kpi-icon">📍</span>
                                    <div class="pi-hotel-kpi-label">Median Rate</div>
                                    <div class="pi-hotel-kpi-value">Rp {median_rate:,.0f}</div>
                                </div>
                                <div class="pi-hotel-kpi" style="--pi-color:#d9534f;">
                                    <span class="pi-hotel-kpi-icon">🔺</span>
                                    <div class="pi-hotel-kpi-label">Max Rate</div>
                                    <div class="pi-hotel-kpi-value">Rp {max_rate:,.0f}</div>
                                </div>
                                <div class="pi-hotel-kpi" style="--pi-color:#2e8a57;">
                                    <span class="pi-hotel-kpi-icon">🔻</span>
                                    <div class="pi-hotel-kpi-label">Min Rate</div>
                                    <div class="pi-hotel-kpi-value">Rp {min_rate:,.0f}</div>
                                </div>
                                <div class="pi-hotel-kpi" style="--pi-color:#0D7FCC;">
                                    <span class="pi-hotel-kpi-icon">🌙</span>
                                    <div class="pi-hotel-kpi-label">Total Room Nights</div>
                                    <div class="pi-hotel-kpi-value">{total_room_nights:,.0f}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Negotiation Simulator
                            st.markdown('<div class="pi-section-divider">Negotiation Simulator</div>',
                                        unsafe_allow_html=True)

                            target_discount  = st.slider("🎯 Target Discount (%)", 0, 30, 10, key="pi_hotel_disc")
                            negotiated_rate  = avg_rate * (1 - target_discount / 100)
                            estimated_saving = (avg_rate - negotiated_rate) * total_room_nights

                            st.markdown(f"""
                            <div class="pi-sim-grid">
                                <div class="pi-sim-card pi-sim-current">
                                    <div class="pi-sim-card-label">Current Avg Rate / Malam</div>
                                    <div class="pi-sim-card-value">Rp {avg_rate:,.0f}</div>
                                    <div class="pi-sim-card-sub">basis {total_room_nights:,.0f} room nights</div>
                                </div>
                                <div class="pi-sim-card pi-sim-neg">
                                    <div class="pi-sim-card-label">Negotiated Rate (–{target_discount}%)</div>
                                    <div class="pi-sim-card-value">Rp {negotiated_rate:,.0f}</div>
                                    <div class="pi-sim-card-sub">target rate kontrak</div>
                                </div>
                                <div class="pi-sim-card pi-sim-saving">
                                    <span class="pi-sim-saving-icon">💰</span>
                                    <div class="pi-sim-card-label">Estimated Saving</div>
                                    <div class="pi-sim-card-value">Rp {estimated_saving:,.0f}</div>
                                    <div class="pi-sim-card-sub">total potensi hemat</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.expander("📘 Cara Perhitungan Estimated Saving", expanded=False):
                                st.markdown(f"""
                                <div class="pi-insight">
                                    <strong>Formula:</strong><br>
                                    Saving = (Avg Rate − Negotiated Rate) × Total Room Nights<br><br>
                                    = (Rp {avg_rate:,.0f} − Rp {negotiated_rate:,.0f})
                                    × <em>{total_room_nights:,.0f} malam</em><br>
                                    = <strong>Rp {estimated_saving:,.0f}</strong>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="pi-insight">⚠️ Data tidak cukup untuk hotel yang dipilih.</div>',
                                    unsafe_allow_html=True)

            # ======================================
            # TAB 7: SANKEY FLOW — MTRAX Blue Theme
            # ======================================
            _render_tab6()
        with tab7:

            st.markdown("""
            <style>
            :root {
                --sk-blue:      #1BA0E2;
                --sk-blue-mid:  #1494C6;
                --sk-blue-dark: #0D7FCC;
                --sk-navy:      #062440;
                --sk-lime:      #e2f871;
                --sk-lime-v:    #98ea16;
                --sk-bg:        #f0f8ff;
                --sk-border:    #cce4f4;
                --sk-surface:   #ffffff;
                --sk-muted:     #6a8fa0;
                --sk-text:      #1a2a3a;
            }

            /* ════════════ HEADER ════════════ */
            .sk-header {
                background: linear-gradient(135deg, #062440 0%, #0D7FCC 52%, #1BA0E2 100%);
                border-radius: 14px;
                padding: 28px 36px;
                margin-bottom: 24px;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(13,127,204,0.28);
            }
            .sk-header::before {
                content: '';
                position: absolute; top: -70px; right: -50px;
                width: 240px; height: 240px; border-radius: 50%;
                background: rgba(255,255,255,0.05); pointer-events: none;
            }
            .sk-header::after {
                content: '';
                position: absolute; bottom: -50px; left: 30%;
                width: 200px; height: 200px; border-radius: 50%;
                background: rgba(226,248,113,0.08); pointer-events: none;
            }
            .sk-header-inner {
                display: flex; align-items: center; gap: 18px;
                position: relative; z-index: 1;
            }
            .sk-header-icon {
                width: 54px; height: 54px; border-radius: 13px;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.24);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.7em; flex-shrink: 0;
            }
            .sk-header-body  { flex: 1; }
            .sk-header-title {
                color: #fff; font-size: 1.45em; font-weight: 700;
                margin: 0; letter-spacing: -0.01em;
            }
            .sk-header-sub {
                color: rgba(255,255,255,0.58); font-size: 0.82em; margin-top: 4px;
            }
            .sk-header-badge {
                display: inline-flex; align-items: center; gap: 7px;
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(255,255,255,0.22);
                border-radius: 100px; padding: 6px 16px;
                font-size: 0.72em; font-weight: 600;
                color: rgba(255,255,255,0.88); letter-spacing: 0.06em;
                white-space: nowrap;
            }
            .sk-badge-dot {
                width: 6px; height: 6px; border-radius: 50%;
                background: var(--sk-lime);
                box-shadow: 0 0 6px var(--sk-lime);
                animation: sk-blink 2s ease infinite;
            }
            @keyframes sk-blink { 0%,100%{opacity:1} 50%{opacity:.3} }

            /* ════════════ FLOW PATH INDICATOR ════════════ */
            .sk-flow-path {
                display: flex; align-items: center; justify-content: center;
                gap: 0; margin-bottom: 20px;
            }
            .sk-flow-node {
                background: var(--sk-surface);
                border: 1.5px solid var(--sk-border);
                border-radius: 8px; padding: 10px 20px;
                font-size: 0.82em; font-weight: 700;
                color: var(--sk-navy); letter-spacing: 0.02em;
                display: flex; align-items: center; gap: 8px;
                box-shadow: 0 2px 8px rgba(13,127,204,0.08);
            }
            .sk-flow-node-icon { font-size: 1.1em; }
            .sk-flow-arrow {
                display: flex; align-items: center;
                padding: 0 4px;
            }
            .sk-flow-arrow-line {
                width: 36px; height: 2px;
                background: linear-gradient(90deg, var(--sk-border), var(--sk-blue));
            }
            .sk-flow-arrow-head {
                width: 0; height: 0;
                border-top: 5px solid transparent;
                border-bottom: 5px solid transparent;
                border-left: 8px solid var(--sk-blue);
            }
            .sk-flow-node.active {
                background: linear-gradient(135deg, #f0f8ff, #e6f4fb);
                border-color: var(--sk-blue);
                color: var(--sk-blue-dark);
            }

            /* ════════════ FILTER PANEL ════════════ */
            .sk-filter-panel {
                background: var(--sk-surface);
                border: 1px solid var(--sk-border);
                border-left: 4px solid var(--sk-blue);
                border-radius: 10px;
                padding: 16px 20px 12px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(13,127,204,0.07);
            }
            .sk-filter-label {
                font-size: 0.69em; font-weight: 700;
                color: var(--sk-blue); text-transform: uppercase;
                letter-spacing: 0.12em; margin-bottom: 10px;
                display: flex; align-items: center; gap: 8px;
            }
            .sk-filter-tag {
                background: var(--sk-bg); border: 1px solid var(--sk-border);
                color: var(--sk-blue); border-radius: 20px;
                padding: 2px 10px; font-size: 0.85em; font-weight: 600;
            }

            /* ════════════ SECTION DIVIDER ════════════ */
            .sk-divider {
                display: flex; align-items: center; gap: 14px;
                margin: 22px 0 16px;
                font-size: 0.69em; font-weight: 700;
                color: var(--sk-muted); letter-spacing: 0.14em;
                text-transform: uppercase;
            }
            .sk-divider::before,
            .sk-divider::after {
                content: ''; flex: 1; height: 1px; background: var(--sk-border);
            }

            /* ════════════ STAT CARDS ════════════ */
            .sk-stat-grid {
                display: grid; grid-template-columns: repeat(4,1fr);
                gap: 12px; margin-bottom: 18px;
            }
            .sk-stat {
                background: var(--sk-surface);
                border-radius: 10px; padding: 16px 14px 12px;
                border-top: 3px solid var(--sk-color, #1BA0E2);
                box-shadow: 0 2px 8px rgba(13,127,204,0.07);
                text-align: center;
                transition: transform .18s ease, box-shadow .18s ease;
            }
            .sk-stat:hover { transform: translateY(-2px); box-shadow: 0 5px 16px rgba(13,127,204,0.13); }
            .sk-stat-icon  { font-size: 1.35em; margin-bottom: 6px; display: block; }
            .sk-stat-label {
                font-size: 0.66em; font-weight: 600; color: var(--sk-muted);
                text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 5px;
            }
            .sk-stat-value {
                font-size: 1.15em; font-weight: 700;
                color: var(--sk-color, #1BA0E2); line-height: 1.2;
            }

            /* ════════════ ACTIVE FILTER BADGE ════════════ */
            .sk-active-filter {
                background: #e6f4fb; border: 1px solid var(--sk-border);
                border-left: 4px solid var(--sk-blue);
                border-radius: 8px; padding: 10px 18px;
                margin-bottom: 16px;
                font-size: 0.82em; color: var(--sk-blue-dark);
                display: flex; gap: 20px; align-items: center; flex-wrap: wrap;
            }

            /* ════════════ DOWNLOAD / LOCKED ════════════ */
            .sk-locked {
                background: var(--sk-bg); border: 1px solid var(--sk-border);
                border-left: 3px solid var(--sk-blue); border-radius: 6px;
                padding: 10px 16px; font-size: 0.82em; color: var(--sk-blue);
                display: flex; align-items: center; gap: 8px; margin-top: 8px;
            }

            /* ════════════ EMPTY / WARN ════════════ */
            .sk-warn {
                background: var(--sk-bg); border: 1px solid var(--sk-border);
                border-left: 4px solid var(--sk-blue); border-radius: 8px;
                padding: 16px 20px; font-size: 0.88em; color: var(--sk-navy);
            }

            /* ════════════ RADIO PILL — WILAYAH ════════════ */
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] {
                display:inline-flex!important; background:#1BA0E2;
                border-radius:50px; padding:3px; gap:0;
            }
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] > label {
                cursor:pointer; padding:5px 20px!important; border-radius:50px!important;
                font-size:0.78em!important; font-weight:500!important;
                color:rgba(255,255,255,0.80)!important; margin:0!important;
            }
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] > label > div:first-child { display:none!important; }
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) { background:white!important; }
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] > label:has(input:checked) > div:last-child p { color:#1BA0E2!important; font-weight:700!important; }
            div[data-testid="stRadio"].st-key-sankey_country_radio > label { display:none!important; }
            </style>

            <!-- ── Header ── -->
            <div class="sk-header">
                <div class="sk-header-inner">
                    <div class="sk-header-icon">🌊</div>
                    <div class="sk-header-body">
                        <div class="sk-header-title">Sankey Flow Analysis</div>
                        <div class="sk-header-sub">
                            Visualisasi alur pengeluaran travel ·
                            Perusahaan → Kota → Hotel · berbasis Invoice Amount
                        </div>
                    </div>
                    <div class="sk-header-badge">
                        <span class="sk-badge-dot"></span>FLOW MAP
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Flow Path Indicator ─────────────────────────────────────────
            st.markdown("""
            <div class="sk-flow-path">
                <div class="sk-flow-node active">
                    <span class="sk-flow-node-icon">🏢</span> Perusahaan
                </div>
                <div class="sk-flow-arrow">
                    <div class="sk-flow-arrow-line"></div>
                    <div class="sk-flow-arrow-head"></div>
                </div>
                <div class="sk-flow-node active">
                    <span class="sk-flow-node-icon">🏙️</span> Kota
                </div>
                <div class="sk-flow-arrow">
                    <div class="sk-flow-arrow-line"></div>
                    <div class="sk-flow-arrow-head"></div>
                </div>
                <div class="sk-flow-node active">
                    <span class="sk-flow-node-icon">🏨</span> Hotel
                </div>
            </div>
            """, unsafe_allow_html=True)

            required_sankey  = ["Nama Perusahaan", "Hotel Name", "Invoice Amount"]
            city_sankey_col  = next(
                (c for c in ["City", "City Destination"] if c in df_overview.columns), None
            )

            if not all(c in df_overview.columns for c in required_sankey) or city_sankey_col is None:
                st.markdown(
                    '<div class="sk-warn">⚠️ Kolom yang dibutuhkan tidak lengkap untuk Sankey Flow.</div>',
                    unsafe_allow_html=True
                )
            else:
                # ── Filter Panel ─────────────────────────────────────────────
                st.markdown("""
                <div class="sk-filter-panel">
                    <div class="sk-filter-label">
                        ⚙ Filter &amp; Parameter
                        <span class="sk-filter-tag">Semua perubahan langsung memperbarui diagram</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                row_ctrl = st.columns([1.4, 1, 1, 1, 1])

                with row_ctrl[0]:
                    st.markdown(
                        "<div style='font-size:0.70em;font-weight:700;color:#6a8fa0;"
                        "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>"
                        "Wilayah</div>",
                        unsafe_allow_html=True
                    )
                    sankey_country_filter = st.radio(
                        label="sankey_wilayah",
                        options=["Domestik", "Internasional"],
                        index=0, horizontal=True,
                        label_visibility="collapsed",
                        key="sankey_country_radio"
                    )

                with row_ctrl[1]:
                    top_n_company = st.selectbox(
                        "🏢 Perusahaan", [5, 8, 10, 15, 20],
                        index=2, key="sankey_top_company"
                    )
                with row_ctrl[2]:
                    top_n_city = st.selectbox(
                        "🏙️ Kota", [5, 8, 10, 15, 20, 30],
                        index=2, key="sankey_top_city"
                    )
                with row_ctrl[3]:
                    top_n_hotel = st.selectbox(
                        "🏨 Hotel", [10, 15, 20, 30, 50],
                        index=2, key="sankey_top_hotel"
                    )
                with row_ctrl[4]:
                    chart_height = st.selectbox(
                        "📐 Tinggi Chart", [600, 750, 900, 1100],
                        index=1, key="sankey_height"
                    )

                # ── Apply wilayah filter ──────────────────────────────────────
                if "Country" in df_overview.columns:
                    _df_sk      = df_overview.copy()
                    _df_sk["_cu"] = _df_sk["Country"].astype(str).str.strip().str.upper()
                    domestic_alias = ["INDONESIA", "ID", "IDN"]

                    if sankey_country_filter == "Domestik":
                        df_sankey    = _df_sk[_df_sk["_cu"].isin(domestic_alias)].drop(columns=["_cu"])
                        sankey_label = "🇮🇩 Domestik"
                    else:
                        df_sankey    = _df_sk[~_df_sk["_cu"].isin(domestic_alias)].drop(columns=["_cu"])
                        sankey_label = "🌐 Internasional"
                else:
                    df_sankey    = df_overview.copy()
                    sankey_label = "🌏 Semua"

                if df_sankey.empty:
                    st.markdown(
                        f'<div class="sk-warn">⚠️ Tidak ada data untuk filter: <strong>{sankey_label}</strong></div>',
                        unsafe_allow_html=True
                    )
                else:
                    df_sk = df_sankey.dropna(
                        subset=["Nama Perusahaan", city_sankey_col, "Hotel Name", "Invoice Amount"]
                    ).copy()

                    # Compute tops
                    top_companies = (df_sk.groupby("Nama Perusahaan")["Invoice Amount"]
                                     .sum().nlargest(top_n_company).index)
                    top_cities    = (df_sk.groupby(city_sankey_col)["Invoice Amount"]
                                     .sum().nlargest(top_n_city).index)
                    top_hotels    = (df_sk.groupby("Hotel Name")["Invoice Amount"]
                                     .sum().nlargest(top_n_hotel).index)

                    df_sk = df_sk[
                        df_sk["Nama Perusahaan"].isin(top_companies) &
                        df_sk[city_sankey_col].isin(top_cities) &
                        df_sk["Hotel Name"].isin(top_hotels)
                    ]

                    if df_sk.empty:
                        st.markdown(
                            '<div class="sk-warn">⚠️ Tidak ada data setelah filter diterapkan.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        # ── Active filter badge ───────────────────────────────
                        total_flow_spend = df_sk["Invoice Amount"].sum()
                        total_raw_spend  = df_overview["Invoice Amount"].sum() if "Invoice Amount" in df_overview.columns else 0
                        coverage_pct     = (total_flow_spend / total_raw_spend * 100) if total_raw_spend > 0 else 0

                        st.markdown(f"""
                        <div class="sk-active-filter">
                            <span>🔎 <b>Filter aktif</b></span>
                            <span>Wilayah: <b>{sankey_label}</b></span>
                            <span>Top <b>{top_n_company}</b> Perusahaan
                                · <b>{top_n_city}</b> Kota
                                · <b>{top_n_hotel}</b> Hotel</span>
                            <span>Total Spend:
                                <b>Rp{total_flow_spend:,.0f}</b>
                                <span style='background:#1BA0E2;color:white;border-radius:20px;
                                             padding:1px 10px;font-size:0.88em;margin-left:4px;'>
                                    {coverage_pct:.1f}% dari total
                                </span>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                        # ── Stat Cards ────────────────────────────────────────
                        n_co = df_sk["Nama Perusahaan"].nunique()
                        n_ci = df_sk[city_sankey_col].nunique()
                        n_ht = df_sk["Hotel Name"].nunique()
                        n_rn = df_sk["Number of Rooms Night"].sum() if "Number of Rooms Night" in df_sk.columns else 0

                        st.markdown(f"""
                        <div class="sk-stat-grid">
                            <div class="sk-stat" style="--sk-color:#1BA0E2;">
                                <span class="sk-stat-icon">🏢</span>
                                <div class="sk-stat-label">Perusahaan</div>
                                <div class="sk-stat-value">{n_co}</div>
                            </div>
                            <div class="sk-stat" style="--sk-color:#1494C6;">
                                <span class="sk-stat-icon">🏙️</span>
                                <div class="sk-stat-label">Kota Tujuan</div>
                                <div class="sk-stat-value">{n_ci}</div>
                            </div>
                            <div class="sk-stat" style="--sk-color:#0D7FCC;">
                                <span class="sk-stat-icon">🏨</span>
                                <div class="sk-stat-label">Hotel</div>
                                <div class="sk-stat-value">{n_ht}</div>
                            </div>
                            <div class="sk-stat" style="--sk-color:#062440;">
                                <span class="sk-stat-icon">💰</span>
                                <div class="sk-stat-label">Total Spend</div>
                                <div class="sk-stat-value">Rp{total_flow_spend/1e9:.1f}B</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # ── Build Sankey data ─────────────────────────────────
                        co_ci = (df_sk.groupby(["Nama Perusahaan", city_sankey_col])
                                 ["Invoice Amount"].sum().reset_index())
                        ci_ho = (df_sk.groupby([city_sankey_col, "Hotel Name"])
                                 ["Invoice Amount"].sum().reset_index())

                        companies = list(co_ci["Nama Perusahaan"].unique())
                        cities    = list(co_ci[city_sankey_col].unique())
                        hotels    = list(ci_ho["Hotel Name"].unique())
                        node_labels = companies + cities + hotels

                        co_idx = {c: i                              for i, c in enumerate(companies)}
                        ci_idx = {c: len(companies) + i              for i, c in enumerate(cities)}
                        ho_idx = {h: len(companies)+len(cities)+i    for i, h in enumerate(hotels)}

                        sources, targets, values = [], [], []
                        for _, row in co_ci.iterrows():
                            if row["Nama Perusahaan"] in co_idx and row[city_sankey_col] in ci_idx:
                                sources.append(co_idx[row["Nama Perusahaan"]])
                                targets.append(ci_idx[row[city_sankey_col]])
                                values.append(row["Invoice Amount"])
                        for _, row in ci_ho.iterrows():
                            if row[city_sankey_col] in ci_idx and row["Hotel Name"] in ho_idx:
                                sources.append(ci_idx[row[city_sankey_col]])
                                targets.append(ho_idx[row["Hotel Name"]])
                                values.append(row["Invoice Amount"])

                        # MTRAX blue-family node colors
                        # Companies → blue shades, Cities → teal/mid-blue, Hotels → lime+navy
                        CO_COLORS = [
                            "#1BA0E2", "#1494C6", "#0D7FCC", "#47b5e8",
                            "#75caf0", "#a3dcf6", "#cce4f4", "#e6f4fb",
                            "#5a8fc0", "#2e6fa0", "#062440", "#0a3d62",
                            "#1565C0", "#1976D2", "#1E88E5", "#42A5F5"
                        ]
                        CI_COLORS = [
                            "#0D7FCC", "#1494C6", "#1BA0E2", "#47b5e8",
                            "#75caf0", "#a3dcf6", "#5a8fc0", "#2e6fa0",
                            "#0a4d7a", "#1565C0", "#1976D2", "#1E88E5",
                            "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB"
                        ]
                        HO_COLORS = [
                            "#98ea16", "#e2f871", "#b8e040", "#cce84a",
                            "#a8d030", "#062440", "#0D7FCC", "#1BA0E2",
                            "#47b5e8", "#5a8fc0", "#2e6fa0", "#0a3d62",
                            "#1494C6", "#75caf0", "#a3dcf6", "#cce4f4"
                        ]

                        co_colors = [CO_COLORS[i % len(CO_COLORS)] for i in range(len(companies))]
                        ci_colors = [CI_COLORS[i % len(CI_COLORS)] for i in range(len(cities))]
                        ho_colors = [HO_COLORS[i % len(HO_COLORS)] for i in range(len(hotels))]
                        node_colors = co_colors + ci_colors + ho_colors

                        link_colors = ["rgba(27,160,226,0.28)"] * len(sources)

                        # ── Sankey figure ─────────────────────────────────────
                        st.markdown('<div class="sk-divider">Diagram Sankey</div>', unsafe_allow_html=True)

                        fig_sankey = go.Figure(go.Sankey(
                            arrangement="snap",
                            textfont=dict(
                                family="'Segoe UI', Arial, sans-serif",
                                size=11, color="#1a2a3a"
                            ),
                            node=dict(
                                pad=24, thickness=18,
                                line=dict(color="rgba(255,255,255,0.6)", width=0.6),
                                label=node_labels,
                                color=node_colors
                            ),
                            link=dict(
                                source=sources,
                                target=targets,
                                value=values,
                                color=link_colors
                            )
                        ))

                        fig_sankey.update_layout(
                            paper_bgcolor="#f0f8ff",
                            plot_bgcolor="#f0f8ff",
                            height=chart_height,
                            margin=dict(l=16, r=16, t=58, b=16),
                            title=dict(
                                text=(
                                    f"<b>Sankey Flow: Perusahaan → Kota → Hotel</b>"
                                    f"<span style='color:#6a8fa0;font-size:11px;'>"
                                    f"  ·  Top {top_n_company} Perusahaan"
                                    f"  ·  Top {top_n_city} Kota"
                                    f"  ·  Top {top_n_hotel} Hotel"
                                    f"  ·  {sankey_label}"
                                    f"</span>"
                                ),
                                x=0.01, xanchor="left",
                                font=dict(size=13, color="#1a2a3a")
                            )
                        )

                        st.plotly_chart(fig_sankey, use_container_width=True)

                        # ── Top Tables ────────────────────────────────────────
                        st.markdown('<div class="sk-divider">Tabel Ringkasan</div>', unsafe_allow_html=True)

                        tbl_col1, tbl_col2, tbl_col3 = st.columns(3)

                        top_co_tbl = (df_sk.groupby("Nama Perusahaan")["Invoice Amount"]
                                      .sum().sort_values(ascending=False)
                                      .reset_index().head(top_n_company))
                        top_ci_tbl = (df_sk.groupby(city_sankey_col)["Invoice Amount"]
                                      .sum().sort_values(ascending=False)
                                      .reset_index().head(top_n_city))
                        top_ht_tbl = (df_sk.groupby("Hotel Name")["Invoice Amount"]
                                      .sum().sort_values(ascending=False)
                                      .reset_index().head(top_n_hotel))

                        with tbl_col1:
                            st.markdown("""
                            <div style='background:#f0f8ff;border:1px solid #cce4f4;
                                        border-left:3px solid #1BA0E2;border-radius:8px;
                                        padding:10px 16px 4px;margin-bottom:8px;'>
                                <div style='font-size:0.72em;font-weight:700;color:#1BA0E2;
                                            text-transform:uppercase;letter-spacing:0.08em;'>
                                    🏢 Top Perusahaan
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(
                                top_co_tbl.style
                                .format({"Invoice Amount": "Rp{:,.0f}"})
                                .background_gradient(subset=["Invoice Amount"], cmap="Blues"),
                                use_container_width=True, hide_index=True
                            )

                        with tbl_col2:
                            st.markdown("""
                            <div style='background:#f0f8ff;border:1px solid #cce4f4;
                                        border-left:3px solid #1494C6;border-radius:8px;
                                        padding:10px 16px 4px;margin-bottom:8px;'>
                                <div style='font-size:0.72em;font-weight:700;color:#1494C6;
                                            text-transform:uppercase;letter-spacing:0.08em;'>
                                    🏙️ Top Kota
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(
                                top_ci_tbl.style
                                .format({"Invoice Amount": "Rp{:,.0f}"})
                                .background_gradient(subset=["Invoice Amount"], cmap="Blues"),
                                use_container_width=True, hide_index=True
                            )

                        with tbl_col3:
                            st.markdown("""
                            <div style='background:#f0f8ff;border:1px solid #cce4f4;
                                        border-left:3px solid #0D7FCC;border-radius:8px;
                                        padding:10px 16px 4px;margin-bottom:8px;'>
                                <div style='font-size:0.72em;font-weight:700;color:#0D7FCC;
                                            text-transform:uppercase;letter-spacing:0.08em;'>
                                    🏨 Top Hotel
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(
                                top_ht_tbl.style
                                .format({"Invoice Amount": "Rp{:,.0f}"})
                                .background_gradient(subset=["Invoice Amount"], cmap="PuBu"),
                                use_container_width=True, hide_index=True
                            )

                        # ── Download ──────────────────────────────────────────
                        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                        output_sankey = BytesIO()
                        export_df = (
                            df_sk.groupby(["Nama Perusahaan", city_sankey_col, "Hotel Name"])
                            ["Invoice Amount"].sum()
                            .reset_index()
                            .sort_values("Invoice Amount", ascending=False)
                        )
                        export_df.to_excel(output_sankey, index=False, sheet_name="Sankey Flow")
                        output_sankey.seek(0)

                        if st.session_state.get("role") == "Admin":
                            st.download_button(
                                label="⬇️ Download Data Sankey Flow (Excel)",
                                data=output_sankey,
                                file_name=(
                                    f"sankey_flow_{sankey_label.replace('🇮🇩','').replace('🌐','').strip()}_"
                                    f"{datetime.now().strftime('%Y%m%d')}.xlsx"
                                ),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_sankey_flow")
                        else:
                            st.markdown(
                                '<div class="sk-locked">🔒 Download hanya tersedia untuk <strong>Admin</strong></div>',
                                unsafe_allow_html=True
                            )

        # ======================================
        # TAB 8: DATA HOTEL
        # ======================================
        with tab8:
            def _render_tab8():
                st.markdown("<div class='section-title'>Data Hotel</div>", unsafe_allow_html=True)

                # =========================
                # FILTER DOMESTIK / INTERNATIONAL
                # =========================
                if "Country" in df_overview.columns:

                    st.markdown("""
                    <style>
                    div[data-testid="stRadio"].st-key-tab8_country_radio > div[role="radiogroup"]{
                        display:inline-flex!important;
                        background:#1BA0E2;
                        border-radius:50px;
                        padding:3px;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio > div[role="radiogroup"] > label{
                        cursor:pointer;
                        padding:4px 16px!important;
                        border-radius:50px!important;
                        font-size:0.78em!important;
                        font-weight:500!important;
                        color:rgba(255,255,255,0.80)!important;
                        margin:0!important;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio > div[role="radiogroup"] > label > div:first-child{
                        display:none!important;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio 
                    > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked){
                        background:white!important;
                        color:#1BA0E2!important;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio 
                    > div[role="radiogroup"] > label:has(input:checked) > div:last-child p{
                        color:#1BA0E2!important;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio > label{
                        display:none!important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    tab8_country_filter = st.radio(
                        label="filter_tab8",
                        options=["Domestik", "Internasional"],
                        index=0,
                        horizontal=True,
                        label_visibility="collapsed",
                        key="tab8_country_radio"
                    )

                    # Copy dataframe
                    _df_tab8 = df_overview.copy()

                    # Normalisasi country
                    _df_tab8["_country_up"] = (
                        _df_tab8["Country"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                    )

                    # Alias domestik (jika data tidak konsisten)
                    domestic_alias = ["INDONESIA", "ID", "IDN"]

                    if tab8_country_filter == "Domestik":
                        df_tab8 = _df_tab8[_df_tab8["_country_up"].isin(domestic_alias)].copy()
                        tab8_label = "🇮🇩 Domestik"
                    else:
                        df_tab8 = _df_tab8[~_df_tab8["_country_up"].isin(domestic_alias)].copy()
                        tab8_label = "🌍 Internasional"

                    df_tab8.drop(columns=["_country_up"], inplace=True)

                    if df_tab8.empty:
                        st.warning(f"⚠️ Tidak ada data untuk filter: {tab8_label}")
                        return  # keluar dari tab ini saja, tidak menghentikan tab lain

                else:
                    df_tab8 = df_overview.copy()
                    tab8_label = "Semua Data"

                # =========================
                # VISUALISASI
                # =========================
                cols1, cols2 = st.columns(2)

                # ==================================================
                # COL 1 — TOP 100 HOTEL
                # ==================================================
                with cols1:
                    if "Hotel Name" in df_tab8.columns and "Number of Rooms Night" in df_tab8.columns:

                        top_hotels_tab = (
                            df_tab8.groupby("Hotel Name")["Number of Rooms Night"]
                            .sum()
                            .sort_values(ascending=False)
                            .head(100)
                            .reset_index()
                        )

                        top_hotels_tab["Rank"] = top_hotels_tab.index + 1
                        top_hotels_tab["Highlight"] = top_hotels_tab["Rank"].apply(
                            lambda x: "Top 20" if x <= 20 else "Others"
                        )

                        fig_hotels = px.bar(
                            top_hotels_tab,
                            x="Number of Rooms Night",
                            y="Hotel Name",
                            orientation="h",
                            color="Highlight",
                            color_discrete_map={
                                "Top 20": "#1BA0E2",
                                "Others": "#e0e0e0"
                            },
                            title=f"Top 100 Hotels by Total Room Nights · {tab8_label}"
                        )

                        fig_hotels.update_traces(
                            texttemplate="%{x:,.0f}",
                            textposition="outside",
                            textfont_size=10
                        )

                        fig_hotels.update_layout(
                            height=1700,
                            yaxis=dict(
                                autorange="reversed",
                                tickfont=dict(size=10)
                            ),
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            margin=dict(l=10, r=80, t=50, b=10)
                        )

                        st.plotly_chart(fig_hotels, use_container_width=True)

                        # Download
                        output_hotels = BytesIO()
                        top_hotels_tab.drop(columns=["Rank", "Highlight"]).to_excel(
                            output_hotels,
                            index=False,
                            sheet_name="Top 100 Hotels"
                        )
                        output_hotels.seek(0)

                        if st.session_state.get("role") == "Admin":
                            st.download_button(
                                label="⬇️ Download Data",
                                data=output_hotels,
                                file_name="top_100_hotels_by_room_nights.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_top100_hotels")
                        else:
                            st.markdown("""
                            <div style='background:#f9f9f9;border-left:3px solid #1BA0E2;
                            border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;'>
                            🔒 Download hanya tersedia untuk <strong>Admin</strong>
                            </div>
                            """, unsafe_allow_html=True)

                # ==================================================
                # COL 2 — TOP 100 CITY
                # ==================================================
                with cols2:

                    city_col = next(
                        (c for c in ["City", "City Destination"] if c in df_tab8.columns),
                        None
                    )

                    if city_col and "Number of Rooms Night" in df_tab8.columns:

                        top_cities_tab = (
                            df_tab8.groupby(city_col)["Number of Rooms Night"]
                            .sum()
                            .sort_values(ascending=False)
                            .head(100)
                            .reset_index()
                        )

                        top_cities_tab["Rank"] = top_cities_tab.index + 1
                        top_cities_tab["Highlight"] = top_cities_tab["Rank"].apply(
                            lambda x: "Top 20" if x <= 20 else "Others"
                        )

                        fig_cities_tab = px.bar(
                            top_cities_tab,
                            x="Number of Rooms Night",
                            y=city_col,
                            orientation="h",
                            color="Highlight",
                            color_discrete_map={
                                "Top 20": "#1BA0E2",
                                "Others": "#e0e0e0"
                            },
                            title=f"Top 100 Cities by Total Room Nights · {tab8_label}"
                        )

                        fig_cities_tab.update_traces(
                            texttemplate="%{x:,.0f}",
                            textposition="outside",
                            textfont_size=10
                        )

                        fig_cities_tab.update_layout(
                            height=1700,
                            yaxis=dict(
                                autorange="reversed",
                                tickfont=dict(size=10)
                            ),
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            margin=dict(l=10, r=80, t=50, b=10)
                        )

                        st.plotly_chart(fig_cities_tab, use_container_width=True)

                        # Download
                        output_cities = BytesIO()
                        top_cities_tab.drop(columns=["Rank", "Highlight"]).to_excel(
                            output_cities,
                            index=False,
                            sheet_name="Top 100 Cities"
                        )
                        output_cities.seek(0)

                        if st.session_state.get("role") == "Admin":
                            st.download_button(
                                label="⬇️ Download Data",
                                data=output_cities,
                                file_name="top_100_cities_by_room_nights.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_top100_cities")
                        else:
                            st.markdown("""
                            <div style='background:#f9f9f9;border-left:3px solid #1BA0E2;
                            border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;'>
                            🔒 Download hanya tersedia untuk <strong>Admin</strong>
                            </div>
                            """, unsafe_allow_html=True)

            # ======================================
            # TAB 9: DENDROGRAM CLUSTERING — MTRAX Blue Theme
            # ======================================
            _render_tab8()
        with tab9:
            from plotly.subplots import make_subplots as _make_subplots
            from scipy.cluster.hierarchy import linkage as sk_linkage, dendrogram as scipy_dendrogram, fcluster
            from sklearn.preprocessing import normalize as sk_normalize

            # ── CSS ─────────────────────────────────────────────────────────
            st.markdown("""
            <style>
            :root {
                --dnd-blue:      #1BA0E2;
                --dnd-blue-mid:  #1494C6;
                --dnd-blue-dark: #0D7FCC;
                --dnd-navy:      #062440;
                --dnd-lime:      #e2f871;
                --dnd-lime-v:    #98ea16;
                --dnd-bg:        #f0f8ff;
                --dnd-border:    #cce4f4;
                --dnd-surface:   #ffffff;
                --dnd-muted:     #6a8fa0;
                --dnd-text:      #1a2a3a;
            }

            /* ════════════ HEADER ════════════ */
            .dnd-header {
                background: linear-gradient(135deg, #062440 0%, #0D7FCC 52%, #1BA0E2 100%);
                border-radius: 14px;
                padding: 28px 36px;
                margin-bottom: 24px;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(13,127,204,0.28);
            }
            .dnd-header::before {
                content: '';
                position: absolute; top: -70px; right: -50px;
                width: 240px; height: 240px; border-radius: 50%;
                background: rgba(255,255,255,0.05); pointer-events: none;
            }
            .dnd-header::after {
                content: '';
                position: absolute; bottom: -50px; left: 28%;
                width: 200px; height: 200px; border-radius: 50%;
                background: rgba(226,248,113,0.08); pointer-events: none;
            }
            .dnd-header-inner {
                display: flex; align-items: center; gap: 18px;
                position: relative; z-index: 1;
            }
            .dnd-header-icon {
                width: 54px; height: 54px; border-radius: 13px;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.24);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.7em; flex-shrink: 0;
            }
            .dnd-header-body  { flex: 1; }
            .dnd-header-title { color: #fff; font-size: 1.45em; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
            .dnd-header-sub   { color: rgba(255,255,255,0.58); font-size: 0.82em; margin-top: 4px; }
            .dnd-header-badge {
                display: inline-flex; align-items: center; gap: 7px;
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(255,255,255,0.22);
                border-radius: 100px; padding: 6px 16px;
                font-size: 0.72em; font-weight: 600;
                color: rgba(255,255,255,0.88); letter-spacing: 0.06em;
                white-space: nowrap;
            }
            .dnd-badge-dot {
                width: 6px; height: 6px; border-radius: 50%;
                background: var(--dnd-lime);
                box-shadow: 0 0 6px var(--dnd-lime);
                animation: dnd-blink 2s ease infinite;
            }
            @keyframes dnd-blink { 0%,100%{opacity:1} 50%{opacity:.3} }

            /* ════════════ SECTION DIVIDER ════════════ */
            .dnd-divider {
                display: flex; align-items: center; gap: 14px;
                margin: 26px 0 18px;
                font-size: 0.69em; font-weight: 700;
                color: var(--dnd-muted); letter-spacing: 0.14em;
                text-transform: uppercase;
            }
            .dnd-divider::before,
            .dnd-divider::after { content: ''; flex: 1; height: 1px; background: var(--dnd-border); }

            /* ════════════ FILTER PANEL ════════════ */
            .dnd-filter-panel {
                background: var(--dnd-surface);
                border: 1px solid var(--dnd-border);
                border-left: 4px solid var(--dnd-blue);
                border-radius: 10px;
                padding: 16px 20px 12px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(13,127,204,0.07);
            }
            .dnd-filter-label {
                font-size: 0.69em; font-weight: 700;
                color: var(--dnd-blue); text-transform: uppercase;
                letter-spacing: 0.12em; margin-bottom: 10px;
                display: flex; align-items: center; gap: 8px;
            }
            .dnd-filter-tag {
                background: var(--dnd-bg); border: 1px solid var(--dnd-border);
                color: var(--dnd-blue); border-radius: 20px;
                padding: 2px 10px; font-size: 0.85em; font-weight: 600;
            }

            /* ════════════ KPI CARDS ════════════ */
            .dnd-kpi-grid {
                display: grid; grid-template-columns: repeat(4,1fr);
                gap: 14px; margin-bottom: 20px;
            }
            .dnd-kpi {
                background: var(--dnd-surface);
                border-radius: 10px; padding: 18px 16px 14px;
                border-top: 3px solid var(--dnd-color, #1BA0E2);
                box-shadow: 0 2px 10px rgba(13,127,204,0.08);
                transition: transform .18s ease, box-shadow .18s ease;
            }
            .dnd-kpi:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(13,127,204,0.15); }
            .dnd-kpi-label {
                font-size: 0.67em; font-weight: 600; color: var(--dnd-muted);
                text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 8px;
            }
            .dnd-kpi-value { font-size: 1.35em; font-weight: 700; color: var(--dnd-color, #1BA0E2); line-height: 1.2; }
            .dnd-kpi-sub   { font-size: 0.72em; color: var(--dnd-muted); margin-top: 4px; }

            /* ════════════ CLUSTER CARDS ════════════ */
            .dnd-cluster-grid {
                display: grid; grid-template-columns: repeat(4,1fr);
                gap: 14px; margin-top: 8px;
            }
            .dnd-cluster-card {
                background: var(--dnd-surface);
                border-radius: 12px; padding: 18px 18px 16px;
                border-top: 4px solid var(--cl-color, #1BA0E2);
                box-shadow: 0 2px 12px rgba(13,127,204,0.08);
                transition: transform .18s ease, box-shadow .18s ease;
                position: relative; overflow: hidden;
            }
            .dnd-cluster-card::before {
                content: '';
                position: absolute; top: -20px; right: -20px;
                width: 80px; height: 80px; border-radius: 50%;
                background: var(--cl-color, #1BA0E2);
                opacity: 0.06;
            }
            .dnd-cluster-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(13,127,204,0.14); }

            .dnd-cluster-head {
                display: flex; justify-content: space-between;
                align-items: center; margin-bottom: 12px;
            }
            .dnd-cluster-name {
                font-weight: 700; font-size: 0.92em;
                color: var(--cl-color, #1BA0E2);
            }
            .dnd-cluster-count {
                background: color-mix(in srgb, var(--cl-color, #1BA0E2) 12%, white);
                color: var(--cl-color, #1BA0E2);
                border: 1px solid color-mix(in srgb, var(--cl-color, #1BA0E2) 25%, white);
                border-radius: 100px; padding: 2px 10px;
                font-size: 0.70em; font-weight: 700;
            }
            .dnd-cluster-members {
                font-size: 0.74em; color: #4a5a6a;
                line-height: 1.6; margin-bottom: 14px;
                min-height: 48px;
            }
            .dnd-cluster-stat {
                background: var(--dnd-bg);
                border-radius: 8px; padding: 10px 12px;
                border-left: 3px solid var(--cl-color, #1BA0E2);
            }
            .dnd-cluster-stat-value {
                font-size: 1.0em; font-weight: 700;
                color: var(--cl-color, #1BA0E2); line-height: 1.2;
            }
            .dnd-cluster-stat-label {
                font-size: 0.68em; font-weight: 600;
                color: var(--dnd-muted); margin-top: 2px;
            }

            /* ════════════ INSIGHT BOX ════════════ */
            .dnd-insight {
                background: var(--dnd-bg);
                border: 1px solid var(--dnd-border);
                border-left: 4px solid var(--dnd-blue);
                border-radius: 8px; padding: 14px 18px;
                font-size: 0.84em; color: #2a3a4a; line-height: 1.65;
                margin-top: 4px;
            }
            .dnd-insight strong { color: var(--dnd-navy); }
            .dnd-insight em     { color: var(--dnd-blue); font-style: normal; font-weight: 600; }
            </style>

            <!-- ── Header ── -->
            <div class="dnd-header">
                <div class="dnd-header-inner">
                    <div class="dnd-header-icon">🌿</div>
                    <div class="dnd-header-body">
                        <div class="dnd-header-title">Hierarchical Clustering — Dendrogram</div>
                        <div class="dnd-header-sub">Segmentasi pola perjalanan berbasis kemiripan · Ward / Complete / Average / Single linkage</div>
                    </div>
                    <div class="dnd-header-badge">
                        <span class="dnd-badge-dot"></span>CLUSTERING
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Filter Panel ────────────────────────────────────────────────
            st.markdown("""
            <div class="dnd-filter-panel">
                <div class="dnd-filter-label">
                    ⚙ Parameter Clustering
                    <span class="dnd-filter-tag">Semua perubahan langsung memperbarui chart</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            dend_r1 = st.columns([2, 2, 2, 2])
            with dend_r1[0]:
                dend_entity = st.selectbox(
                    "🎯 Entitas", ["Hotel", "Perusahaan", "Kota"], key="dend_entity",
                    help="Pilih entitas yang akan dikelompokkan"
                )
            with dend_r1[1]:
                dend_metric_col = st.selectbox(
                    "📐 Metric",
                    ["Invoice Amount", "Number of Rooms Night", "Travel Request Number"],
                    key="dend_metric",
                    help="Metric yang digunakan sebagai dasar clustering"
                )
            with dend_r1[2]:
                dend_method = st.selectbox(
                    "🔗 Linkage",
                    ["ward", "complete", "average", "single"],
                    key="dend_method",
                    help="Algoritma penggabungan klaster"
                )
            with dend_r1[3]:
                dend_top_n = st.selectbox(
                    "🔢 Top N", [10, 15, 20, 25, 30, 40, 50],
                    index=2, key="dend_top_n",
                    help="Jumlah entitas teratas yang dianalisis"
                )

            dend_r2 = st.columns([3, 2])
            with dend_r2[0]:
                n_clusters_dend = st.slider(
                    "🎨 Jumlah Klaster", 2, 8, 4, key="dend_clusters",
                    help="Jumlah kelompok yang terbentuk dari pemotongan dendrogram"
                )
            with dend_r2[1]:
                dend_show_bar = st.checkbox(
                    "Tampilkan bar spend chart", value=True, key="dend_bar"
                )

            # ── Data preparation ─────────────────────────────────────────────
            _city_col = next((c for c in df_all.columns if c in ["City", "City Destination"]), None)
            entity_map_dend  = {"Hotel": "Hotel Name", "Perusahaan": "Nama Perusahaan", "Kota": _city_col}
            pivot_rows_dend  = {
                "Hotel":     "Nama Perusahaan",
                "Perusahaan": _city_col if _city_col else "Hotel Name",
                "Kota":      "Nama Perusahaan"
            }

            entity_col_dend = entity_map_dend.get(dend_entity)
            row_col_dend    = pivot_rows_dend.get(dend_entity)
            metric_col_dend = next((c for c in df_all.columns if c == dend_metric_col), None)

            if (entity_col_dend and entity_col_dend in df_all.columns
                    and metric_col_dend
                    and row_col_dend and row_col_dend in df_all.columns):

                df_dend = df_all[[entity_col_dend, row_col_dend, metric_col_dend]].dropna()
                top_ent = df_dend.groupby(entity_col_dend)[metric_col_dend].sum().nlargest(dend_top_n).index
                df_dend = df_dend[df_dend[entity_col_dend].isin(top_ent)]
                pivot_dend = (df_dend.groupby([entity_col_dend, row_col_dend])[metric_col_dend]
                              .sum().unstack(fill_value=0))
                X_dend = sk_normalize(pivot_dend.values, norm="l2")

                if X_dend.shape[0] >= 2:
                    Z_dend       = sk_linkage(X_dend, method=dend_method, metric="euclidean")
                    cluster_ids  = fcluster(Z_dend, t=n_clusters_dend, criterion="maxclust")

                    # MTRAX-aligned cluster palette — blue family + lime accents
                    DEND_PALETTE = [
                        "#1BA0E2",  # MTRAX blue
                        "#0D7FCC",  # blue dark
                        "#98ea16",  # lime vivid
                        "#062440",  # navy
                        "#1494C6",  # blue mid
                        "#e2f871",  # lime light
                        "#47b5e8",  # blue light
                        "#5a8fc0",  # steel blue
                    ]

                    def fmt_val(v, col):
                        return f"Rp{v:,.0f}" if col == "Invoice Amount" else f"{v:,.0f}"

                    labels_list     = pivot_dend.index.tolist()
                    total_spend_dend = df_dend[metric_col_dend].sum()
                    avg_spend_dend   = total_spend_dend / len(pivot_dend) if len(pivot_dend) else 0
                    leaf_colors_map  = {
                        lbl: DEND_PALETTE[(cid - 1) % len(DEND_PALETTE)]
                        for lbl, cid in zip(labels_list, cluster_ids)
                    }

                    # ── KPI Cards ──────────────────────────────────────────
                    st.markdown('<div class="dnd-divider">Ringkasan Analisis</div>', unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="dnd-kpi-grid">
                        <div class="dnd-kpi" style="--dnd-color:#1BA0E2;">
                            <div class="dnd-kpi-label">Total Entitas</div>
                            <div class="dnd-kpi-value">{len(pivot_dend)}</div>
                            <div class="dnd-kpi-sub">dari Top {dend_top_n} {dend_entity}</div>
                        </div>
                        <div class="dnd-kpi" style="--dnd-color:#0D7FCC;">
                            <div class="dnd-kpi-label">Jumlah Klaster</div>
                            <div class="dnd-kpi-value">{n_clusters_dend}</div>
                            <div class="dnd-kpi-sub">linkage: {dend_method}</div>
                        </div>
                        <div class="dnd-kpi" style="--dnd-color:#1494C6;">
                            <div class="dnd-kpi-label">Total {dend_metric_col[:16]}</div>
                            <div class="dnd-kpi-value">{fmt_val(total_spend_dend, metric_col_dend)}</div>
                            <div class="dnd-kpi-sub">keseluruhan entitas</div>
                        </div>
                        <div class="dnd-kpi" style="--dnd-color:#062440;">
                            <div class="dnd-kpi-label">Rata-rata per Entitas</div>
                            <div class="dnd-kpi-value">{fmt_val(avg_spend_dend, metric_col_dend)}</div>
                            <div class="dnd-kpi-sub">baseline clustering</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Build dendrogram ────────────────────────────────────
                    color_thresh_dend = Z_dend[-(n_clusters_dend - 1), 2] if n_clusters_dend > 1 else 0
                    dend_no_plot = scipy_dendrogram(
                        Z_dend, labels=labels_list, no_plot=True,
                        color_threshold=color_thresh_dend
                    )
                    leaves_order    = dend_no_plot["leaves"]
                    labels_ordered  = [labels_list[i] for i in leaves_order]
                    leaf_xs         = {lbl: 5 + 10 * i for i, lbl in enumerate(labels_ordered)}
                    spend_series    = (df_dend.groupby(entity_col_dend)[metric_col_dend]
                                       .sum().reindex(labels_ordered).fillna(0))

                    # ── Build Plotly figure ─────────────────────────────────
                    st.markdown('<div class="dnd-divider">Dendrogram Visualisasi</div>', unsafe_allow_html=True)

                    if dend_show_bar:
                        fig_ply = _make_subplots(
                            rows=1, cols=2,
                            column_widths=[0.68, 0.32],
                            horizontal_spacing=0.04,
                            shared_yaxes=True
                        )
                        dr, dc, br, bc = 1, 1, 1, 2
                    else:
                        fig_ply = go.Figure()
                        dr = dc = br = bc = None

                    def _add(trace):
                        if dend_show_bar:
                            fig_ply.add_trace(trace, row=dr, col=dc)
                        else:
                            fig_ply.add_trace(trace)

                    # Edge lines — subtle blue
                    for xi, yi in zip(dend_no_plot["icoord"], dend_no_plot["dcoord"]):
                        _add(go.Scatter(
                            x=yi, y=xi, mode="lines",
                            line=dict(color="rgba(27,160,226,0.32)", width=1.6),
                            hoverinfo="skip", showlegend=False
                        ))

                    # Leaf nodes
                    for lbl in labels_ordered:
                        xp  = leaf_xs[lbl]
                        nc  = leaf_colors_map.get(lbl, "#1BA0E2")
                        cid = cluster_ids[labels_list.index(lbl)]
                        sv  = spend_series.get(lbl, 0)
                        _add(go.Scatter(
                            x=[0], y=[xp],
                            mode="markers+text",
                            marker=dict(size=10, color=nc,
                                        line=dict(color="white", width=2),
                                        symbol="circle"),
                            text=[lbl],
                            textposition="middle left",
                            textfont=dict(size=9, color=nc),
                            hovertemplate=(
                                f"<b>{lbl}</b><br>"
                                f"Klaster: <b>{cid}</b><br>"
                                f"{dend_metric_col}: <b>{fmt_val(sv, metric_col_dend)}</b>"
                                "<extra></extra>"
                            ),
                            showlegend=False
                        ))

                    # Cut threshold line
                    if n_clusters_dend > 1:
                        y_range = [min(leaf_xs.values()) - 5, max(leaf_xs.values()) + 5]
                        _add(go.Scatter(
                            x=[color_thresh_dend, color_thresh_dend], y=y_range,
                            mode="lines",
                            line=dict(color="#d9534f", width=1.6, dash="dash"),
                            name=f"Cut @ {color_thresh_dend:.3f}",
                            showlegend=True
                        ))

                    # Bar chart (spend per entity)
                    if dend_show_bar:
                        bar_colors = [leaf_colors_map.get(l, "#1BA0E2") for l in labels_ordered]
                        fig_ply.add_trace(
                            go.Bar(
                                y=[leaf_xs[l] for l in labels_ordered],
                                x=spend_series.values,
                                orientation="h",
                                marker=dict(color=bar_colors, opacity=0.85,
                                            line=dict(color="white", width=0.5)),
                                showlegend=False,
                                name="Spend"
                            ),
                            row=br, col=bc
                        )

                    # Legend entries per cluster
                    for i in range(n_clusters_dend):
                        fig_ply.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker=dict(size=10, color=DEND_PALETTE[i % len(DEND_PALETTE)]),
                            name=f"Klaster {i + 1}",
                            showlegend=True
                        ))

                    # Layout
                    chart_h  = max(520, len(pivot_dend) * 26 + 80)
                    max_diss = max((max(d) for d in dend_no_plot["dcoord"]), default=1.0)
                    x_left   = -max(0.55 * max_diss, 0.35)

                    _ax = dict(
                        showgrid=True,
                        gridcolor="rgba(204,228,244,0.50)",
                        gridwidth=0.6,
                        zeroline=False,
                        tickfont=dict(size=8, color="#6a8fa0"),
                        linecolor="#cce4f4",
                        linewidth=1,
                        showline=True
                    )

                    fig_ply.update_layout(
                        height=chart_h,
                        paper_bgcolor="#f0f8ff",
                        plot_bgcolor="#f0f8ff",
                        font=dict(size=11, color="#1a2a3a"),
                        title=dict(
                            text=(
                                f"<b>Dendrogram — {dend_entity}</b>"
                                f"<span style='color:#6a8fa0;font-size:11px;'>"
                                f"  ·  Linkage: {dend_method}"
                                f"  ·  {n_clusters_dend} Klaster"
                                f"  ·  Metric: {dend_metric_col}"
                                f"</span>"
                            ),
                            x=0.01, xanchor="left",
                            font=dict(size=13, color="#1a2a3a")
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.01,
                            xanchor="right", x=1,
                            bgcolor="rgba(240,248,255,0.95)",
                            bordercolor="#cce4f4",
                            borderwidth=1,
                            font=dict(size=10, color="#1a2a3a")
                        ),
                        margin=dict(l=10, r=20, t=60, b=40),
                        hovermode="closest",
                        bargap=0.10
                    )

                    fig_ply.update_xaxes(**_ax)
                    fig_ply.update_yaxes(**_ax)
                    fig_ply.update_xaxes(
                        title_text="Dissimilarity",
                        range=[x_left, max_diss * 1.08],
                        row=dr, col=dc
                    )
                    fig_ply.update_yaxes(
                        showticklabels=False, showgrid=False,
                        row=dr, col=dc
                    )
                    if dend_show_bar:
                        fig_ply.update_xaxes(
                            title_text=dend_metric_col[:18],
                            row=br, col=bc
                        )
                        fig_ply.update_yaxes(
                            showticklabels=False, showgrid=False,
                            row=br, col=bc
                        )

                    st.plotly_chart(fig_ply, use_container_width=True)

                    # ── Cluster Summary Cards ──────────────────────────────
                    st.markdown('<div class="dnd-divider">Ringkasan Klaster</div>', unsafe_allow_html=True)

                    cluster_df_dend = pd.DataFrame({
                        dend_entity: labels_list,
                        "Klaster":   cluster_ids,
                        "Total": (
                            df_dend.groupby(entity_col_dend)[metric_col_dend]
                            .sum().reindex(labels_list).values
                        )
                    }).sort_values(["Klaster", "Total"], ascending=[True, False])

                    unique_clusters = sorted(cluster_df_dend["Klaster"].unique())

                    # Render cards in rows of 4
                    for row_start in range(0, len(unique_clusters), 4):
                        row_clusters = unique_clusters[row_start:row_start + 4]
                        cols_c = st.columns(len(row_clusters))

                        for col_c, cid in zip(cols_c, row_clusters):
                            c_color  = DEND_PALETTE[(cid - 1) % len(DEND_PALETTE)]
                            sub      = cluster_df_dend[cluster_df_dend["Klaster"] == cid]
                            total_c  = sub["Total"].sum()
                            pct_c    = total_c / total_spend_dend * 100 if total_spend_dend else 0
                            members  = sub[dend_entity].tolist()
                            shown    = members[:4]
                            more_n   = max(0, len(members) - 4)
                            members_html = "<br>".join(
                                [f"<span style='color:#4a5a6a;'>• {m}</span>" for m in shown]
                            )
                            if more_n > 0:
                                members_html += (
                                    f"<br><span style='color:#a0b0c0;font-size:0.88em;'>"
                                    f"+{more_n} lainnya…</span>"
                                )

                            with col_c:
                                st.markdown(f"""
                                <div class="dnd-cluster-card" style="--cl-color:{c_color};">
                                    <div class="dnd-cluster-head">
                                        <span class="dnd-cluster-name">Klaster {cid}</span>
                                        <span class="dnd-cluster-count">{len(sub)} entitas</span>
                                    </div>
                                    <div class="dnd-cluster-members">{members_html}</div>
                                    <div class="dnd-cluster-stat">
                                        <div class="dnd-cluster-stat-value">{fmt_val(total_c, metric_col_dend)}</div>
                                        <div class="dnd-cluster-stat-label">{pct_c:.1f}% dari total · {dend_metric_col[:18]}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    # ── Insight Box ────────────────────────────────────────
                    st.markdown('<div class="dnd-divider">Interpretasi</div>', unsafe_allow_html=True)

                    # Find dominant cluster
                    dominant = cluster_df_dend.groupby("Klaster")["Total"].sum().idxmax()
                    dominant_pct = (cluster_df_dend[cluster_df_dend["Klaster"] == dominant]["Total"].sum()
                                    / total_spend_dend * 100)

                    st.markdown(f"""
                    <div class="dnd-insight">
                        <strong>Hasil Clustering — {dend_entity} · {dend_method} linkage · {n_clusters_dend} klaster</strong><br><br>
                        Dari <em>{len(pivot_dend)} {dend_entity}</em> yang dianalisis,
                        terbentuk <em>{n_clusters_dend} klaster</em> berdasarkan kemiripan pola
                        <em>{dend_metric_col}</em>.<br><br>
                        <strong>Klaster {dominant}</strong> mendominasi dengan kontribusi
                        <em>{dominant_pct:.1f}%</em> dari total.
                        Entitas pada klaster yang sama memiliki profil perjalanan serupa dan berpotensi
                        untuk dikelola dengan <strong>strategi pengadaan bersama</strong> guna meningkatkan
                        leverage negosiasi.
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style='background:#f0f8ff;border:1px solid #cce4f4;
                            border-left:4px solid #1BA0E2;border-radius:8px;
                            padding:16px 20px;font-size:0.88em;color:#0D7FCC;'>
                    ⚠️ Data tidak cukup untuk membuat dendrogram dengan parameter yang dipilih.
                    Pastikan kolom <strong>Hotel Name</strong>, <strong>Nama Perusahaan</strong>,
                    dan <strong>Invoice Amount</strong> tersedia.
                </div>
                """, unsafe_allow_html=True)

        # ======================================
        # TAB 10: EXPORT
        # ======================================
        with tab10:
            st.markdown("<div class='section-title'>Export Data</div>", unsafe_allow_html=True)
            st.markdown("Export your data in various formats for further analysis.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### CSV Format")
                st.markdown("Compatible with Excel, Google Sheets")
                if st.button("Download CSV", use_container_width=True, type="primary"):
                    csv = df_all.to_csv(index=False).encode("utf-8")
                    if st.session_state.get('role') == 'Admin':
                        st.download_button("⬇️ Download",data=csv,
                            file_name=f"mtrax_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",use_container_width=True,
                            key="dl_export_csv")
                    else:
                        st.markdown("""<div style='background:#f9f9f9;border-left:3px solid #1BA0E2;border-radius:6px;
                        padding:10px 16px;font-size:0.82em;color:#1BA0E2;'>🔒 Download hanya tersedia untuk <strong>Admin</strong></div>""",
                        unsafe_allow_html=True)

            with col2:
                st.markdown("#### Excel Format")
                st.markdown("Microsoft Excel workbook")
                if st.button("Download Excel", use_container_width=True, type="primary"):
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        df_all.to_excel(writer, index=False, sheet_name="Travel Data")
                    if st.session_state.get('role') == 'Admin':
                        st.download_button("⬇️ Download",data=buffer.getvalue(),
                            file_name=f"mtrax_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="dl_export_xlsx")
                    else:
                        st.markdown("""<div style='background:#f9f9f9;border-left:3px solid #1BA0E2;border-radius:6px;
                        padding:10px 16px;font-size:0.82em;color:#1BA0E2;'>🔒 Download hanya tersedia untuk <strong>Admin</strong></div>""",
                        unsafe_allow_html=True)

            with col3:
                st.markdown("#### JSON Format")
                st.markdown("For APIs and data exchange")
                if st.button("Download JSON", use_container_width=True, type="primary"):
                    json_data = df_all.to_json(orient="records", date_format="iso")
                    if st.session_state.get('role') == 'Admin':
                        st.download_button("⬇️ Download",data=json_data,
                            file_name=f"mtrax_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",use_container_width=True,
                            key="dl_export_json")
                    else:
                        st.markdown("""<div style='background:#f9f9f9;border-left:3px solid #1BA0E2;border-radius:6px;
                        padding:10px 16px;font-size:0.82em;color:#1BA0E2;'>🔒 Download hanya tersedia untuk <strong>Admin</strong></div>""",
                        unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### Export Statistics")

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Records", f"{len(df_all):,}")
            with col2: st.metric("Total Columns", f"{len(df_all.columns)}")
            with col3:
                memory_usage = df_all.memory_usage(deep=True).sum()/1024/1024
                st.metric("Memory Usage", f"{memory_usage:.2f} MB")
            with col4: st.metric("Est. File Size", f"{memory_usage*0.8:.2f} MB")
            
        # ======================================
        # TAB 11: PATRA JASA GROUP vs NON-PATRA JASA
        # ======================================
        with tab11:
            st.markdown("<div class='section-title'>Patra Jasa Group vs Non-Patra Jasa — Perbandingan Hotel Domestik</div>", unsafe_allow_html=True)

            # ── Kamus Hotel Patra Jasa Group ──
            PATRA_JASA_HOTELS = [
                "The Patra Bali Resort & Villas",
                "Patra Semarang Hotel & Convention",
                "Patra Cirebon Hotel & Convention",
                "Patra Malioboro Hotel",
                "Patra Dumai Hotel",
                "Patra Bandung Hotel",
                "Patra Jakarta Hotel",
                "Patra Anyer Hotel",
                "Patra Parapat Hotel",
            ]
            PATRA_JASA_NORMALIZED = [h.lower().strip() for h in PATRA_JASA_HOTELS]

            # ── Validasi kolom minimum ──
            _req_cols = {"Hotel Name", "Country"}
            _missing = _req_cols - set(df_all.columns)
            if _missing:
                st.warning(f"⚠️ Kolom berikut tidak ditemukan di data: {', '.join(_missing)}. Tab ini membutuhkan kolom tersebut.")
            else:
                # ── Filter INDONESIA saja ──
                df_pj = df_all.copy()
                df_pj["_country_up"] = df_pj["Country"].astype(str).str.strip().str.upper()
                df_pj = df_pj[df_pj["_country_up"] == "INDONESIA"].copy()

                if df_pj.empty:
                    st.info("ℹ️ Tidak ada data domestik (Country = INDONESIA) setelah filter global diterapkan.")
                else:
                    # ── Klasifikasi Patra / Non-Patra ──
                    df_pj["_hotel_norm"] = df_pj["Hotel Name"].astype(str).str.lower().str.strip()
                    df_pj["Grup Hotel"] = df_pj["_hotel_norm"].apply(
                        lambda x: "Patra Jasa Group" if x in PATRA_JASA_NORMALIZED else "Non-Patra Jasa"
                    )

                    # ── Tambahkan kolom bulan & tahun ──
                    # PERBAIKAN: sebelumnya hanya .dt.month tanpa memperhitungkan tahun,
                    # sehingga bulan yang sama dari tahun berbeda bisa tercampur/menyembunyikan data.
                    # PERBAIKAN 2: tambah fallback antar-kolom tanggal per baris — kalau kolom utama
                    # gagal di-parse (NaT) untuk sebagian baris, baris itu tidak lagi hilang diam-diam
                    # dari agregasi bulanan, melainkan dicoba pakai kolom tanggal lain yang tersedia.
                    _date_col_pj = None
                    for _c in ["Issue Time", "Check in Date"]:
                        if _c in df_pj.columns:
                            _date_col_pj = _c
                            break
                    if _date_col_pj:
                        df_pj["_dt"] = pd.to_datetime(df_pj[_date_col_pj], errors="coerce", dayfirst=True)
                        for _fb_col in [c for c in ["Issue Time", "Check in Date", "Check out Date"]
                                        if c in df_pj.columns and c != _date_col_pj]:
                            _fb_dt = pd.to_datetime(df_pj[_fb_col], errors="coerce", dayfirst=True)
                            df_pj["_dt"] = df_pj["_dt"].fillna(_fb_dt)
                        df_pj["_month"] = df_pj["_dt"].dt.month
                        df_pj["_year"] = df_pj["_dt"].dt.year

                        _n_unparsed = df_pj["_dt"].isna().sum()
                        if _n_unparsed > 0:
                            st.caption(
                                f"⚠️ {_n_unparsed:,} baris tidak punya tanggal yang bisa dibaca "
                                f"(kolom {_date_col_pj} dan alternatifnya kosong/format tidak dikenali) "
                                f"— baris ini tidak masuk ke tabel bulanan di bawah."
                            )

                    # ── Filter Nama Perusahaan (opsional) ──
                    st.markdown("""
                    <div style='background:var(--clr-surface,#fff);border:1px solid #e8eaf0;
                                border-left:4px solid #1BA0E2;border-radius:6px;
                                padding:10px 16px 8px 16px;margin-bottom:16px;'>
                        <div style='font-size:0.72em;font-weight:700;color:#1BA0E2;
                                    text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;'>
                            🔍 Filter Tab Ini
                        </div>
                    </div>""", unsafe_allow_html=True)

                    _pj_col1, _pj_col2, _pj_col3 = st.columns([1.6, 1, 1])

                    with _pj_col1:
                        if "Nama Perusahaan" in df_pj.columns:
                            _pj_company_opts = sorted(df_pj["Nama Perusahaan"].dropna().unique().tolist())
                            _pj_selected_companies = st.multiselect(
                                "🏢 Filter Perusahaan",
                                options=_pj_company_opts,
                                default=[],
                                placeholder="Semua perusahaan",
                                key="pj_company_filter"
                            )
                            if _pj_selected_companies:
                                df_pj = df_pj[df_pj["Nama Perusahaan"].isin(_pj_selected_companies)]
                        else:
                            _pj_selected_companies = []

                    with _pj_col2:
                        # PERBAIKAN: selector tahun eksplisit agar tabel bulanan tidak pernah
                        # mencampur bulan yang sama dari tahun berbeda (mis. Jan 2025 + Jan 2026
                        # tergabung jadi satu kolom "Jan"). Default ke tahun terbaru yang ada di data.
                        if "_year" in df_pj.columns and df_pj["_year"].notna().any():
                            _pj_year_opts = sorted(df_pj["_year"].dropna().astype(int).unique().tolist(), reverse=True)
                            _pj_selected_year = st.selectbox(
                                "📅 Tahun (Tabel Bulanan)",
                                options=_pj_year_opts,
                                index=0,
                                key="pj_year_select"
                            )
                        else:
                            _pj_selected_year = None

                    with _pj_col3:
                        _pj_metric_opt = st.selectbox(
                            "📊 Metrik Tabel",
                            options=["Invoice (unique)", "Room Nights"],
                            key="pj_metric_select"
                        )

                    # ── Hitung aggregasi ──
                    # Invoice unique = jumlah baris unik per Travel Request Number (atau row count jika tidak ada)
                    if "Travel Request Number" in df_pj.columns:
                        _inv_col = "Travel Request Number"
                        pj_inv  = df_pj[df_pj["Grup Hotel"] == "Patra Jasa Group"][_inv_col].nunique()
                        npj_inv = df_pj[df_pj["Grup Hotel"] == "Non-Patra Jasa"][_inv_col].nunique()
                    else:
                        pj_inv  = len(df_pj[df_pj["Grup Hotel"] == "Patra Jasa Group"])
                        npj_inv = len(df_pj[df_pj["Grup Hotel"] == "Non-Patra Jasa"])

                    total_inv = pj_inv + npj_inv
                    pj_inv_pct  = (pj_inv  / total_inv * 100) if total_inv > 0 else 0
                    npj_inv_pct = (npj_inv / total_inv * 100) if total_inv > 0 else 0

                    # Room Nights
                    if "Number of Rooms Night" in df_pj.columns:
                        df_pj["Number of Rooms Night"] = pd.to_numeric(df_pj["Number of Rooms Night"], errors="coerce")
                        pj_rn  = df_pj[df_pj["Grup Hotel"] == "Patra Jasa Group"]["Number of Rooms Night"].sum()
                        npj_rn = df_pj[df_pj["Grup Hotel"] == "Non-Patra Jasa"]["Number of Rooms Night"].sum()
                    else:
                        pj_rn, npj_rn = 0, 0

                    total_rn = pj_rn + npj_rn
                    pj_rn_pct  = (pj_rn  / total_rn * 100) if total_rn > 0 else 0
                    npj_rn_pct = (npj_rn / total_rn * 100) if total_rn > 0 else 0

                    # ── Metric Cards ──
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                    with _mc1:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Total Invoice (Domestik)</div>
                            <div class='metric-value'>{total_inv:,}</div>
                        </div>""", unsafe_allow_html=True)
                    with _mc2:
                        st.markdown(f"""
                        <div class='metric-box' style='border-left-color:#1BA0E2;'>
                            <div class='metric-label'>Patra Jasa Group</div>
                            <div class='metric-value' style='color:#1BA0E2;'>{pj_inv:,}</div>
                            <div style='font-size:0.78em;color:#888;margin-top:4px;'>{pj_inv_pct:.1f}% dari total</div>
                        </div>""", unsafe_allow_html=True)
                    with _mc3:
                        st.markdown(f"""
                        <div class='metric-box' style='border-left-color:#ff8c00;'>
                            <div class='metric-label'>Non-Patra Jasa</div>
                            <div class='metric-value' style='color:#ff8c00;'>{npj_inv:,}</div>
                            <div style='font-size:0.78em;color:#888;margin-top:4px;'>{npj_inv_pct:.1f}% dari total</div>
                        </div>""", unsafe_allow_html=True)
                    with _mc4:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Total Room Nights</div>
                            <div class='metric-value'>{total_rn:,.0f}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                    # ── Pie Charts ──
                    st.markdown("<div class='section-title'>Proporsi Perbandingan</div>", unsafe_allow_html=True)
                    _pie1, _pie2 = st.columns(2)

                    with _pie1:
                        if total_inv > 0:
                            fig_pj_inv = go.Figure(data=[go.Pie(
                                labels=["Patra Jasa Group", "Non-Patra Jasa"],
                                values=[pj_inv, npj_inv],
                                hole=0.55,
                                marker=dict(colors=["#1BA0E2", "#ff8c00"],
                                            line=dict(color="white", width=2)),
                                textinfo="percent",
                                textfont=dict(size=13),
                                hovertemplate="<b>%{label}</b><br>Invoice: %{value:,}<br>Proporsi: %{percent}<extra></extra>"
                            )])
                            fig_pj_inv.update_layout(
                                title=dict(text="Invoice Unique — Patra vs Non-Patra", font=dict(size=14)),
                                height=340,
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                                            xanchor="center", x=0.5, font=dict(size=11)),
                                margin=dict(l=10, r=10, t=60, b=20),
                                annotations=[dict(
                                    text=f"<b>{total_inv:,}</b><br><span style='font-size:10px'>total</span>",
                                    x=0.5, y=0.5, font=dict(size=15), showarrow=False
                                )]
                            )
                            st.plotly_chart(fig_pj_inv, use_container_width=True)

                            # Tabel ringkasan invoice
                            st.markdown(f"""
                            <table style='width:100%;border-collapse:collapse;font-size:0.83em;'>
                              <thead>
                                <tr style='background:#f0f8ff;'>
                                  <th style='padding:6px 10px;text-align:left;border:1px solid #e0e0e0;color:#555;'>Grup</th>
                                  <th style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;color:#555;'>Invoice</th>
                                  <th style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;color:#555;'>%</th>
                                </tr>
                              </thead>
                              <tbody>
                                <tr>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;'>
                                    <span style='display:inline-block;width:10px;height:10px;background:#1BA0E2;border-radius:2px;margin-right:6px;vertical-align:middle;'></span>Patra Jasa Group
                                  </td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;color:#1BA0E2;'>{pj_inv:,}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;'>{pj_inv_pct:.1f}%</td>
                                </tr>
                                <tr>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;'>
                                    <span style='display:inline-block;width:10px;height:10px;background:#ff8c00;border-radius:2px;margin-right:6px;vertical-align:middle;'></span>Non-Patra Jasa
                                  </td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;color:#ff8c00;'>{npj_inv:,}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;'>{npj_inv_pct:.1f}%</td>
                                </tr>
                                <tr style='background:#f9f9f9;'>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;font-weight:600;'>Total</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;'>{total_inv:,}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;'>100%</td>
                                </tr>
                              </tbody>
                            </table>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Tidak ada data invoice untuk ditampilkan.")

                    with _pie2:
                        if total_rn > 0:
                            fig_pj_rn = go.Figure(data=[go.Pie(
                                labels=["Patra Jasa Group", "Non-Patra Jasa"],
                                values=[pj_rn, npj_rn],
                                hole=0.55,
                                marker=dict(colors=["#1BA0E2", "#ff8c00"],
                                            line=dict(color="white", width=2)),
                                textinfo="percent",
                                textfont=dict(size=13),
                                hovertemplate="<b>%{label}</b><br>Room Nights: %{value:,.0f}<br>Proporsi: %{percent}<extra></extra>"
                            )])
                            fig_pj_rn.update_layout(
                                title=dict(text="Room Nights — Patra vs Non-Patra", font=dict(size=14)),
                                height=340,
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                                            xanchor="center", x=0.5, font=dict(size=11)),
                                margin=dict(l=10, r=10, t=60, b=20),
                                annotations=[dict(
                                    text=f"<b>{total_rn:,.0f}</b><br><span style='font-size:10px'>total</span>",
                                    x=0.5, y=0.5, font=dict(size=15), showarrow=False
                                )]
                            )
                            st.plotly_chart(fig_pj_rn, use_container_width=True)

                            # Tabel ringkasan room nights
                            st.markdown(f"""
                            <table style='width:100%;border-collapse:collapse;font-size:0.83em;'>
                              <thead>
                                <tr style='background:#f0f8ff;'>
                                  <th style='padding:6px 10px;text-align:left;border:1px solid #e0e0e0;color:#555;'>Grup</th>
                                  <th style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;color:#555;'>Room Nights</th>
                                  <th style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;color:#555;'>%</th>
                                </tr>
                              </thead>
                              <tbody>
                                <tr>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;'>
                                    <span style='display:inline-block;width:10px;height:10px;background:#1BA0E2;border-radius:2px;margin-right:6px;vertical-align:middle;'></span>Patra Jasa Group
                                  </td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;color:#1BA0E2;'>{pj_rn:,.0f}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;'>{pj_rn_pct:.1f}%</td>
                                </tr>
                                <tr>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;'>
                                    <span style='display:inline-block;width:10px;height:10px;background:#ff8c00;border-radius:2px;margin-right:6px;vertical-align:middle;'></span>Non-Patra Jasa
                                  </td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;color:#ff8c00;'>{npj_rn:,.0f}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;'>{npj_rn_pct:.1f}%</td>
                                </tr>
                                <tr style='background:#f9f9f9;'>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;font-weight:600;'>Total</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;'>{total_rn:,.0f}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;'>100%</td>
                                </tr>
                              </tbody>
                            </table>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Kolom 'Number of Rooms Night' tidak ditemukan atau tidak ada data.")

                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                    # ── Tabel Bulanan ──
                    _pj_year_label = f" · Tahun {_pj_selected_year}" if _pj_selected_year is not None else ""
                    st.markdown(
                        f"<div class='section-title'>Tabel Perbandingan Bulanan — "
                        f"{'Invoice (unique)' if _pj_metric_opt == 'Invoice (unique)' else 'Room Nights'}"
                        f"{_pj_year_label}</div>",
                        unsafe_allow_html=True
                    )

                    # PERBAIKAN: scope tabel bulanan ke SATU tahun yang dipilih di atas —
                    # ini mencegah bulan yang sama dari tahun berbeda tergabung jadi satu kolom
                    # (mis. data Jan 2025 + Jan 2026 tidak lagi ikut menumpuk/menyembunyikan bulan lain).
                    df_pj_month_table = df_pj.copy()
                    if _pj_selected_year is not None and "_year" in df_pj_month_table.columns:
                        df_pj_month_table = df_pj_month_table[df_pj_month_table["_year"] == _pj_selected_year]

                    if _date_col_pj and "_month" in df_pj_month_table.columns and not df_pj_month_table.empty:
                        MONTH_NAMES = {
                            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
                            7:"Jul",8:"Agt",9:"Sep",10:"Okt",11:"Nov",12:"Des"
                        }

                        if _pj_metric_opt == "Invoice (unique)":
                            if "Travel Request Number" in df_pj_month_table.columns:
                                _monthly_agg = (
                                    df_pj_month_table.groupby(["Grup Hotel","_month"])["Travel Request Number"]
                                    .nunique()
                                    .reset_index(name="Nilai")
                                )
                            else:
                                _monthly_agg = (
                                    df_pj_month_table.groupby(["Grup Hotel","_month"])
                                    .size()
                                    .reset_index(name="Nilai")
                                )
                        else:
                            if "Number of Rooms Night" in df_pj_month_table.columns:
                                _monthly_agg = (
                                    df_pj_month_table.groupby(["Grup Hotel","_month"])["Number of Rooms Night"]
                                    .sum()
                                    .reset_index(name="Nilai")
                                )
                            else:
                                st.warning("Kolom 'Number of Rooms Night' tidak tersedia.")
                                _monthly_agg = pd.DataFrame()

                        if not _monthly_agg.empty:
                            _pivot = _monthly_agg.pivot_table(
                                index="Grup Hotel",
                                columns="_month",
                                values="Nilai",
                                fill_value=0
                            )
                            # PERBAIKAN: normalisasi label kolom ke int murni — pivot_table bisa
                            # menghasilkan kolom bertipe float (mis. 4.0) kalau ada NaN tercampur
                            # di data sumber sebelum di-groupby, sehingga pengecekan
                            # "if _m not in _pivot.columns" di bawah (memakai int biasa) gagal
                            # mengenali kolom yang sebenarnya sudah ada, lalu menimpanya dengan 0.
                            _pivot.columns = [int(c) for c in _pivot.columns]

                            # Pastikan semua 12 bulan ada
                            for _m in range(1, 13):
                                if _m not in _pivot.columns:
                                    _pivot[_m] = 0
                            _pivot = _pivot[[m for m in range(1, 13)]]
                            _pivot["Total"] = _pivot.sum(axis=1)
                            _pivot = _pivot.reset_index()

                            # Reorder rows: Patra Jasa dulu
                            _row_order = ["Patra Jasa Group", "Non-Patra Jasa"]
                            _pivot["_sort"] = _pivot["Grup Hotel"].map(
                                {r: i for i, r in enumerate(_row_order)}
                            ).fillna(99)
                            _pivot = _pivot.sort_values("_sort").drop(columns=["_sort"])

                            # Tambah row % share Patra
                            _totals_by_month = {m: _pivot[m].sum() for m in range(1, 13)}
                            _pj_row = _pivot[_pivot["Grup Hotel"] == "Patra Jasa Group"]
                            _share_row = {"Grup Hotel": "% Patra share"}
                            for _m in range(1, 13):
                                _denom = _totals_by_month[_m]
                                _num = _pj_row[_m].values[0] if not _pj_row.empty else 0
                                _share_row[_m] = f"{(_num/_denom*100):.0f}%" if _denom > 0 else "-"
                            _total_denom = _pivot["Total"].sum()
                            _pj_total = _pj_row["Total"].values[0] if not _pj_row.empty else 0
                            _share_row["Total"] = f"{(_pj_total/_total_denom*100):.0f}%" if _total_denom > 0 else "-"

                            # Build HTML table
                            _th_style = "padding:7px 10px;background:#f0f8ff;border:1px solid #d0dde8;font-size:0.8em;color:#444;text-align:center;white-space:nowrap;"
                            _th_left  = "padding:7px 10px;background:#f0f8ff;border:1px solid #d0dde8;font-size:0.8em;color:#444;text-align:left;white-space:nowrap;min-width:150px;"

                            _html_tbl = f"""
                            <div style='overflow-x:auto;'>
                            <table style='width:100%;border-collapse:collapse;font-size:0.82em;'>
                              <thead>
                                <tr>
                                  <th style='{_th_left}'>Grup Hotel</th>
                                  {"".join(f"<th style='{_th_style}'>{MONTH_NAMES[m]}</th>" for m in range(1,13))}
                                  <th style='{_th_style}font-weight:700;background:#daeaf8;'>Total</th>
                                </tr>
                              </thead>
                              <tbody>
                            """

                            for _, row in _pivot.iterrows():
                                _grup = row["Grup Hotel"]
                                if _grup == "Patra Jasa Group":
                                    _row_bg  = "background:rgba(27,160,226,0.07);"
                                    _val_col = "color:#1BA0E2;font-weight:600;"
                                    _badge   = "<span style='background:#e6f4fb;color:#0D7FCC;border-radius:20px;padding:1px 7px;font-size:0.78em;font-weight:600;margin-left:4px;'>PJ</span>"
                                else:
                                    _row_bg  = "background:rgba(255,140,0,0.05);"
                                    _val_col = "color:#cc6600;font-weight:600;"
                                    _badge   = "<span style='background:#fff4e6;color:#b05a00;border-radius:20px;padding:1px 7px;font-size:0.78em;font-weight:600;margin-left:4px;'>NPJ</span>"

                                _html_tbl += f"<tr style='{_row_bg}'>"
                                _html_tbl += f"<td style='padding:7px 10px;border:1px solid #e0e0e0;font-weight:500;'>{_grup}{_badge}</td>"
                                for _m in range(1, 13):
                                    _v = int(row[_m]) if _pj_metric_opt == "Invoice (unique)" else f"{row[_m]:,.0f}"
                                    _html_tbl += f"<td style='padding:7px 10px;border:1px solid #e0e0e0;text-align:right;{_val_col}'>{_v}</td>"
                                _total_v = int(row["Total"]) if _pj_metric_opt == "Invoice (unique)" else f"{row['Total']:,.0f}"
                                _html_tbl += f"<td style='padding:7px 10px;border:1px solid #daeaf8;text-align:right;background:#daeaf8;{_val_col}'>{_total_v}</td>"
                                _html_tbl += "</tr>"

                            # Row total gabungan
                            _html_tbl += "<tr style='background:#f5f5f5;font-weight:600;'>"
                            _html_tbl += "<td style='padding:7px 10px;border:1px solid #e0e0e0;'>Total Domestik</td>"
                            for _m in range(1, 13):
                                _col_total = _pivot[_m].sum()
                                _v = int(_col_total) if _pj_metric_opt == "Invoice (unique)" else f"{_col_total:,.0f}"
                                _html_tbl += f"<td style='padding:7px 10px;border:1px solid #e0e0e0;text-align:right;'>{_v}</td>"
                            _grand = _pivot["Total"].sum()
                            _grand_v = int(_grand) if _pj_metric_opt == "Invoice (unique)" else f"{_grand:,.0f}"
                            _html_tbl += f"<td style='padding:7px 10px;border:1px solid #daeaf8;text-align:right;background:#daeaf8;'>{_grand_v}</td>"
                            _html_tbl += "</tr>"

                            # Row % share Patra
                            _html_tbl += "<tr style='background:#fafafa;font-style:italic;'>"
                            _html_tbl += "<td style='padding:7px 10px;border:1px solid #e0e0e0;font-size:0.78em;color:#888;'>% Patra Jasa share</td>"
                            for _m in range(1, 13):
                                _html_tbl += f"<td style='padding:7px 10px;border:1px solid #e0e0e0;text-align:right;font-size:0.78em;color:#888;'>{_share_row[_m]}</td>"
                            _html_tbl += f"<td style='padding:7px 10px;border:1px solid #daeaf8;text-align:right;font-size:0.78em;color:#888;background:#daeaf8;'>{_share_row['Total']}</td>"
                            _html_tbl += "</tr>"

                            _html_tbl += "</tbody></table></div>"
                            st.markdown(_html_tbl, unsafe_allow_html=True)

                            # Download tabel
                            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                            _dl_pivot = _pivot.rename(columns={m: MONTH_NAMES[m] for m in range(1,13)})
                            _output_pj = BytesIO()
                            _dl_pivot.to_excel(_output_pj, index=False, sheet_name="Patra vs Non-Patra")
                            _output_pj.seek(0)
                            if st.session_state.get("role") == "Admin":
                                st.download_button(
                                    label="⬇️ Download Tabel",
                                    data=_output_pj,
                                    file_name=f"patra_jasa_comparison_{_pj_metric_opt.replace(' ','_')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="dl_patra_jasa_comparison")
                            else:
                                st.markdown("""
                                <div style='background:#f9f9f9;border-left:3px solid #1BA0E2;border-radius:6px;
                                padding:10px 16px;font-size:0.82em;color:#1BA0E2;display:flex;align-items:center;gap:8px;'>
                                    <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.info(f"ℹ️ Tidak ada data bulanan untuk ditampilkan pada tahun {_pj_selected_year}.")
                    else:
                        if not _date_col_pj:
                            st.info("ℹ️ Kolom tanggal (Issue Time / Check in Date) tidak ditemukan. Tabel bulanan tidak dapat ditampilkan.")
                        elif df_pj_month_table.empty:
                            st.info(f"ℹ️ Tidak ada data untuk tahun {_pj_selected_year} setelah filter yang aktif diterapkan. Coba pilih tahun lain di atas.")
                        else:
                            st.info("ℹ️ Tabel bulanan tidak dapat ditampilkan untuk kombinasi filter saat ini.")

                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                    # ── Kamus Referensi + Data Per Hotel ──
                    with st.expander(f"📋 Detail Per Hotel Patra Jasa Group — Invoice & Room Nights Bulanan{_pj_year_label}"):

                        MONTH_NAMES_KM = {
                            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
                            7:"Jul",8:"Agt",9:"Sep",10:"Okt",11:"Nov",12:"Des"
                        }

                        # Data hanya hotel Patra Jasa — pakai df_pj_month_table (sudah di-scope ke tahun terpilih)
                        # agar konsisten dengan tabel perbandingan bulanan di atas dan tidak mencampur tahun.
                        df_kamus = df_pj_month_table[df_pj_month_table["Grup Hotel"] == "Patra Jasa Group"].copy()

                        # Normalisasi nama hotel ke nama kanonik dari kamus
                        _norm_to_canonical = {h.lower().strip(): h for h in PATRA_JASA_HOTELS}
                        df_kamus["Nama Hotel Canonical"] = df_kamus["_hotel_norm"].map(_norm_to_canonical).fillna(df_kamus["Hotel Name"])

                        _km_th_h  = "padding:6px 9px;background:#e6f4fb;border:1px solid #b8d9f0;font-size:0.78em;font-weight:600;color:#0D7FCC;text-align:center;white-space:nowrap;"
                        _km_th_l  = "padding:6px 9px;background:#e6f4fb;border:1px solid #b8d9f0;font-size:0.78em;font-weight:600;color:#0D7FCC;text-align:left;white-space:nowrap;min-width:30px;"
                        _km_th_nm = "padding:6px 9px;background:#e6f4fb;border:1px solid #b8d9f0;font-size:0.78em;font-weight:600;color:#0D7FCC;text-align:left;white-space:nowrap;min-width:200px;"
                        _km_td    = "padding:5px 9px;border:1px solid #e0e0e0;text-align:right;font-size:0.8em;"
                        _km_td_l  = "padding:5px 9px;border:1px solid #e0e0e0;text-align:left;font-size:0.8em;"
                        _km_td_tot= "padding:5px 9px;border:1px solid #b8d9f0;text-align:right;font-size:0.8em;font-weight:600;background:#daeaf8;color:#0D7FCC;"

                        # ── TABEL 1: INVOICE UNIQUE PER HOTEL ──
                        st.markdown("<div style='font-size:0.85em;font-weight:600;color:#1BA0E2;margin:8px 0 6px 0;'>Invoice Unique per Hotel per Bulan</div>", unsafe_allow_html=True)

                        if _date_col_pj and "_month" in df_kamus.columns and not df_kamus.empty:
                            if "Travel Request Number" in df_kamus.columns:
                                _km_inv_agg = (
                                    df_kamus.groupby(["Nama Hotel Canonical", "_month"])["Travel Request Number"]
                                    .nunique()
                                    .reset_index(name="Nilai")
                                )
                            else:
                                _km_inv_agg = (
                                    df_kamus.groupby(["Nama Hotel Canonical", "_month"])
                                    .size().reset_index(name="Nilai")
                                )

                            _km_inv_pivot = _km_inv_agg.pivot_table(
                                index="Nama Hotel Canonical", columns="_month",
                                values="Nilai", fill_value=0
                            )
                            _km_inv_pivot.columns = [int(c) for c in _km_inv_pivot.columns]  # normalisasi tipe kolom
                            for _m in range(1, 13):
                                if _m not in _km_inv_pivot.columns:
                                    _km_inv_pivot[_m] = 0
                            _km_inv_pivot = _km_inv_pivot[[m for m in range(1, 13)]]
                            _km_inv_pivot["Total"] = _km_inv_pivot.sum(axis=1)
                            _km_inv_pivot = _km_inv_pivot.sort_values("Total", ascending=False).reset_index()

                            _html_km_inv = f"""
                            <div style='overflow-x:auto;margin-bottom:18px;'>
                            <table style='border-collapse:collapse;font-size:0.82em;min-width:100%;'>
                              <thead><tr>
                                <th style='{_km_th_l}'>No</th>
                                <th style='{_km_th_nm}'>Nama Hotel</th>
                                {"".join(f"<th style='{_km_th_h}'>{MONTH_NAMES_KM[m]}</th>" for m in range(1,13))}
                                <th style='{_km_th_h}background:#daeaf8;'>Total</th>
                              </tr></thead>
                              <tbody>
                            """
                            for _i, _row in _km_inv_pivot.iterrows():
                                _bg = "background:#fafeff;" if _i % 2 == 0 else "background:#f4faff;"
                                _html_km_inv += f"<tr style='{_bg}'>"
                                _html_km_inv += f"<td style='{_km_td_l}color:#999;'>{_i+1}</td>"
                                _html_km_inv += f"<td style='{_km_td_l}font-weight:500;'>{_row['Nama Hotel Canonical']}</td>"
                                for _m in range(1, 13):
                                    _v = int(_row[_m])
                                    _col_style = _km_td if _v > 0 else _km_td + "color:#ccc;"
                                    _html_km_inv += f"<td style='{_col_style}'>{_v if _v > 0 else '—'}</td>"
                                _html_km_inv += f"<td style='{_km_td_tot}'>{int(_row['Total']):,}</td>"
                                _html_km_inv += "</tr>"

                            # Baris total kolom
                            _html_km_inv += f"<tr style='background:#e6f4fb;font-weight:700;'>"
                            _html_km_inv += f"<td style='{_km_td_l}'></td>"
                            _html_km_inv += f"<td style='{_km_td_l}font-weight:700;color:#0D7FCC;'>Total</td>"
                            for _m in range(1, 13):
                                _col_sum = int(_km_inv_pivot[_m].sum())
                                _html_km_inv += f"<td style='{_km_th_h}'>{_col_sum:,}</td>"
                            _html_km_inv += f"<td style='{_km_th_h}background:#c8e0f5;'>{int(_km_inv_pivot['Total'].sum()):,}</td>"
                            _html_km_inv += "</tr>"

                            _html_km_inv += "</tbody></table></div>"
                            st.markdown(_html_km_inv, unsafe_allow_html=True)
                        else:
                            st.info("Data invoice per hotel tidak tersedia.")

                        # ── TABEL 2: ROOM NIGHTS PER HOTEL ──
                        st.markdown("<div style='font-size:0.85em;font-weight:600;color:#1BA0E2;margin:14px 0 6px 0;'>Room Nights per Hotel per Bulan</div>", unsafe_allow_html=True)

                        if _date_col_pj and "_month" in df_kamus.columns and "Number of Rooms Night" in df_kamus.columns and not df_kamus.empty:
                            _km_rn_agg = (
                                df_kamus.groupby(["Nama Hotel Canonical", "_month"])["Number of Rooms Night"]
                                .sum().reset_index(name="Nilai")
                            )

                            _km_rn_pivot = _km_rn_agg.pivot_table(
                                index="Nama Hotel Canonical", columns="_month",
                                values="Nilai", fill_value=0
                            )
                            _km_rn_pivot.columns = [int(c) for c in _km_rn_pivot.columns]  # normalisasi tipe kolom
                            for _m in range(1, 13):
                                if _m not in _km_rn_pivot.columns:
                                    _km_rn_pivot[_m] = 0
                            _km_rn_pivot = _km_rn_pivot[[m for m in range(1, 13)]]
                            _km_rn_pivot["Total"] = _km_rn_pivot.sum(axis=1)
                            _km_rn_pivot = _km_rn_pivot.sort_values("Total", ascending=False).reset_index()

                            _html_km_rn = f"""
                            <div style='overflow-x:auto;margin-bottom:12px;'>
                            <table style='border-collapse:collapse;font-size:0.82em;min-width:100%;'>
                              <thead><tr>
                                <th style='{_km_th_l}'>No</th>
                                <th style='{_km_th_nm}'>Nama Hotel</th>
                                {"".join(f"<th style='{_km_th_h}'>{MONTH_NAMES_KM[m]}</th>" for m in range(1,13))}
                                <th style='{_km_th_h}background:#daeaf8;'>Total</th>
                              </tr></thead>
                              <tbody>
                            """
                            for _i, _row in _km_rn_pivot.iterrows():
                                _bg = "background:#fafeff;" if _i % 2 == 0 else "background:#f4faff;"
                                _html_km_rn += f"<tr style='{_bg}'>"
                                _html_km_rn += f"<td style='{_km_td_l}color:#999;'>{_i+1}</td>"
                                _html_km_rn += f"<td style='{_km_td_l}font-weight:500;'>{_row['Nama Hotel Canonical']}</td>"
                                for _m in range(1, 13):
                                    _v = _row[_m]
                                    _v_fmt = f"{_v:,.0f}" if _v > 0 else "—"
                                    _col_style = _km_td if _v > 0 else _km_td + "color:#ccc;"
                                    _html_km_rn += f"<td style='{_col_style}'>{_v_fmt}</td>"
                                _html_km_rn += f"<td style='{_km_td_tot}'>{_row['Total']:,.0f}</td>"
                                _html_km_rn += "</tr>"

                            # Baris total kolom
                            _html_km_rn += f"<tr style='background:#e6f4fb;font-weight:700;'>"
                            _html_km_rn += f"<td style='{_km_td_l}'></td>"
                            _html_km_rn += f"<td style='{_km_td_l}font-weight:700;color:#0D7FCC;'>Total</td>"
                            for _m in range(1, 13):
                                _col_sum = _km_rn_pivot[_m].sum()
                                _html_km_rn += f"<td style='{_km_th_h}'>{_col_sum:,.0f}</td>"
                            _html_km_rn += f"<td style='{_km_th_h}background:#c8e0f5;'>{_km_rn_pivot['Total'].sum():,.0f}</td>"
                            _html_km_rn += "</tr>"

                            _html_km_rn += "</tbody></table></div>"
                            st.markdown(_html_km_rn, unsafe_allow_html=True)
                        else:
                            st.info("Data room nights per hotel tidak tersedia (kolom 'Number of Rooms Night' tidak ditemukan).")

    # ======================================
    # DISCLAIMER + FOOTER
    # ======================================
    st.markdown("""
    <div style="background:white;padding:20px;border-radius:4px;border-left:4px solid #1BA0E2;font-size:0.9em;">
        <b>Disclaimer &amp; Compliance Notice</b><br><br>
        Aplikasi ini disediakan untuk tujuan analisis internal. Output yang dihasilkan tidak bersifat final,
        tidak mengikat, dan harus melalui proses validasi serta persetujuan sesuai kebijakan perusahaan yang berlaku.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='divider' style='margin-top:50px;'></div>
    <div style='text-align:center;padding:25px;color:#888888;font-size:0.85em;'>
        © 2025 Dikembangkan oleh 
        <a href="https://www.linkedin.com/in/rifyalt/" target="_blank" style='color:#1BA0E2;text-decoration:none;font-weight:500;'>
            Rifyal Tumber
        </a> · MTRAX Travel Analytics
    </div>
    """, unsafe_allow_html=True)


# ===============================
# ROUTING — WAJIB PALING BAWAH
# ── DIMODIFIKASI: tambah pengecekan pending_2fa + pemulihan sesi dari cookie ──
# ===============================
_try_restore_session_from_cookie()  # pulihkan login kalau cookie sesi masih valid (mis. setelah reload halaman)

if not st.session_state.get("authenticated"):

    if st.session_state.get("pending_2fa"):
        # ── Password sudah benar, tunggu OTP ─────────────────────────
        username = st.session_state.pending_user
        totp_secrets = _load_totp_secrets()

        # Semua user berbagi secret admin — enrolled jika [totp][admin] ada di secrets.toml
        enrolled_via_toml = False
        try:
            enrolled_via_toml = bool(st.secrets["totp"]["admin"])
        except Exception:
            pass

        # ── Hanya percaya secrets.toml sebagai bukti enrollment permanen.
        # session_state.totp_enrolled bersifat volatile (hilang saat restart),
        # sehingga tidak boleh dijadikan satu-satunya penentu.
        # User dianggap enrolled HANYA jika secret sudah ada di secrets.toml.
        already_enrolled = enrolled_via_toml

        if already_enrolled:
            twofa_verify_page(username)   # user lama → langsung input OTP
        else:
            # Pastikan totp_enrolled di-reset agar tidak menyebabkan
            # inkonsistensi pada sesi berikutnya
            st.session_state.totp_enrolled[username] = False
            twofa_setup_page(username)    # user baru → setup QR code dulu

    else:
        login_page()   # belum login sama sekali

else:
    check_session_timeout()
    main_app()
