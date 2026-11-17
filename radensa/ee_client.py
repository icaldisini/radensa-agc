"""
Inisialisasi Earth Engine via service account - tanpa ee.Authenticate() interaktif,
supaya jalan di GitHub Actions maupun lokal dengan kode yang sama persis.

Satu env var, EE_SERVICE_ACCOUNT_JSON, menerima DUA bentuk nilai sekaligus:
  - Path lokal ke file key .json (dipakai saat development di komputer sendiri)
  - Isi JSON mentah dari key itu (dipakai di GitHub Actions, dari Secrets -
    CI tidak punya file lokal untuk direferensikan lewat path)
Bentuknya dideteksi otomatis: kalau nilainya diawali '{', dianggap JSON mentah
dan ditulis ke file sementara dulu; selain itu dianggap path.

project_id untuk ee.Initialize() diambil dari field "project_id" di dalam key
itu sendiri, jadi tidak perlu di-hardcode terpisah di mana pun.
"""
from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Sama seperti radensa/supabase_client.py - dimuat sekali di sini supaya
# semua pemanggil init_ee() otomatis kebagian .env.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


@lru_cache(maxsize=1)
def init_ee() -> str:
    """Inisialisasi Earth Engine. Mengembalikan project_id yang dipakai. Idempoten."""
    raw = os.environ.get("EE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise RuntimeError(
            "EE_SERVICE_ACCOUNT_JSON belum diset. Isi dengan path ke file key "
            "service account (lokal) atau isi JSON key itu sendiri (CI)."
        )

    if raw.startswith("{"):
        key_path = Path(tempfile.gettempdir()) / "radensa_ee_key.json"
        key_path.write_text(raw, encoding="utf-8")
    else:
        key_path = Path(raw)
        if not key_path.exists():
            raise FileNotFoundError(f"File key Earth Engine tidak ditemukan: {key_path}")

    info = json.loads(key_path.read_text(encoding="utf-8"))
    sa_email = info["client_email"]
    project_id = info["project_id"]

    import ee

    credentials = ee.ServiceAccountCredentials(sa_email, str(key_path))
    ee.Initialize(credentials=credentials, project=project_id)
    return project_id
