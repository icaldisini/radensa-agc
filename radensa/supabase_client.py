"""
Klien Supabase bersama untuk seluruh script ingest/forecast Radensa.

Dibuat "opsional secara sengaja": jika env var belum diset (misal saat
mengembangkan Fase 2 lokal sebelum project Supabase siap), get_client()
mengembalikan None dan pemanggil wajib menangani itu (biasanya: tulis ke
file lokal saja, lewati langkah upsert).

Env var:
  SUPABASE_URL               - URL project (boleh dipakai bersama anon key)
  SUPABASE_SERVICE_ROLE_KEY  - WAJIB untuk menulis (insert/upsert), karena
                                RLS hanya mengizinkan "select" publik.
                                JANGAN PERNAH commit nilai ini ke repo.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def get_client() -> Optional["Client"]:  # noqa: F821
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    from supabase import create_client

    return create_client(url, key)


def require_client():
    """Sama seperti get_client(), tapi raise error jelas bila belum diset."""
    client = get_client()
    if client is None:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY belum diset. "
            "Set env var ini (lihat .env.example) atau jalankan dengan "
            "--no-upsert untuk mode lokal-saja."
        )
    return client
