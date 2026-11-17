"""
Fase 5 (bagian 1) - Muat panel kecamatan-epi_week dari Supabase, siap dipakai
radensa/features.py.

Menggabungkan dbd_weekly + iklim_weekly jadi satu panel panjang (long format)
per (kecamatan, tahun, minggu), mengisi jenis='klimatologi' untuk sel iklim
yang belum ada (dipakai forecast.py, bukan training), dan memvalidasi tidak
ada data dummy/hardcoded yang lolos diam-diam -- sesuai prinsip yang sama
seperti notebook lama (raise error, bukan isi nilai palsu).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radensa.supabase_client import require_client

PAGE = 1000


def _paginated_select(client, table: str, columns: str, **filters) -> list[dict]:
    """Select dengan pagination eksplisit -- lihat catatan yang sama di
    radensa/ingest_iklim.py tentang limit default PostgREST ~1000 baris."""
    out: list[dict] = []
    offset = 0
    while True:
        q = client.table(table).select(columns)
        for key, val in filters.items():
            if key.endswith("__gte"):
                q = q.gte(key[:-5], val)
            elif key.endswith("__lte"):
                q = q.lte(key[:-5], val)
            else:
                q = q.eq(key, val)
        r = q.range(offset, offset + PAGE - 1).execute()
        out.extend(r.data)
        if len(r.data) < PAGE:
            break
        offset += PAGE
    return out


def load_master_kecamatan(client) -> pd.DataFrame:
    rows = _paginated_select(client, "kecamatan", "kode_kec,nama,kabupaten")
    df = pd.DataFrame(rows)
    if len(df) != 44:
        raise RuntimeError(f"Tabel kecamatan harus 44 baris, dapat {len(df)}")
    return df


def load_neighbors(client, tipe: str = "feature") -> dict[str, list[str]]:
    """
    tipe='feature' untuk fitur model, 'queen' untuk statistik spasial (Moran's I).

    Dikembalikan ber-key NAMA kecamatan (bukan kode_kec!) -- WAJIB konsisten
    dengan kolom 'kecamatan' di panel (radensa/features.py: build_features()
    melakukan lookup tetangga lewat df['kecamatan'], yang isinya nama, bukan
    kode). Bug nyata yang pernah terjadi: versi awal fungsi ini mengembalikan
    dict ber-key kode_kec ("4.32") sementara build_features mencari pakai nama
    ("KEBAYORAN LAMA") -- mismatch itu lolos tanpa error saat training karena
    get_fold_xy() memanggil .fillna(0) pada fitur spasial yang jadi NaN semua,
    sehingga model terlatih dengan fitur spasial = 0 konstan tanpa ada yang
    sadar. Baru ketahuan di forecast_step() yang tidak punya fallback diam-diam.
    """
    master = load_master_kecamatan(client)
    kode_to_nama = dict(zip(master["kode_kec"], master["nama"]))
    rows = _paginated_select(client, "kecamatan_tetangga", "kode_kec,tetangga_kode", tipe=tipe)
    out: dict[str, list[str]] = {}
    for r in rows:
        nama = kode_to_nama[r["kode_kec"]]
        tetangga_nama = kode_to_nama[r["tetangga_kode"]]
        out.setdefault(nama, []).append(tetangga_nama)
    return out


def load_panel(tahun_min: int | None = None, tahun_max: int | None = None) -> pd.DataFrame:
    """
    Panel gabungan dbd_weekly INNER JOIN iklim_weekly (hanya epi week yang
    PUNYA keduanya), di-key kode_kec bukan nama -- nama/kabupaten ditempel
    balik dari tabel kecamatan di akhir supaya kompatibel dengan radensa/
    features.py yang memakai kolom 'kecamatan' sebagai key groupby.

    Baris dengan is_final=False (masih dalam jendela revisi DBD) TETAP
    disertakan -- pemanggil (training/CV) yang memutuskan apakah mau
    memfilternya, karena untuk histori lama semua baris sudah final.
    """
    client = require_client()
    master = load_master_kecamatan(client)
    kode_to_nama = dict(zip(master["kode_kec"], master["nama"]))
    kode_to_kab = dict(zip(master["kode_kec"], master["kabupaten"]))

    filters = {}
    if tahun_min is not None:
        filters["tahun__gte"] = tahun_min
    if tahun_max is not None:
        filters["tahun__lte"] = tahun_max

    dbd_rows = _paginated_select(
        client, "dbd_weekly", "kode_kec,tahun,minggu,kasus,is_final", **filters
    )
    iklim_rows = _paginated_select(
        client, "iklim_weekly",
        "kode_kec,tahun,minggu,curah_hujan_kumulatif,curah_hujan_max_harian,suhu_rata,kelembapan_rata,jenis",
        **filters,
    )
    if not dbd_rows:
        raise RuntimeError(
            "Tabel dbd_weekly kosong untuk rentang tahun ini -- jalankan "
            "radensa/ingest_dbd.py backfill dulu."
        )
    if not iklim_rows:
        raise RuntimeError(
            "Tabel iklim_weekly kosong untuk rentang tahun ini -- jalankan "
            "radensa/ingest_iklim.py backfill dulu."
        )

    df_dbd = pd.DataFrame(dbd_rows).rename(columns={"kasus": "kasus_DBD"})
    df_iklim = pd.DataFrame(iklim_rows)

    panel = df_dbd.merge(df_iklim, on=["kode_kec", "tahun", "minggu"], how="inner")

    missing_dbd = set(zip(df_dbd.kode_kec, df_dbd.tahun, df_dbd.minggu)) - \
        set(zip(panel.kode_kec, panel.tahun, panel.minggu))
    if missing_dbd:
        raise RuntimeError(
            f"{len(missing_dbd)} baris DBD tidak punya pasangan iklim (epi week "
            f"belum diekstrak GEE): contoh {sorted(missing_dbd)[:5]}"
        )

    panel["kecamatan"] = panel["kode_kec"].map(kode_to_nama)
    panel["kabupaten"] = panel["kode_kec"].map(kode_to_kab)
    panel = panel.sort_values(["kecamatan", "tahun", "minggu"]).reset_index(drop=True)
    return panel


def load_iklim_raw() -> pd.DataFrame:
    """Seluruh iklim_weekly APA ADANYA (tidak di-join DBD) -- basis klimatologi
    fallback untuk minggu forecast yang belum punya observasi GEE asli."""
    client = require_client()
    rows = _paginated_select(
        client, "iklim_weekly",
        "kode_kec,tahun,minggu,curah_hujan_kumulatif,curah_hujan_max_harian,suhu_rata,kelembapan_rata",
    )
    if not rows:
        raise RuntimeError("Tabel iklim_weekly kosong.")
    return pd.DataFrame(rows)


def load_iklim_for_weeks(targets: list[tuple[int, int]]) -> pd.DataFrame:
    """Baris iklim_weekly APA ADANYA untuk (tahun,minggu) spesifik (observasi
    GEE asli yang kebetulan sudah terbit untuk minggu forecast terdekat)."""
    client = require_client()
    out = []
    for tahun, minggu in targets:
        rows = _paginated_select(
            client, "iklim_weekly",
            "kode_kec,tahun,minggu,curah_hujan_kumulatif,curah_hujan_max_harian,suhu_rata,kelembapan_rata",
            tahun=tahun, minggu=minggu,
        )
        out.extend(rows)
    return pd.DataFrame(out) if out else pd.DataFrame(
        columns=["kode_kec", "tahun", "minggu", "curah_hujan_kumulatif",
                 "curah_hujan_max_harian", "suhu_rata", "kelembapan_rata"])
