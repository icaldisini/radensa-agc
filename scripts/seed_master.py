"""
Fase 1 (lanjutan) - Seed tabel referensi `kecamatan` & `kecamatan_tetangga` di
Supabase dari refdata/ yang sudah dibekukan di Fase 0 (scripts/build_master.py).

Jalankan sekali setelah schema (sql/001_schema.sql) dieksekusi, dan setiap
kali refdata/ di-build ulang (mis. kalau geometri sumber berubah).

Butuh SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY di .env (lihat .env.example).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
REFDATA = ROOT / "refdata"


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ROOT / ".env")


def read_master() -> list[dict]:
    path = REFDATA / "master_kecamatan.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 44:
        raise RuntimeError(f"master_kecamatan.csv harus 44 baris, dapat {len(rows)}")
    out = []
    for r in rows:
        out.append({
            "kode_kec": r["kode_kec"],
            "slug": r["slug"],
            "nama": r["nama_shp"],
            "nama_dinkes": r["nama_dinkes"],
            "kabupaten": r["kabupaten"],
            "kab_kode": int(r["kab_kode"]),
            "kec_kode_dinkes": int(r["kec_kode_dinkes"]),
            "lon": float(r["lon"]),
            "lat": float(r["lat"]),
        })
    return out


def read_neighbors(master_rows: list[dict]) -> list[dict]:
    """Ubah neighbors.json (ber-key NAMA) jadi baris (kode_kec, tetangga_kode, tipe)."""
    payload = json.loads((REFDATA / "neighbors.json").read_text(encoding="utf-8"))
    nama_to_kode = {r["nama"]: r["kode_kec"] for r in master_rows}

    out = []
    for tipe in ("queen", "feature"):
        for nama, tetangga_list in payload[tipe].items():
            kode = nama_to_kode[nama]
            for t_nama in tetangga_list:
                out.append({
                    "kode_kec": kode,
                    "tetangga_kode": nama_to_kode[t_nama],
                    "tipe": tipe,
                })
    return out


def main() -> None:
    load_env()
    import os
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY belum diset di .env")
    client = create_client(url, key)

    print("[1/3] Baca refdata/")
    kec_rows = read_master()
    nbr_rows = read_neighbors(kec_rows)
    print(f"  kecamatan        : {len(kec_rows)} baris")
    print(f"  kecamatan_tetangga: {len(nbr_rows)} baris "
          f"({sum(1 for r in nbr_rows if r['tipe'] == 'queen')} queen, "
          f"{sum(1 for r in nbr_rows if r['tipe'] == 'feature')} feature)")

    print("[2/3] Upsert kecamatan")
    client.table("kecamatan").upsert(kec_rows, on_conflict="kode_kec").execute()
    print("  OK")

    print("[3/3] Upsert kecamatan_tetangga")
    # Hapus lama dulu supaya relasi yang sudah tidak berlaku ikut terbuang
    # (refdata bisa berubah kalau geometri sumber di-build ulang).
    client.table("kecamatan_tetangga").delete().neq("kode_kec", "").execute()
    BATCH = 500
    for i in range(0, len(nbr_rows), BATCH):
        chunk = nbr_rows[i:i + BATCH]
        client.table("kecamatan_tetangga").upsert(
            chunk, on_conflict="kode_kec,tetangga_kode,tipe"
        ).execute()
    print("  OK")

    # Verifikasi akhir
    r1 = client.table("kecamatan").select("kode_kec", count="exact").execute()
    r2 = client.table("kecamatan_tetangga").select("kode_kec", count="exact").execute()
    print(f"\nVerifikasi: kecamatan={r1.count} baris, kecamatan_tetangga={r2.count} baris")
    if r1.count != 44:
        raise RuntimeError(f"Tabel kecamatan harus 44 baris, dapat {r1.count}")
    if r2.count != len(nbr_rows):
        raise RuntimeError(f"Tabel kecamatan_tetangga harus {len(nbr_rows)} baris, dapat {r2.count}")
    print("SEED SELESAI.")


if __name__ == "__main__":
    main()
