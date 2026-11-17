"""
Fase 2 - Scraper kasus DBD mingguan dari surveilans-dinkes.jakarta.go.id.

Endpoint terverifikasi (diuji manual 17 Nov 2026, lihat memory agen):
    POST https://surveilans-dinkes.jakarta.go.id/sarsbaru/rs_rekap.php
    prov=1 & kab=<1..6> & kec=0 & rs=0 & penyakit=6 (DBD) & golum=0 & jk=0
    & sp=0 & jdata=2 (mingguan) & mpar1=<minggu_awal> & mpar2=<minggu_akhir>
    & mpar3=<tahun> & tbl_proses=P+R+O+S+E+S

Satu POST per (kabupaten, tahun) mengembalikan SEMUA kecamatan di kabupaten
itu sekaligus untuk seluruh rentang minggu yang diminta -> 6 request per
tahun sudah mencakup 44 kecamatan.

PENTING - konvensi minggu:
  Nomor minggu yang dikembalikan situs ini adalah "Minggu Kalender
  Epidemiologi" MMWR (Minggu-Sabtu) Dinkes, BUKAN ISO week. Nomor minggu
  dari situs dipakai APA ADANYA sebagai (tahun, minggu) epi -- lihat
  radensa/epiweek.py. Jangan konversi nomor ini lewat isocalendar().

Strategi revisi data:
  Dinkes merevisi angka ke belakang saat laporan RS yang telat baru masuk.
  Karena itu pipeline mingguan TIDAK hanya menarik minggu terbaru, tapi
  melakukan rolling re-scrape atas beberapa minggu terakhir (default:
  ROLLING_WINDOW_WEEKS) dan meng-upsert semuanya -- revisi otomatis
  tertangkap tanpa logika khusus.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radensa.epiweek import add_epi_weeks, iter_epi_weeks, last_complete_epi_week
from radensa.supabase_client import get_client

ROOT = Path(__file__).resolve().parent.parent
REFDATA = ROOT / "refdata"
MASTER_CSV = REFDATA / "master_kecamatan.csv"

BASE_URL = "https://surveilans-dinkes.jakarta.go.id/sarsbaru/rs_rekap.php"
PENYAKIT_DBD = 6
REQUEST_TIMEOUT = 60
REQUEST_DELAY_S = 0.6  # sopan terhadap server Dinkes antar-request

# Minggu di dalam jendela ini masih dianggap "bisa direvisi" -> is_final=False.
# Disamakan dengan jendela rolling re-scrape default supaya konsisten:
# selama sebuah minggu masih di-scrape ulang tiap minggu, dia belum final.
ROLLING_WINDOW_WEEKS = 8


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s.upper().strip())
    return s


@dataclass(frozen=True)
class KecamatanRef:
    kode_kec: str
    nama_dinkes_norm: str
    kab_kode: int


def load_master() -> list[KecamatanRef]:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"{MASTER_CSV} tidak ada. Jalankan scripts/build_master.py dulu (Fase 0)."
        )
    out = []
    with MASTER_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(
                KecamatanRef(
                    kode_kec=row["kode_kec"],
                    nama_dinkes_norm=_norm(row["nama_dinkes"]),
                    kab_kode=int(row["kab_kode"]),
                )
            )
    if len(out) != 44:
        raise RuntimeError(f"master_kecamatan.csv harus 44 baris, dapat {len(out)}")
    return out


def _post_rekap(session: requests.Session, kab_kode: int, tahun: int,
                 minggu_awal: int, minggu_akhir: int) -> str:
    payload = {
        "prov": 1, "kab": kab_kode, "kec": 0, "rs": 0,
        "penyakit": PENYAKIT_DBD, "golum": 0, "jk": 0, "sp": 0,
        "jdata": 2,  # 2 = Mingguan
        "mpar1": minggu_awal, "mpar2": minggu_akhir, "mpar3": tahun,
        # Field form lain diisi default supaya PHP tidak keberatan undefined index.
        "hpar1": 1, "hpar2": tahun,
        "bpar1": 1, "bpar2": 1, "bpar3": tahun,
        "tpar1": tahun, "tpar2": tahun,
        "tbl_proses": "P R O S E S",
    }
    resp = session.post(BASE_URL, data=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def _parse_rekap_table(html: str) -> tuple[list[int], list[tuple[str, dict[int, int]]]]:
    """
    Parse tabel hasil "No | Nama | <minggu1> | <minggu2> | ...".
    Mengembalikan (daftar_nomor_minggu, [(nama_kecamatan, {minggu: kasus}), ...]).
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        raise RuntimeError("Tidak ada <table> pada respons - kemungkinan form berubah")

    target = None
    for t in tables:
        header_cells = [c.get_text(strip=True) for c in t.find_all(["th", "td"])[:2]]
        if header_cells[:1] == ["No"]:
            target = t
            break
    if target is None:
        target = tables[-1]

    rows = target.find_all("tr")
    if len(rows) < 1:
        raise RuntimeError("Tabel hasil kosong")

    header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
    week_numbers = [int(x) for x in header_cells[2:] if x.strip().isdigit()]
    if not week_numbers:
        raise RuntimeError(f"Gagal membaca header minggu dari tabel: {header_cells}")

    out = []
    for r in rows[1:]:
        cells = [c.get_text(strip=True) for c in r.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        nama = cells[1]
        counts_raw = cells[2:2 + len(week_numbers)]
        counts = {}
        for wk, val in zip(week_numbers, counts_raw):
            val = val.replace("\xa0", "").strip()
            counts[wk] = int(val) if val.isdigit() else 0
        out.append((nama, counts))
    return week_numbers, out


def scrape_kabupaten(session: requests.Session, kab_kode: int, tahun: int,
                      minggu_awal: int, minggu_akhir: int,
                      master_by_kab: dict[int, list[KecamatanRef]]
                      ) -> list[tuple[str, int, int, int]]:
    """Scrape satu kabupaten untuk rentang minggu dalam satu tahun epi.

    Return list of (kode_kec, tahun, minggu, kasus).
    """
    html = _post_rekap(session, kab_kode, tahun, minggu_awal, minggu_akhir)
    weeks, rows = _parse_rekap_table(html)

    name_to_kode = {ref.nama_dinkes_norm: ref.kode_kec for ref in master_by_kab[kab_kode]}
    expected = set(name_to_kode)
    seen = set()

    out = []
    for nama_raw, counts in rows:
        key = _norm(nama_raw)
        kode_kec = name_to_kode.get(key)
        if kode_kec is None:
            raise RuntimeError(
                f"Kecamatan '{nama_raw}' (kab {kab_kode}) tidak ada di master_kecamatan.csv. "
                f"Situs Dinkes mungkin berubah nama -- perbarui scripts/build_master.py."
            )
        seen.add(key)
        for wk, kasus in counts.items():
            if minggu_awal <= wk <= minggu_akhir:
                out.append((kode_kec, tahun, wk, kasus))

    missing = expected - seen
    if missing:
        raise RuntimeError(
            f"Kabupaten {kab_kode} tahun {tahun}: kecamatan tidak muncul di respons: {missing}"
        )
    return out


def fetch_dbd_weekly(targets: Iterable[tuple[int, int]],
                      master: list[KecamatanRef] | None = None,
                      ) -> list[dict]:
    """
    Ambil kasus DBD untuk daftar (tahun, minggu) epi yang diminta.

    Mengelompokkan per tahun (form Dinkes tidak bisa lintas tahun dalam satu
    request) lalu satu request per kabupaten mencakup seluruh rentang minggu
    kontigu tahun itu.
    """
    master = master or load_master()
    master_by_kab: dict[int, list[KecamatanRef]] = defaultdict(list)
    for ref in master:
        master_by_kab[ref.kab_kode].append(ref)

    by_tahun: dict[int, set[int]] = defaultdict(set)
    for tahun, minggu in targets:
        by_tahun[tahun].add(minggu)

    records: list[dict] = []
    session = requests.Session()
    final_cutoff = add_epi_weeks(*last_complete_epi_week(), -ROLLING_WINDOW_WEEKS)

    for tahun in sorted(by_tahun):
        minggu_set = by_tahun[tahun]
        m_awal, m_akhir = min(minggu_set), max(minggu_set)
        for kab_kode in sorted(master_by_kab):
            rows = scrape_kabupaten(session, kab_kode, tahun, m_awal, m_akhir, master_by_kab)
            for kode_kec, thn, wk, kasus in rows:
                if wk not in minggu_set:
                    continue
                is_final = (thn, wk) <= final_cutoff
                records.append({
                    "kode_kec": kode_kec, "tahun": thn, "minggu": wk,
                    "kasus": kasus, "is_final": is_final,
                })
            time.sleep(REQUEST_DELAY_S)

    found = {(r["kode_kec"], r["tahun"], r["minggu"]) for r in records}
    expected_count = len(master) * len(targets if isinstance(targets, (list, set)) else list(targets))
    if len(found) != expected_count:
        raise RuntimeError(
            f"Hasil scrape tidak lengkap: {len(found)}/{expected_count} kombinasi "
            f"kecamatan x minggu."
        )
    return records


def upsert_supabase(records: list[dict]) -> None:
    client = get_client()
    if client is None:
        print("[ingest_dbd] SUPABASE_* env var belum diset -> lewati upsert, mode lokal saja.")
        return
    # Batch upsert supaya satu kali approval/roundtrip untuk seluruh baris.
    BATCH = 500
    for i in range(0, len(records), BATCH):
        chunk = records[i:i + BATCH]
        client.table("dbd_weekly").upsert(
            chunk, on_conflict="kode_kec,tahun,minggu"
        ).execute()
    print(f"[ingest_dbd] Upsert ke Supabase selesai: {len(records)} baris.")


def write_local_csv(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["kode_kec", "tahun", "minggu", "kasus", "is_final"])
        w.writeheader()
        for r in sorted(records, key=lambda r: (r["tahun"], r["minggu"], r["kode_kec"])):
            w.writerow(r)
    print(f"[ingest_dbd] Ditulis lokal: {out_path} ({len(records)} baris)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weeks-back", type=int, default=ROLLING_WINDOW_WEEKS,
                    help="Rolling re-scrape N minggu terakhir yang sudah lengkap (default: 8)")
    p.add_argument("--tahun", type=int, help="Backfill manual: tahun epi")
    p.add_argument("--minggu-awal", type=int, help="Backfill manual: minggu awal")
    p.add_argument("--minggu-akhir", type=int, help="Backfill manual: minggu akhir")
    p.add_argument("--out", type=Path, default=ROOT / "data_raw" / "dbd_weekly_latest.csv")
    p.add_argument("--no-upsert", action="store_true", help="Lewati upsert Supabase, lokal saja")
    args = p.parse_args()

    t0 = time.time()
    if args.tahun and args.minggu_awal and args.minggu_akhir:
        targets = list(iter_epi_weeks((args.tahun, args.minggu_awal), (args.tahun, args.minggu_akhir)))
    else:
        end = last_complete_epi_week()
        start = add_epi_weeks(*end, -(args.weeks_back - 1))
        targets = list(iter_epi_weeks(start, end))

    print(f"[ingest_dbd] Target: {targets[0]} s/d {targets[-1]} ({len(targets)} minggu epi)")
    records = fetch_dbd_weekly(targets)
    write_local_csv(records, args.out)

    if not args.no_upsert:
        upsert_supabase(records)

    dt_ms = int((time.time() - t0) * 1000)
    n_final = sum(1 for r in records if r["is_final"])
    print(f"[ingest_dbd] Selesai dalam {dt_ms}ms. "
          f"{len(records)} baris ({n_final} final, {len(records) - n_final} provisional).")


if __name__ == "__main__":
    main()
