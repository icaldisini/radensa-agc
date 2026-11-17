"""
Fase 3 - Ekstraksi iklim mingguan (CHIRPS + ERA5-Land) dari Google Earth Engine,
per kecamatan, pada batas EPI WEEK (bukan ISO week).

KEPUTUSAN 17 Nov 2026: seluruh histori 2016-2026 diekstrak ulang memakai
batas minggu epidemiologi MMWR (Minggu-Sabtu) milik Dinkes, bukan ISO week
(Senin-Minggu) seperti notebook lama (Fitur_Iklim_2026.ipynb). Lihat
radensa/epiweek.py dan memory "radensa-epi-week-mismatch" untuk alasannya:
tahun 2026 offset 6 hari antara dua sistem minggu itu, sehingga fitur iklim
dan target kasus tahun 2026 di notebook lama tidak benar-benar sejajar.

Sumber data (fallback berurutan, sama seperti notebook lama):
  1. CHIRPS Daily (UCSB-CHG/CHIRPS/DAILY) - curah hujan
  2. CHIRPS v3 (UCSB-CHC/CHIRPS/V3/DAILY_SAT) - cadangan kalau (1) tidak lengkap
  3. ERA5-Land total_precipitation_sum - cadangan terakhir kalau keduanya gagal
  ERA5-Land DAILY_AGGR dipakai untuk suhu & kelembapan (Magnus-Tetens),
  SELALU wajib lengkap 7/7 hari; kalau tidak, minggu itu dilewati (akan
  tertangkap lagi di rolling run minggu berikutnya setelah data terbit).

Geometri 44 kecamatan dibaca dari refdata/kecamatan_44.geojson (dibekukan di
Fase 0) - TIDAK mengunduh shapefile 234MB dari Google Drive seperti dulu.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import unicodedata
import re
import queue
import threading
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radensa.ee_client import init_ee
from radensa.epiweek import add_epi_weeks, epi_week_range, iter_epi_weeks, last_complete_epi_week
from radensa.supabase_client import get_client

ROOT = Path(__file__).resolve().parent.parent
REFDATA = ROOT / "refdata"

CLIMATE_COLS = [
    "curah_hujan_kumulatif", "curah_hujan_max_harian", "suhu_rata", "kelembapan_rata",
]
REQUEST_DELAY_S = 0.3
# .getInfo() ke GEE (httplib2 di baliknya) kadang hang TANPA batas waktu kalau
# server lambat/jaringan terputus -- terbukti nyata: backfill pernah diam 1 jam
# penuh di satu minggu tanpa error maupun progres sama sekali. Timeout + 1 retry
# di sini mencegah itu terulang; executor dibuang (bukan di-reuse) supaya thread
# yang macet tidak menumpuk selamanya di background.
WEEK_TIMEOUT_S = 90
WEEK_MAX_RETRY = 1


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.upper().strip())


def load_master() -> dict[str, str]:
    """nama_shp (ter-normalisasi) -> kode_kec."""
    path = REFDATA / "master_kecamatan.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {_norm(r["nama_shp"]): r["kode_kec"] for r in rows}
    if len(out) != 44:
        raise RuntimeError(f"master_kecamatan.csv harus 44 baris, dapat {len(out)}")
    return out


def load_fc_jakarta():
    """Bangun ee.FeatureCollection dari refdata/kecamatan_44.geojson (44 kecamatan)."""
    import ee

    geo = json.loads((REFDATA / "kecamatan_44.geojson").read_text(encoding="utf-8"))
    if len(geo["features"]) != 44:
        raise RuntimeError(f"kecamatan_44.geojson harus 44 fitur, dapat {len(geo['features'])}")

    features = []
    for feat in geo["features"]:
        geom = ee.Geometry(feat["geometry"])
        props = {"nama_shp": feat["properties"]["nama_shp"]}
        features.append(ee.Feature(geom, props))
    return ee.FeatureCollection(features)


def _process_era5_daily(img):
    """Konversi suhu K->C dan hitung RH (%) via formula Magnus-Tetens."""
    t_c = img.select("temperature_2m").subtract(273.15)
    td_c = img.select("dewpoint_temperature_2m").subtract(273.15)
    es = t_c.multiply(17.67).divide(t_c.add(243.5)).exp().multiply(6.112)
    e = td_c.multiply(17.67).divide(td_c.add(243.5)).exp().multiply(6.112)
    rh = e.divide(es).multiply(100.0).clamp(0.0, 100.0)
    return img.addBands(t_c.rename("suhu_c")).addBands(rh.rename("kelembapan_pct"))


def fetch_one_epi_week(fc_jakarta, year: int, week: int, nama_to_kode: dict) -> list[dict] | None:
    """
    Ekstrak satu epi week untuk 44 kecamatan. Return None (bukan raise) kalau
    data satelit untuk minggu itu belum terbit - dibedakan dari error data
    rusak/tidak valid, yang tetap me-raise seperti notebook lama.
    """
    import ee

    start, _ = epi_week_range(year, week)
    start_str = start.isoformat()
    # filterDate GEE: akhir eksklusif -> Minggu pertama epi week berikutnya,
    # selalu persis start+7 hari (epi week selalu 7 hari penuh, lihat epiweek.py).
    end_str = (start + timedelta(days=7)).isoformat()

    era5_col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_str, end_str)
    chirps_v2 = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                 .filterDate(start_str, end_str).select("precipitation"))
    chirps_v3 = (ee.ImageCollection("UCSB-CHC/CHIRPS/V3/DAILY_SAT")
                 .filterDate(start_str, end_str).select("precipitation"))

    n_era5 = era5_col.size().getInfo()
    if n_era5 != 7:
        print(f"  lewati {year}-W{week:02d}: ERA5-Land baru {n_era5}/7 hari "
              f"({start_str} s/d {end_str}) - belum terbit, coba lagi minggu depan.")
        return None

    n_v2 = chirps_v2.size().getInfo()
    n_v3 = chirps_v3.size().getInfo()
    if n_v2 == 7:
        precip_col, sumber_hujan = chirps_v2, "chirps_v2"
    elif n_v3 == 7:
        precip_col, sumber_hujan = chirps_v3, "chirps_v3"
    else:
        precip_col = era5_col.select(["total_precipitation_sum"], ["precipitation"]) \
            .map(lambda img: img.multiply(1000.0))
        sumber_hujan = "era5"
        print(f"  {year}-W{week:02d}: CHIRPS tidak lengkap (v2={n_v2}, v3={n_v3}), pakai ERA5.")

    chirps_sum = precip_col.sum().rename("curah_hujan_kumulatif")
    chirps_max = precip_col.max().rename("curah_hujan_max_harian")
    era5_processed = era5_col.map(_process_era5_daily).select(["suhu_c", "kelembapan_pct"]).mean()
    suhu = era5_processed.select("suhu_c").rename("suhu_rata")
    kelembapan = era5_processed.select("kelembapan_pct").rename("kelembapan_rata")

    weekly_img = ee.Image.cat([chirps_sum, chirps_max, suhu, kelembapan])
    weekly_focal = weekly_img.focalMean(radius=40000, units="meters", kernelType="circle")
    weekly_filled = weekly_img.unmask(weekly_focal)

    reduced = weekly_filled.reduceRegions(
        collection=fc_jakarta, reducer=ee.Reducer.mean(), scale=1000, tileScale=4
    )
    res = reduced.getInfo()
    feats = res.get("features", [])
    if len(feats) != 44:
        raise RuntimeError(
            f"{year}-W{week:02d}: reduceRegions hasil {len(feats)} fitur, harus 44."
        )

    records = []
    for feat in feats:
        props = feat["properties"]
        nama = _norm(props.get("nama_shp", ""))
        kode_kec = nama_to_kode.get(nama)
        if kode_kec is None:
            raise RuntimeError(f"nama_shp '{props.get('nama_shp')}' tidak ada di master_kecamatan.csv")

        vals = {c: props.get(c) for c in CLIMATE_COLS}
        if any(v is None for v in vals.values()):
            centroid = ee.Geometry(feat["geometry"]).centroid()
            fallback = weekly_filled.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=centroid, scale=1000, bestEffort=True
            ).getInfo()
            vals = {k: (v if v is not None else fallback.get(k)) for k, v in vals.items()}

        if any(v is None or not math.isfinite(float(v)) for v in vals.values()):
            raise RuntimeError(f"Nilai iklim tidak valid: {nama}, {year}-W{week:02d}")

        records.append({
            "kode_kec": kode_kec, "tahun": year, "minggu": week,
            "curah_hujan_kumulatif": round(float(vals["curah_hujan_kumulatif"]), 1),
            "curah_hujan_max_harian": round(float(vals["curah_hujan_max_harian"]), 1),
            "suhu_rata": round(float(vals["suhu_rata"]), 2),
            "kelembapan_rata": round(float(vals["kelembapan_rata"]), 2),
            "sumber_hujan": sumber_hujan, "jenis": "observasi",
        })
    return records


def existing_weeks_in_supabase(client, tahun_min: int, tahun_max: int) -> dict:
    """
    {(tahun,minggu): jumlah_baris} untuk rentang tahun yang relevan.

    Di-paginate eksplisit (.range()) -- PostgREST/supabase-py membatasi hasil
    .execute() ke default page size (umumnya 1000 baris). Tanpa pagination,
    begitu tabel sudah besar (pasca backfill penuh, >20rb baris), fungsi ini
    diam-diam salah hitung dan menyangka minggu yang sudah lengkap sebagai
    belum lengkap -> minggu itu diproses ulang sia-sia (boros panggilan GEE,
    walau tidak merusak data karena upsert tetap idempoten).
    """
    if client is None:
        return {}
    counts: dict = {}
    PAGE = 1000
    offset = 0
    while True:
        r = (client.table("iklim_weekly").select("tahun,minggu")
             .gte("tahun", tahun_min).lte("tahun", tahun_max)
             .range(offset, offset + PAGE - 1).execute())
        for row in r.data:
            key = (row["tahun"], row["minggu"])
            counts[key] = counts.get(key, 0) + 1
        if len(r.data) < PAGE:
            break
        offset += PAGE
    return counts


def _fetch_week_with_timeout(fc_jakarta, year: int, week: int, nama_to_kode: dict) -> list[dict] | None:
    """
    Bungkus fetch_one_epi_week() dengan batas waktu WEEK_TIMEOUT_S detik.

    Pakai threading.Thread(daemon=True) + queue, BUKAN ThreadPoolExecutor:
    worker ThreadPoolExecutor standar bukan daemon thread, jadi begitu satu
    minggu hang (persis yang terjadi nyata: backfill diam 1 jam penuh tanpa
    progres), proses utama akan "lanjut" melewati timeout tapi interpreter
    Python TIDAK PERNAH keluar di akhir skrip -- dia menunggu thread yang
    macet itu selesai, yang mana tidak akan pernah terjadi. Thread daemon
    otomatis ditinggal mati saat proses exit, tidak memblokir apa pun.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, WEEK_MAX_RETRY + 2):
        q: queue.Queue = queue.Queue(maxsize=1)

        def _target():
            try:
                q.put(("ok", fetch_one_epi_week(fc_jakarta, year, week, nama_to_kode)))
            except BaseException as exc:  # noqa: BLE001 -- diteruskan ke thread utama, bukan ditelan
                q.put(("err", exc))

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=WEEK_TIMEOUT_S)

        if t.is_alive():
            print(f"  TIMEOUT {year}-W{week:02d} (>{WEEK_TIMEOUT_S}s, percobaan {attempt}) "
                  f"-- kemungkinan GEE lambat/jaringan terputus. Thread lama ditinggal (daemon).")
            last_exc = TimeoutError(f"{year}-W{week:02d} timeout setelah {WEEK_MAX_RETRY + 1} percobaan")
            continue

        status, payload = q.get()
        if status == "ok":
            return payload
        raise payload  # error data asli (bukan timeout) -- jangan ditelan, lempar lagi
    raise last_exc


def fetch_iklim_weekly(targets: list[tuple[int, int]], skip_existing: bool = True) -> list[dict]:
    project_id = init_ee()
    print(f"[ingest_iklim] Earth Engine siap (project={project_id})")

    nama_to_kode = load_master()
    fc_jakarta = load_fc_jakarta()
    client = get_client()

    todo = targets
    if skip_existing and client is not None:
        years = [y for y, _ in targets]
        done = existing_weeks_in_supabase(client, min(years), max(years))
        todo = [t for t in targets if done.get(t, 0) < 44]
        skipped = len(targets) - len(todo)
        if skipped:
            print(f"[ingest_iklim] {skipped} minggu sudah lengkap di Supabase, dilewati "
                  f"(pakai --force untuk paksa ulang semua).")

    all_records: list[dict] = []
    skipped_no_data = []
    for i, (year, week) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {year}-W{week:02d} ...")
        recs = _fetch_week_with_timeout(fc_jakarta, year, week, nama_to_kode)
        if recs is None:
            skipped_no_data.append((year, week))
        else:
            all_records.extend(recs)
            if client is not None:
                client.table("iklim_weekly").upsert(
                    recs, on_conflict="kode_kec,tahun,minggu"
                ).execute()
        time.sleep(REQUEST_DELAY_S)

    if skipped_no_data:
        print(f"[ingest_iklim] {len(skipped_no_data)} minggu dilewati (data satelit belum terbit): "
              f"{skipped_no_data}")
    return all_records


def write_local_csv(records: list[dict], out_path: Path) -> None:
    if not records:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["kode_kec", "tahun", "minggu"] + CLIMATE_COLS + ["sumber_hujan", "jenis"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(records, key=lambda r: (r["tahun"], r["minggu"], r["kode_kec"])):
            w.writerow(r)
    print(f"[ingest_iklim] Ditulis lokal: {out_path} ({len(records)} baris)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weeks-back", type=int, default=8,
                    help="Rolling re-extract N minggu terakhir (mode default)")
    p.add_argument("--tahun-mulai", type=int, help="Backfill: tahun epi awal")
    p.add_argument("--tahun-akhir", type=int, help="Backfill: tahun epi akhir (inklusif)")
    p.add_argument("--force", action="store_true", help="Abaikan cek 'sudah ada', proses ulang semua")
    p.add_argument("--out", type=Path, default=ROOT / "data_raw" / "iklim_weekly_latest.csv")
    args = p.parse_args()

    t0 = time.time()
    if args.tahun_mulai and args.tahun_akhir:
        targets = list(iter_epi_weeks((args.tahun_mulai, 1), (args.tahun_akhir, 52)))
        # pangkas ke minggu terakhir yang benar2 valid di tahun_akhir (tangani 52 vs 53)
        end = last_complete_epi_week()
        targets = [t for t in targets if t <= end]
    else:
        end = last_complete_epi_week()
        start = add_epi_weeks(*end, -(args.weeks_back - 1))
        targets = list(iter_epi_weeks(start, end))

    print(f"[ingest_iklim] Target: {targets[0]} s/d {targets[-1]} ({len(targets)} minggu epi)")
    records = fetch_iklim_weekly(targets, skip_existing=not args.force)
    write_local_csv(records, args.out)

    dt_s = time.time() - t0
    print(f"[ingest_iklim] Selesai dalam {dt_s:.0f}s. {len(records)} baris baru diekstrak.")


if __name__ == "__main__":
    main()
