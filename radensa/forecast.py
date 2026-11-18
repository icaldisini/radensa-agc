"""
Fase 6 (bagian 2) - Orkestrasi forecast mingguan: 5 minggu ke depan per
kecamatan, pakai model yang SEDANG aktif (is_active=true di model_runs).

Asumsi desain (lihat .github/workflows/weekly.yml): script ini dijalankan
SETELAH radensa/train.py (--refit-only mingguan, atau retune penuh kalau
sudah waktunya) pada job CI yang SAMA -- artefak model (models/<versi>.pkl)
masih ada di disk lokal job itu karena belum di-checkout ulang. Ini sengaja
dipilih dibanding upload model ke Supabase Storage: lebih sederhana, dan
tidak ada skenario nyata forecast.py perlu jalan TANPA train.py mendahuluinya
di job yang sama.

LINGKUP YANG SENGAJA BELUM DIKERJAKAN (jujur, bukan diam-diam dilewati):
  1. Iklim minggu forecast h=3-5 pakai ramalan NOAA GFS -- rencana awal
     "Observasi + GFS". Untuk h=1-2, dicek dulu ke iklim_weekly: kalau
     observasi GEE asli untuk minggu itu SUDAH terbit (kadang terjadi,
     karena lag satelit CHIRPS/ERA5 tidak selalu seketat buffer pelaporan
     DBD), dipakai. Kalau belum ada SAMA SEKALI (h=3-5 pasti begini,
     h=1-2 kadang), fallback ke klimatologi (rata-rata historis per
     kecamatan-minggu) -- ini baseline yang SAH, dipakai juga di notebook
     riset asli, bukan nilai dummy. Integrasi GFS menyusul sebagai
     peningkatan, bukan kebutuhan untuk sistem jalan dengan benar.
  2. Uncertainty Quantification (bootstrap CI 95%, 300 iterasi di notebook,
     ~7 menit) tidak dihitung ulang di sini -- mahal untuk job mingguan.
     ci_lower_95/ci_upper_95 di tabel forecast tetap NULL (kolom nullable).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radensa.data_loader import load_iklim_for_weeks, load_iklim_raw, load_neighbors, load_panel
from radensa.epiweek import add_epi_weeks, current_epi_week, epi_to_date
from radensa.features import FEATURE_COLS, forecast_step
from radensa.supabase_client import require_client
from radensa.zonasi import (
    assign_zona_ec, assign_zona_ewars, compute_endemicity_channel,
    compute_thresholds, upsert_endemicity_channel, upsert_thresholds,
)

N_FORECAST = 5
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CLIMATE_COLS = ["curah_hujan_kumulatif", "suhu_rata", "kelembapan_rata"]


def load_active_model(client) -> tuple[object, dict, dict, str]:
    row = (client.table("model_runs").select("*").eq("is_active", True).limit(1).execute())
    if not row.data:
        raise RuntimeError(
            "Tidak ada model aktif (model_runs.is_active=true) -- jalankan "
            "radensa/train.py minimal sekali dulu."
        )
    meta = row.data[0]
    artifact_path = MODEL_DIR / meta["artifact_path"]
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artefak model aktif '{meta['versi']}' tidak ada di {artifact_path}. "
            f"Kalau menjalankan forecast.py terpisah dari train.py (beda job CI), "
            f"artefak lokal tidak ikut -- lihat docstring modul ini soal asumsi desain."
        )
    with artifact_path.open("rb") as fh:
        blob = pickle.load(fh)
    return blob["model"], blob["enc_map"], meta, meta["versi"]


def climatology_fallback(iklim_raw: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata historis per (kode_kec, minggu) -- lihat docstring modul: ini
    baseline SAH untuk minggu forecast yang belum punya observasi GEE sama sekali."""
    return iklim_raw.groupby(["kode_kec", "minggu"])[CLIMATE_COLS].mean().reset_index()


def build_climate_for_forecast_weeks(
    forecast_weeks: list[tuple[int, int]], kec_list: list[str]
) -> dict[tuple[int, int], pd.DataFrame]:
    """
    {(tahun,minggu): DataFrame[kecamatan, curah_hujan_kumulatif, suhu_rata, kelembapan_rata]}

    Prioritas per minggu: observasi GEE asli (kalau sudah terbit & lengkap
    44 kecamatan) -> klimatologi historis per (kecamatan, minggu_tahun).
    """
    raw = load_iklim_raw()
    clim = climatology_fallback(raw).set_index(["kode_kec", "minggu"])
    real = load_iklim_for_weeks(forecast_weeks)

    out = {}
    for tahun, minggu in forecast_weeks:
        sub = real[(real.tahun == tahun) & (real.minggu == minggu)]
        if len(sub) == len(kec_list) and len(kec_list) == len(set(sub.kode_kec)):
            df = sub[["kode_kec"] + CLIMATE_COLS].copy()
            sumber = "observasi"
        else:
            rows = []
            for kec in kec_list:
                key = (kec, minggu)
                vals = clim.loc[key] if key in clim.index else clim.xs(minggu, level="minggu").mean()
                rows.append({"kode_kec": kec, **vals[CLIMATE_COLS].to_dict()})
            df = pd.DataFrame(rows)
            sumber = "klimatologi"
        print(f"  iklim {tahun}-W{minggu:02d}: sumber={sumber} ({len(df)} kecamatan)")
        out[(tahun, minggu)] = df
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-forecast", type=int, default=N_FORECAST)
    args = p.parse_args()

    t0 = time.time()
    client = require_client()

    print("[forecast] Memuat panel historis...")
    panel = load_panel()
    neighbors = load_neighbors(client, tipe="feature")
    kode_to_nama = dict(zip(panel["kode_kec"], panel["kecamatan"]))
    kode_list = sorted(panel["kode_kec"].unique())

    last_tahun, last_minggu = max(zip(panel["tahun"], panel["minggu"]))
    print(f"[forecast] Data terakhir: {last_tahun}-W{last_minggu:02d}")

    model, enc_map, meta, model_versi = load_active_model(client)
    print(f"[forecast] Model aktif: {model_versi}")

    # Horizon dihitung dari MINGGU BERJALAN, bukan dari data terakhir. Data Dinkes
    # selalu terlambat 1-2 minggu (mis. data terakhir W44 saat W46 berjalan),
    # padahal yang dibutuhkan pengguna adalah "minggu ini + 4 minggu ke depan".
    # Jadi forecast iteratif jalan dari (data terakhir + 1) sampai (minggu
    # berjalan + n - 1); minggu yang sudah lewat hanya dipakai sebagai batu
    # loncatan fitur lag dan TIDAK disimpan (skema membatasi horizon 1..5).
    now_jkt = datetime.now(timezone(timedelta(hours=7))).date()
    cur_week = current_epi_week(now_jkt)
    target = add_epi_weeks(*cur_week, args.n_forecast - 1)
    if epi_to_date(*target) <= epi_to_date(last_tahun, last_minggu):
        target = add_epi_weeks(last_tahun, last_minggu, args.n_forecast)
    forecast_weeks = []
    cur = (last_tahun, last_minggu)
    while epi_to_date(*cur) < epi_to_date(*target):
        cur = add_epi_weeks(*cur, 1)
        forecast_weeks.append(cur)
    store_weeks = [w for w in forecast_weeks if epi_to_date(*w) >= epi_to_date(*cur_week)]
    if not store_weeks:
        store_weeks = forecast_weeks[-args.n_forecast:]
    store_weeks = store_weeks[:args.n_forecast]
    horizon_of = {w: i for i, w in enumerate(store_weeks, 1)}
    print(f"[forecast] Minggu berjalan: {cur_week[0]}-W{cur_week[1]:02d}")
    print(f"[forecast] Dihitung: {forecast_weeks[0]} s/d {forecast_weeks[-1]} "
          f"({len(forecast_weeks)} langkah); disimpan: {store_weeks[0]} s/d {store_weeks[-1]}")

    print("[forecast] Menyiapkan input iklim per minggu forecast...")
    climate_by_week = build_climate_for_forecast_weeks(forecast_weeks, kode_list)

    print("[forecast] Menghitung threshold zonasi (EWARS + Endemicity Channel)...")
    thresholds = compute_thresholds(panel)
    ec = compute_endemicity_channel(panel)
    upsert_thresholds(client, thresholds)
    upsert_endemicity_channel(client, ec)
    thresh_by_kode = thresholds.set_index("kode_kec").to_dict("index")
    ec_by_key = ec.set_index(["kode_kec", "minggu"]).to_dict("index")

    print("[forecast] Forecast iteratif...")
    history = panel[["kecamatan", "tahun", "minggu", "kasus_DBD",
                      "curah_hujan_kumulatif", "suhu_rata", "kelembapan_rata"]].copy()
    nama_to_kode = {v: k for k, v in kode_to_nama.items()}

    forecast_rows = []
    for h, (tahun, minggu) in enumerate(forecast_weeks, 1):
        iklim_wk = climate_by_week[(tahun, minggu)].copy()
        iklim_wk["kecamatan"] = iklim_wk["kode_kec"].map(kode_to_nama)
        iklim_wk = iklim_wk[["kecamatan"] + CLIMATE_COLS]

        feat = forecast_step(history, tahun, minggu, iklim_wk, neighbors, enc_map)
        X = feat[FEATURE_COLS].fillna(0)
        pred = np.clip(model.predict(X), 0, None)
        feat = feat.assign(kasus_pred=pred)

        print(f"  Minggu {h} ({tahun}-W{minggu:02d}): rata-rata {pred.mean():.2f} kasus/kecamatan")

        new_hist = iklim_wk.copy()
        new_hist["tahun"], new_hist["minggu"] = tahun, minggu
        new_hist["kasus_DBD"] = pred
        history = pd.concat([history, new_hist], ignore_index=True)

        if (tahun, minggu) not in horizon_of:
            continue  # minggu lampau tanpa data: hanya batu loncatan, tidak disimpan
        for kec, kasus_pred in zip(feat["kecamatan"], pred):
            kode = nama_to_kode[kec]
            th = thresh_by_kode.get(kode, {})
            ecr = ec_by_key.get((kode, minggu), {})
            forecast_rows.append({
                "kode_kec": kode, "tahun": tahun, "minggu": minggu, "horizon": horizon_of[(tahun, minggu)],
                "kasus_pred": round(float(kasus_pred), 3),
                "zona_ewars": assign_zona_ewars(kasus_pred, th.get("p50"), th.get("p75")),
                "zona_ec": assign_zona_ec(kasus_pred, ecr.get("ec_mean"), ecr.get("ec_upper_yellow")),
            })

    print("[forecast] Menyimpan hasil ke Supabase...")
    run = (client.table("forecast_runs")
           .insert({"model_versi": model_versi, "basis_tahun": int(last_tahun),
                     "basis_minggu": int(last_minggu)})
           .execute())
    run_id = run.data[0]["id"]
    for r in forecast_rows:
        r["run_id"] = run_id
    try:
        client.table("forecast").insert(forecast_rows).execute()
    except Exception:
        # Jangan tinggalkan run kosong: view forecast_terkini memakai run TERBARU,
        # jadi run tanpa baris akan mengosongkan aplikasi warga.
        client.table("forecast_runs").delete().eq("id", run_id).execute()
        raise

    zona_counts = pd.DataFrame(forecast_rows).groupby(["minggu", "zona_ewars"]).size().unstack(fill_value=0)
    print(f"\n[forecast] Distribusi zona EWARS per minggu:\n{zona_counts}")
    print(f"\n[forecast] Selesai dalam {time.time() - t0:.1f}s. run_id={run_id}, "
          f"{len(forecast_rows)} baris forecast tersimpan.")


if __name__ == "__main__":
    main()
