"""
Fase 6 (bagian 1) - Zonasi surveilans: threshold EWARS (persentil) dan
Endemicity Channel (WHO), sama persis definisinya dengan RADENSA_AGC.ipynb
cell 54 & 56, tapi dihitung dari SELURUH histori yang tersedia (bukan
dibatasi tahun training tertentu).

Kenapa beda dari notebook: di notebook, threshold dihitung dari TRAIN_YEARS
saja untuk menjaga validitas metrik evaluasi (CV) -- itu benar untuk
mengukur generalisasi. Tapi ini kode PRODUKSI, bukan evaluasi: threshold
"berapa kasus yang dianggap normal" untuk operasional zonasi memang
seharusnya pakai seluruh data historis yang ada (makin banyak data, makin
stabil estimasinya), tidak ada risiko leakage karena tidak ada yang
dievaluasi di sini.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ZONA_ORDER = ["Hijau", "Kuning", "Merah"]


def compute_thresholds(panel: pd.DataFrame) -> pd.DataFrame:
    """P25/P50/P75/P90 + mean/std kasus_DBD per kecamatan, dari seluruh histori."""
    g = panel.groupby("kode_kec")["kasus_DBD"]
    out = g.agg(
        p25=lambda x: x.quantile(0.25), p50="median", p75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90), mean="mean", std="std",
    ).reset_index()
    out["basis_tahun_awal"] = int(panel["tahun"].min())
    out["basis_tahun_akhir"] = int(panel["tahun"].max())
    return out


def compute_endemicity_channel(panel: pd.DataFrame) -> pd.DataFrame:
    """ec_mean/ec_std per (kecamatan, minggu), dari seluruh histori -- WHO Endemic Channel."""
    out = (panel.groupby(["kode_kec", "minggu"])["kasus_DBD"]
           .agg(ec_mean="mean", ec_std="std").reset_index())
    out["ec_std"] = out["ec_std"].fillna(0).clip(lower=0.5)
    out["ec_upper_yellow"] = out["ec_mean"] + out["ec_std"]
    out["ec_upper_red"] = out["ec_mean"] + 2 * out["ec_std"]
    return out


def assign_zona_ewars(kasus_pred: float, p50: float, p75: float) -> str:
    if pd.isna(kasus_pred) or pd.isna(p50):
        return "No Data"
    if kasus_pred <= p50:
        return "Hijau"
    if kasus_pred <= p75:
        return "Kuning"
    return "Merah"


def assign_zona_ec(kasus_pred: float, ec_mean: float, ec_upper_yellow: float) -> str:
    if pd.isna(kasus_pred) or pd.isna(ec_mean):
        return "No Data"
    if kasus_pred <= ec_mean:
        return "Hijau"
    if kasus_pred <= ec_upper_yellow:
        return "Kuning"
    return "Merah"


def upsert_thresholds(client, thresholds: pd.DataFrame) -> None:
    rows = thresholds.rename(columns={
        "p25": "p25", "p50": "p50", "p75": "p75", "p90": "p90",
        "mean": "mean_kasus", "std": "std_kasus",
    }).to_dict("records")
    # Skema SQL cuma punya p50,p75,p90,mean,std (p25 tidak dipakai di schema -
    # dibuang di sini, bukan di kolom DB, supaya fungsi ini tetap lengkap
    # hitung semua persentil standar kalau nanti dibutuhkan tempat lain).
    clean_rows = [{
        "kode_kec": r["kode_kec"], "p50": r["p50"], "p75": r["p75"], "p90": r["p90"],
        "mean_kasus": r["mean_kasus"], "std_kasus": r["std_kasus"],
        "basis_tahun_awal": r["basis_tahun_awal"], "basis_tahun_akhir": r["basis_tahun_akhir"],
    } for r in rows]
    client.table("threshold_kecamatan").upsert(clean_rows, on_conflict="kode_kec").execute()


def upsert_endemicity_channel(client, ec: pd.DataFrame) -> None:
    BATCH = 500
    rows = ec.to_dict("records")
    for i in range(0, len(rows), BATCH):
        client.table("endemicity_channel").upsert(
            rows[i:i + BATCH], on_conflict="kode_kec,minggu"
        ).execute()
