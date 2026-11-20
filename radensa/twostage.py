"""
Model dua tahap (21 fitur) = 20 fitur dasar + rata-rata kasus tetangga pada MINGGU YANG SAMA.

Masalah: kasus tetangga pada minggu t BELUM diketahui saat meramal minggu t (sama tidak
diketahuinya dengan targetnya). Nilai aktualnya hanya ada di data latih -- memakainya
langsung membuat akurasi validasi tampak lebih baik daripada yang bisa dicapai saat meramal.

Solusi yang bisa dijalankan saat meramal:
  Tahap 1 : model 20 fitur memprediksi kasus minggu t untuk SEMUA kecamatan.
  Fitur 21: rata-rata prediksi tahap-1 para tetangga pada minggu t (bukan nilai aktual).
  Tahap 2 : model 21 fitur (20 dasar + fitur di atas) -> prediksi akhir.
Tahap 2 dilatih dengan prediksi OUT-OF-FOLD tahap 1 (leave-one-year-out) supaya kualitas
fitur ke-21 di data latih mirip kualitasnya saat meramal.

Uji (kondisi meramal, LightGBM Tweedie parameter tetap, uji 2023-2026): selisih R2 terhadap
20 fitur = -0.006 / +0.007 / +0.002 / +0.004 (rata-rata +0.002). Fitur "oracle" yang memakai
tetangga AKTUAL memberi sekitar +0.022, tetapi itu tidak tersedia saat meramal.

Catatan kecil: endemisitas kecamatan di df_feat dihitung dari seluruh baris latih, termasuk
tahun yang sedang di-hold-out pada langkah OOF -- bocor ringan (satu konstanta per
kecamatan), sama kondisinya dengan yang diuji di atas.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from radensa.features import FEATURE_COLS, FEATURE_COLS_21, SPATIAL_SAMEWEEK, neighbor_mean

RANDOM_STATE = 42
OBJECTIVE = "tweedie"
TARGET = "kasus_DBD"
_RESERVED = ("objective", "random_state", "n_jobs", "verbosity")


def make_lgb(params: dict) -> lgb.LGBMRegressor:
    """
    SATU-SATUNYA tempat model LightGBM dibuat. objective Tweedie harus eksplisit:
    study.best_params dari Optuna hanya memuat parameter yang disetel (tweedie_variance_power,
    dst.) TANPA kunci 'objective'; tanpa baris ini model final jatuh ke objective bawaan
    (regresi L2) dan tweedie_variance_power diabaikan diam-diam.
    """
    p = {k: v for k, v in params.items() if k not in _RESERVED}
    return lgb.LGBMRegressor(objective=OBJECTIVE, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **p)


def fit_two_stage(df_feat: pd.DataFrame, neighbors: dict, params: dict, train_mask):
    """
    df_feat    : hasil build_features() atas seluruh panel (endemisitas sudah terisi).
    train_mask : baris yang dipakai melatih (bool, sejajar df_feat).
    Return (model_tahap1, model_tahap2).
    """
    tr = np.asarray(train_mask, dtype=bool)
    X20 = df_feat[FEATURE_COLS].fillna(0)
    y = df_feat[TARGET].astype(float).to_numpy()
    tahun = df_feat["tahun"].to_numpy()

    model1 = make_lgb(params).fit(X20[tr], y[tr])

    p1 = np.zeros(len(df_feat))
    for k in np.unique(tahun[tr]):
        te = tr & (tahun == k)
        m = make_lgb(params).fit(X20[tr & ~te], y[tr & ~te])
        p1[te] = np.clip(m.predict(X20[te]), 0, None)

    nb = np.nan_to_num(neighbor_mean(df_feat[tr], p1[tr], neighbors))
    X21 = X20[tr].assign(**{SPATIAL_SAMEWEEK: nb})[FEATURE_COLS_21]
    model2 = make_lgb(params).fit(X21, y[tr])
    return model1, model2


def predict_two_stage(model1, model2, X20: pd.DataFrame, keys: pd.DataFrame, neighbors: dict):
    """
    X20  : DataFrame FEATURE_COLS (sudah fillna(0)).
    keys : DataFrame kecamatan, tahun, minggu sejajar baris X20 (satu baris per kecamatan-minggu).
    Return (prediksi_tahap1, prediksi_akhir), keduanya sudah di-clip >= 0.
    """
    p1 = np.clip(model1.predict(X20), 0, None)
    nb = np.nan_to_num(neighbor_mean(keys, p1, neighbors))
    X21 = X20.assign(**{SPATIAL_SAMEWEEK: nb})[FEATURE_COLS_21]
    p2 = np.clip(model2.predict(X21), 0, None)
    return p1, p2
