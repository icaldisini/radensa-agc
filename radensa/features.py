"""
Fase 4 - Feature engineering TUNGGAL: satu fungsi dipakai training DAN forecast.

Di notebook lama (RADENSA_AGC.ipynb), training memakai pandas (cell 13-14:
groupby().shift(), rolling()) sementara forecast iteratif memakai numpy murni
(cell 69: array H, spatial_mean()) -- dua implementasi terpisah untuk logika
yang seharusnya identik. Keduanya kebetulan sama sekarang, tapi begitu salah
satu diubah tanpa mengubah yang lain, keduanya diam-diam berbeda tanpa ada
yang menangkap -- tidak ada test yang membandingkan hasil keduanya.

Modul ini menghapus duplikasi itu: build_features() dipanggil dari
radensa/train.py (atas seluruh histori) MAUPUN radensa/forecast.py (per
langkah iterasi, dengan kasus_DBD minggu-minggu sebelumnya sebagai actual
untuk minggu lampau dan sebagai prediksi model untuk minggu yang baru saja
diramalkan -- lihat forecast_step()).

21 fitur sejak 20 Nov 2026: 20 FEATURE_COLS di bawah + SPATIAL_SAMEWEEK (kasus tetangga di MINGGU
YANG SAMA, hanya lewat model dua tahap di radensa/twostage.py; nilai aktualnya tidak ada saat meramal).

20 FEATURE_COLS (4 domain, dari RADENSA_AGC.ipynb cell 29):
  Epidemiologi lag  : kasus_DBD_lag1, kasus_DBD_lag2,
                       kasus_DBD_rolling4, kasus_DBD_rolling8,
                       kecamatan_endemisitas
  Iklim current     : curah_hujan_kumulatif, suhu_rata, kelembapan_rata
  Iklim lag         : curah_hujan_lag1, curah_hujan_lag2,
                       suhu_rata_lag1, suhu_rata_lag2,
                       kelembapan_rata_lag1, kelembapan_rata_lag2
  Spasial           : spatial_kasus_tetangga_lag1, spatial_kasus_tetangga_lag2
  Musiman/interaksi : minggu_sin, minggu_cos, ch_suhu_interaksi, suhu_sq

PENTING - konvensi minggu: seluruh perhitungan lag/rolling/sin-cos di modul
ini mengasumsikan baris panel sudah berurutan per EPI WEEK (MMWR) tanpa celah
(1 baris per (kecamatan, tahun, minggu) berurutan). minggu_sin/cos memakai
pembagi 52 mengikuti definisi di RADENSA_AGC.ipynb; epi week ke-53 (2020,
2025) tetap dipakai apa adanya sebagai titik data berurutan, hanya nilai
sin/cos-nya sedikit "melebihi" siklus 52 minggu pada tahun itu -- dampaknya
diabaikan karena hanya 2 dari ~10 minggu dalam seluruh histori.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "curah_hujan_kumulatif", "suhu_rata", "kelembapan_rata",
    "curah_hujan_lag1", "curah_hujan_lag2",
    "suhu_rata_lag1", "suhu_rata_lag2",
    "kelembapan_rata_lag1", "kelembapan_rata_lag2",
    "kasus_DBD_lag1", "kasus_DBD_lag2",
    "spatial_kasus_tetangga_lag1", "spatial_kasus_tetangga_lag2",
    "minggu_sin", "minggu_cos",
    "kasus_DBD_rolling4", "kasus_DBD_rolling8",
    "ch_suhu_interaksi", "suhu_sq",
    "kecamatan_endemisitas",
]
# Fitur ke-21 (mulai 20 Nov 2026): rata-rata kasus kecamatan tetangga pada MINGGU YANG SAMA
# dengan baris yang diprediksi. Nilai aktualnya TIDAK tersedia saat meramal, jadi di training
# dan di forecast nilainya adalah rata-rata PREDIKSI tahap-1 para tetangga (model dua tahap,
# lihat radensa/twostage.py), bukan kasus aktual.
SPATIAL_SAMEWEEK = "spatial_kasus"
FEATURE_COLS_21 = FEATURE_COLS + [SPATIAL_SAMEWEEK]
MINGGU_PER_TAHUN = 52  # lihat catatan epi week 53 di atas


def neighbor_mean(df: pd.DataFrame, values, neighbors: dict[str, list[str]]) -> np.ndarray:
    """
    Rata-rata `values` para tetangga tiap baris pada (tahun, minggu) yang SAMA.
    df  : kolom kecamatan, tahun, minggu (satu baris per kecamatan-minggu)
    values : array sejajar baris df (mis. prediksi tahap-1)
    Return array sejajar df; NaN bila kecamatan tidak punya tetangga.
    """
    tmp = df[["kecamatan", "tahun", "minggu"]].copy()
    tmp["v"] = np.asarray(values, dtype=float)
    wide = tmp.pivot(index=["tahun", "minggu"], columns="kecamatan", values="v")
    out = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for kec in wide.columns:
        nb = [n for n in neighbors.get(kec, []) if n in wide.columns]
        out[kec] = wide[nb].mean(axis=1) if nb else np.nan
    long = out.reset_index().melt(id_vars=["tahun", "minggu"], var_name="kecamatan", value_name="nb")
    merged = tmp[["kecamatan", "tahun", "minggu"]].merge(long, on=["kecamatan", "tahun", "minggu"], how="left")
    return merged["nb"].to_numpy()


def build_features(
    panel: pd.DataFrame,
    neighbors: dict[str, list[str]],
    endemisitas_map: dict[str, float] | None = None,
    endemisitas_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Hitung seluruh FEATURE_COLS dari satu panel kecamatan-epi_week berurutan.

    Parameters
    ----------
    panel : DataFrame dengan kolom wajib
        kecamatan, tahun, minggu, kasus_DBD,
        curah_hujan_kumulatif, suhu_rata, kelembapan_rata
        Harus sudah diurutkan (kecamatan, tahun, minggu) tanpa celah minggu
        (lihat radensa/epiweek.iter_epi_weeks untuk membangun urutan ini).
    neighbors : {kecamatan: [daftar tetangga]} -- pakai neighbors.json kunci
        "feature" (berarah, lihat scripts/build_master.py), BUKAN "queen".
    endemisitas_map : {kecamatan: rata-rata kasus_DBD historis}. Kalau None,
        dihitung sendiri dari seluruh `panel` (hanya tepat untuk training
        pada data penuh; forecast/CV HARUS mengoper map yang dihitung dari
        periode training saja, untuk menghindari kebocoran data).
    endemisitas_years : filter tahun yang dipakai saat endemisitas_map
        dihitung otomatis (diabaikan kalau endemisitas_map sudah diberikan).

    Returns
    -------
    DataFrame sama seperti input + seluruh kolom FEATURE_COLS.
    """
    df = panel.sort_values(["kecamatan", "tahun", "minggu"]).reset_index(drop=True).copy()

    # --- lag iklim (1 & 2 minggu) ---
    g = df.groupby("kecamatan")
    df["curah_hujan_lag1"] = g["curah_hujan_kumulatif"].shift(1)
    df["curah_hujan_lag2"] = g["curah_hujan_kumulatif"].shift(2)
    df["suhu_rata_lag1"] = g["suhu_rata"].shift(1)
    df["suhu_rata_lag2"] = g["suhu_rata"].shift(2)
    df["kelembapan_rata_lag1"] = g["kelembapan_rata"].shift(1)
    df["kelembapan_rata_lag2"] = g["kelembapan_rata"].shift(2)

    # --- lag epidemiologi + rolling (shift dulu baru rolling -> tidak bocor minggu ini) ---
    df["kasus_DBD_lag1"] = g["kasus_DBD"].shift(1)
    df["kasus_DBD_lag2"] = g["kasus_DBD"].shift(2)
    df["kasus_DBD_rolling4"] = g["kasus_DBD"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )
    df["kasus_DBD_rolling8"] = g["kasus_DBD"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )

    # --- spasial: rata-rata kasus_DBD tetangga minggu ini, lalu lag 1 & 2 ---
    # (sama seperti RADENSA_AGC.ipynb cell 13: spatial_kasus dihitung per
    # minggu dari kasus_DBD minggu itu juga, baru di-lag setelahnya.)
    nbr_idx = {kec: [n for n in neighbors.get(kec, [])] for kec in df["kecamatan"].unique()}
    kasus_by_week = {
        (yr, wk): grp.set_index("kecamatan")["kasus_DBD"]
        for (yr, wk), grp in df.groupby(["tahun", "minggu"])
    }
    spatial_kasus = np.empty(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        kec_val = kasus_by_week[(row.tahun, row.minggu)]
        nbrs = nbr_idx.get(row.kecamatan, [])
        vals = [kec_val[n] for n in nbrs if n in kec_val.index]
        spatial_kasus[i] = np.mean(vals) if vals else np.nan
    df["spatial_kasus"] = spatial_kasus
    df["spatial_kasus_tetangga_lag1"] = df.groupby("kecamatan")["spatial_kasus"].shift(1)
    df["spatial_kasus_tetangga_lag2"] = df.groupby("kecamatan")["spatial_kasus"].shift(2)

    # --- musiman & interaksi ---
    df["minggu_sin"] = np.sin(2 * np.pi * df["minggu"] / MINGGU_PER_TAHUN)
    df["minggu_cos"] = np.cos(2 * np.pi * df["minggu"] / MINGGU_PER_TAHUN)
    df["ch_suhu_interaksi"] = df["curah_hujan_kumulatif"] * df["suhu_rata"]
    df["suhu_sq"] = df["suhu_rata"] ** 2

    # --- target encoding (endemisitas), dihitung di luar fungsi untuk CV/forecast ---
    if endemisitas_map is None:
        src = df if endemisitas_years is None else df[df["tahun"].isin(endemisitas_years)]
        endemisitas_map = src.groupby("kecamatan")["kasus_DBD"].mean().to_dict()
    fallback = float(np.mean(list(endemisitas_map.values()))) if endemisitas_map else 0.0
    df["kecamatan_endemisitas"] = df["kecamatan"].map(endemisitas_map).fillna(fallback)

    return df


def compute_endemisitas_map(
    panel: pd.DataFrame, years: list[int] | None = None
) -> dict[str, float]:
    """Hitung kecamatan_endemisitas dari periode tertentu saja (dipakai per-fold CV)."""
    src = panel if years is None else panel[panel["tahun"].isin(years)]
    return src.groupby("kecamatan")["kasus_DBD"].mean().to_dict()


def forecast_step(
    history: pd.DataFrame,
    tahun: int,
    minggu: int,
    iklim_minggu_ini: pd.DataFrame,
    neighbors: dict[str, list[str]],
    endemisitas_map: dict[str, float],
) -> pd.DataFrame:
    """
    Bangun SATU baris fitur per kecamatan untuk minggu forecast (tahun, minggu),
    memakai build_features() yang sama persis dengan training -- bukan
    implementasi numpy terpisah seperti notebook lama.

    Parameters
    ----------
    history : seluruh baris panel SEBELUM (tahun, minggu) ini, kolom sama
        seperti build_features(); untuk minggu forecast ke-2 dst, baris
        forecast minggu sebelumnya (dengan kasus_DBD = prediksi model, bukan
        aktual) sudah harus ada di sini -- lihat radensa/forecast.py.
    iklim_minggu_ini : kolom [kecamatan, curah_hujan_kumulatif, suhu_rata,
        kelembapan_rata] untuk (tahun, minggu) ini. Untuk horizon h=1,2 ini
        observasi GEE asli; h=3-5 ramalan NOAA GFS; fallback klimatologi
        kalau keduanya tidak ada (lihat radensa/forecast.py).
    endemisitas_map : SELALU dari data historis aktual (bukan dihitung ulang
        tiap langkah), supaya tidak ikut "belajar" dari kasus_DBD hasil
        prediksi sendiri.

    Returns
    -------
    DataFrame 1 baris per kecamatan berisi FEATURE_COLS, siap predict().
    """
    already = history[(history["tahun"] == tahun) & (history["minggu"] == minggu)]
    if len(already) > 0:
        raise ValueError(
            f"forecast_step dipanggil untuk {tahun}-W{minggu} yang sudah ada "
            f"{len(already)} baris di history -- forecast_step hanya untuk minggu "
            f"yang BELUM diketahui. Kemungkinan minggu forecast tumpang tindih "
            f"dengan data observasi."
        )

    kec_list = iklim_minggu_ini["kecamatan"].tolist()
    new_rows = iklim_minggu_ini.copy()
    new_rows["tahun"] = tahun
    new_rows["minggu"] = minggu
    new_rows["kasus_DBD"] = np.nan  # belum diketahui -- inilah yang mau diprediksi

    combined = pd.concat([history, new_rows], ignore_index=True)
    combined = build_features(
        combined, neighbors, endemisitas_map=endemisitas_map
    )

    out = combined[(combined["tahun"] == tahun) & (combined["minggu"] == minggu)].copy()
    out = out.set_index("kecamatan").loc[kec_list].reset_index()
    missing = out[FEATURE_COLS].isnull().sum()
    if missing.any():
        raise RuntimeError(
            f"forecast_step {tahun}-W{minggu}: fitur kosong, cek history cukup "
            f"panjang (min. 8 minggu ke belakang):\n{missing[missing > 0]}"
        )
    return out[["kecamatan"] + FEATURE_COLS]
