"""
Fase 5 - Training LightGBM Tweedie + rolling CV, memakai radensa/features.py
(bukan implementasi terpisah) dan data dari Supabase (bukan CSV/Drive).

Lingkup yang SENGAJA tidak direplikasi dari RADENSA_AGC.ipynb: notebook riset
membandingkan 5 model (Naive, Poisson, XGBoost, LightGBM, CatBoost) untuk
MEMILIH arsitektur terbaik -- itu pekerjaan riset satu kali, sudah selesai
(LightGBM Tweedie menang, R2=0.7043 pada fold terakhir, mengalahkan naive
baseline 0.5661). train.py ini adalah pipeline RETRAIN otomatis bulanan:
arsitektur sudah final, yang perlu diulang adalah melatihnya lagi dengan
data terbaru setiap bulan. Mereplikasi full 5-model bake-off setiap bulan
hanya buang waktu komputasi tanpa nilai tambah.

Desain CV: rolling/expanding window, generik terhadap tahun yang tersedia
(tidak hardcode 2017-2026 seperti notebook) -- begitu makin banyak tahun
terkumpul di Supabase, jumlah fold bertambah otomatis. Minimal
MIN_TRAIN_YEARS tahun data training sebelum fold pertama dibuat.

Target encoding (kecamatan_endemisitas) dihitung ULANG per fold dari
train_years fold itu saja (bukan dari seluruh data) untuk mencegah
kebocoran data -- lag/rolling/spasial dihitung SEKALI saja di seluruh
panel (tidak bergantung pada split, murni kronologis) lalu kolom
endemisitas di-override per fold supaya tidak ada komputasi berulang
yang sia-sia.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radensa.data_loader import load_neighbors, load_panel
from radensa.epiweek import last_complete_epi_week
from radensa.features import FEATURE_COLS, build_features, compute_endemisitas_map
from radensa.supabase_client import get_client

RANDOM_STATE = 42
TARGET_RAW = "kasus_DBD"
MIN_TRAIN_YEARS = 5
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_STATE)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mask = y_true > 0
    mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            if mask.sum() > 0 else float("nan"))
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2), "MAPE": mape}


def build_folds(all_years: list[int], min_train_years: int = MIN_TRAIN_YEARS) -> list[dict]:
    """Expanding window: fold ke-i latih di semua tahun sebelum tahun ke-i, validasi di tahun ke-i."""
    folds = []
    for i, y in enumerate(all_years):
        if i < min_train_years:
            continue
        folds.append({"fold": len(folds) + 1, "train_years": list(all_years[:i]), "val_years": [y]})
    return folds


def get_fold_xy(df_feat, fold: dict, feature_cols: list[str] = FEATURE_COLS):
    train_mask = df_feat["tahun"].isin(fold["train_years"])
    val_mask = df_feat["tahun"].isin(fold["val_years"])
    X_train = df_feat.loc[train_mask, feature_cols].fillna(0)
    X_val = df_feat.loc[val_mask, feature_cols].fillna(0)
    y_train = df_feat.loc[train_mask, TARGET_RAW].astype(float).values
    y_val = df_feat.loc[val_mask, TARGET_RAW].astype(float).values
    return X_train, y_train, X_val, y_val


LGB_SEARCH_SPACE = {
    "tweedie_variance_power": (1.0, 1.9, "float"),
    "n_estimators": (300, 1500, "int_step100"),
    "num_leaves": (15, 150, "int"),
    "max_depth": (4, 12, "int"),
    "learning_rate": (0.01, 0.15, "float_log"),
    "min_child_samples": (5, 60, "int"),
    "subsample": (0.6, 1.0, "float"),
    "subsample_freq": (1, 7, "int"),
    "colsample_bytree": (0.5, 1.0, "float"),
    "reg_alpha": (1e-3, 10, "float_log"),
    "reg_lambda": (1e-3, 10, "float_log"),
}


def tune_lgb(X_tr, y_tr, X_vl, y_vl, n_trials: int) -> dict:
    """Sama persis search space-nya dengan RADENSA_AGC.ipynb cell 43 (Model 4 - LightGBM)."""

    def objective(trial):
        params = {"objective": "tweedie", "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1}
        for name, (lo, hi, kind) in LGB_SEARCH_SPACE.items():
            if kind == "int":
                params[name] = trial.suggest_int(name, lo, hi)
            elif kind == "int_step100":
                params[name] = trial.suggest_int(name, lo, hi, step=100)
            elif kind == "float":
                params[name] = trial.suggest_float(name, lo, hi)
            elif kind == "float_log":
                params[name] = trial.suggest_float(name, lo, hi, log=True)
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        pred = np.clip(m.predict(X_vl), 0, None)
        return -np.sqrt(mean_squared_error(y_vl, pred))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def run_cv(df_raw, neighbors: dict, folds: list[dict], n_trials: int) -> tuple[list[dict], list[dict]]:
    # Lag/rolling/spasial dihitung SEKALI (murni kronologis, tidak bergantung
    # split); endemisitas di-override per fold di bawah (butuh train_years
    # fold itu saja, supaya tidak bocor).
    df_feat_base = build_features(df_raw, neighbors)

    fold_results, fitted = [], []
    for fold in folds:
        df_feat = df_feat_base.copy()
        enc_map = compute_endemisitas_map(df_raw, years=fold["train_years"])
        fallback = float(np.mean(list(enc_map.values()))) if enc_map else 0.0
        df_feat["kecamatan_endemisitas"] = df_feat["kecamatan"].map(enc_map).fillna(fallback)

        X_tr, y_tr, X_vl, y_vl = get_fold_xy(df_feat, fold)
        best_params = tune_lgb(X_tr, y_tr, X_vl, y_vl, n_trials)
        model = lgb.LGBMRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        pred = np.clip(model.predict(X_vl), 0, None)
        metrics = evaluate(y_vl, pred)
        metrics.update(fold=fold["fold"], val_year=fold["val_years"][0])
        fold_results.append(metrics)
        fitted.append({"model": model, "params": best_params, "enc_map": enc_map})

        print(f"  Fold {fold['fold']} (train {fold['train_years'][0]}-{fold['train_years'][-1]} "
              f"-> val {fold['val_years'][0]}): "
              f"MAE={metrics['MAE']:.3f} RMSE={metrics['RMSE']:.3f} "
              f"R2={metrics['R2']:.4f} MAPE={metrics['MAPE']:.1f}%")
    return fold_results, fitted


def train_final_model(df_raw, neighbors: dict, best_params: dict) -> tuple[lgb.LGBMRegressor, dict]:
    """Model produksi: dilatih di SELURUH data (bukan hanya sampai fold terakhir)."""
    enc_map = compute_endemisitas_map(df_raw)  # seluruh histori -- ini model final, bukan CV
    df_feat = build_features(df_raw, neighbors, endemisitas_map=enc_map)
    X_all = df_feat[FEATURE_COLS].fillna(0)
    y_all = df_feat[TARGET_RAW].astype(float).values
    model = lgb.LGBMRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    model.fit(X_all, y_all)
    return model, enc_map


def refit_latest_params(client, df_raw, neighbors: dict) -> tuple[lgb.LGBMRegressor, dict, dict]:
    """
    "Retrain" ringan mingguan: fit ulang bobot model di seluruh data TERBARU,
    pakai hyperparameter hasil Optuna PALING TERAKHIR yang tersimpan di
    model_runs (tanpa mencari ulang -- itu tugas main()/run_cv() yang jalan
    tiap ~5 minggu, bukan tiap minggu). Dipanggil dari radensa/forecast.py.

    Beda dengan train_final_model(): fungsi itu dipanggil SETELAH run_cv(),
    pakai params dari fold CV yang baru saja selesai dituning. Fungsi ini
    dipanggil TANPA run_cv() sama sekali -- murni reuse params lama + data
    baru, supaya "retrain tiap minggu" tetap murah (hitungan detik, bukan
    menit) walau tetap benar-benar melatih ulang bobot dengan data terkini.
    """
    row = (client.table("model_runs").select("hyperparams,versi")
           .order("created_at", desc=True).limit(1).execute())
    if not row.data:
        raise RuntimeError(
            "Belum ada model_runs sama sekali -- jalankan train.py (retune penuh) "
            "minimal sekali dulu sebelum bisa refit mingguan."
        )
    best_params = row.data[0]["hyperparams"]
    model, enc_map = train_final_model(df_raw, neighbors, best_params)
    return model, enc_map, best_params


def promote_if_better(client, new_metrics: dict, versi: str, max_regression_pct: float = 5.0) -> bool:
    """
    Jadikan model baru aktif KECUALI R2-nya turun > max_regression_pct dari
    model aktif saat ini (sesuai rencana arsitektur: "promote hanya kalau
    R2 tidak turun >5%"). Model pertama selalu dipromosikan (belum ada
    pembanding).
    """
    cur = (client.table("model_runs").select("r2,versi")
           .eq("is_active", True).limit(1).execute())
    if cur.data:
        r2_lama, versi_lama = cur.data[0]["r2"], cur.data[0]["versi"]
        regresi_pct = (r2_lama - new_metrics["R2"]) / abs(r2_lama) * 100 if r2_lama else 0
        if regresi_pct > max_regression_pct:
            print(f"[train] TIDAK dipromosikan: R2 turun {regresi_pct:.1f}% dari model aktif "
                  f"'{versi_lama}' (R2={r2_lama:.4f} -> {new_metrics['R2']:.4f}).")
            return False
        client.table("model_runs").update({"is_active": False}).eq("versi", versi_lama).execute()
    client.table("model_runs").update({"is_active": True}).eq("versi", versi).execute()
    print(f"[train] Dipromosikan jadi model aktif: {versi}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=100, help="Jumlah trial Optuna per fold (default: 100)")
    p.add_argument("--min-train-years", type=int, default=MIN_TRAIN_YEARS)
    p.add_argument("--no-promote", action="store_true", help="Latih & simpan lokal saja, jangan promosikan ke Supabase")
    p.add_argument("--refit-only", action="store_true",
                   help="Retrain ringan mingguan: fit ulang pakai hyperparameter terakhir, "
                        "TANPA Optuna/CV (detik, bukan menit). Lihat refit_latest_params().")
    args = p.parse_args()

    t0 = time.time()
    print("[train] Memuat panel dari Supabase...")
    df_raw = load_panel()
    client = get_client()
    neighbors = load_neighbors(client, tipe="feature") if client else {}
    if not neighbors:
        raise RuntimeError("kecamatan_tetangga kosong -- jalankan scripts/seed_master.py dulu.")

    all_years = sorted(df_raw["tahun"].unique().tolist())
    print(f"[train] Panel: {len(df_raw):,} baris, tahun {all_years[0]}-{all_years[-1]}, "
          f"{df_raw['kecamatan'].nunique()} kecamatan")

    if args.refit_only:
        if client is None:
            raise RuntimeError("--refit-only butuh Supabase (baca hyperparams model_runs terakhir).")
        print("[train] Mode refit ringan (tanpa Optuna/CV)...")
        final_model, final_enc_map, last_params = refit_latest_params(client, df_raw, neighbors)
        last_year, last_week = max(zip(df_raw["tahun"], df_raw["minggu"]))
        versi = f"lgb-{last_year}w{last_week:02d}-refit"
        MODEL_DIR.mkdir(exist_ok=True)
        artifact_path = MODEL_DIR / f"{versi}.pkl"
        with artifact_path.open("wb") as fh:
            pickle.dump({"model": final_model, "enc_map": final_enc_map,
                         "feature_cols": FEATURE_COLS, "params": last_params}, fh)
        fi = dict(zip(FEATURE_COLS, [int(x) for x in final_model.feature_importances_]))
        source = (client.table("model_runs").select("versi,r2,mae,rmse,mape")
                  .order("created_at", desc=True).limit(1).execute()).data[0]
        row = {
            "versi": versi, "algoritma": "LightGBM (Tweedie)", "hyperparams": last_params,
            "r2": source["r2"], "mae": source["mae"], "rmse": source["rmse"], "mape": source["mape"],
            "cv_detail": {"refit_dari": source["versi"],
                          "catatan": "retrain ringan mingguan, metrik dari tuning terakhir, tidak diukur ulang"},
            "feature_importance": dict(sorted(fi.items(), key=lambda kv: -kv[1])),
            "data_sampai_tahun": int(last_year), "data_sampai_minggu": int(last_week),
            "artifact_path": str(artifact_path.name), "is_active": False,
        }
        client.table("model_runs").upsert(row, on_conflict="versi").execute()
        if not args.no_promote:
            promote_if_better(client, source, versi)
        print(f"[train] Refit selesai dalam {(time.time() - t0):.1f}s -> {artifact_path}")
        return

    folds = build_folds(all_years, args.min_train_years)
    if not folds:
        raise RuntimeError(
            f"Data cuma {len(all_years)} tahun, butuh minimal {args.min_train_years + 1} "
            f"untuk 1 fold CV."
        )
    print(f"[train] {len(folds)} fold CV (expanding window, min {args.min_train_years} tahun latih)")

    print(f"\n[train] MODEL: LightGBM (Tweedie), {args.n_trials} trial Optuna/fold")
    fold_results, fitted = run_cv(df_raw, neighbors, folds, args.n_trials)

    mean_r2 = float(np.mean([f["R2"] for f in fold_results]))
    print(f"\n[train] Rata-rata CV: R2={mean_r2:.4f}")

    last = fold_results[-1]
    last_params = fitted[-1]["params"]
    print(f"[train] Melatih model final (seluruh data) pakai params fold terakhir...")
    final_model, final_enc_map = train_final_model(df_raw, neighbors, last_params)

    MODEL_DIR.mkdir(exist_ok=True)
    last_year, last_week = max(zip(df_raw["tahun"], df_raw["minggu"]))
    versi = f"lgb-{last_year}w{last_week:02d}"
    artifact_path = MODEL_DIR / f"{versi}.pkl"
    with artifact_path.open("wb") as fh:
        pickle.dump({"model": final_model, "enc_map": final_enc_map,
                     "feature_cols": FEATURE_COLS, "params": last_params}, fh)
    print(f"[train] Model final disimpan: {artifact_path}")

    fi = dict(zip(FEATURE_COLS, [int(x) for x in final_model.feature_importances_]))
    fi_sorted = dict(sorted(fi.items(), key=lambda kv: -kv[1]))

    if client is not None:
        row = {
            "versi": versi, "algoritma": "LightGBM (Tweedie)",
            "hyperparams": last_params,
            "r2": last["R2"], "mae": last["MAE"], "rmse": last["RMSE"], "mape": last["MAPE"],
            "cv_detail": fold_results,
            "feature_importance": fi_sorted,
            "data_sampai_tahun": int(last_year), "data_sampai_minggu": int(last_week),
            "artifact_path": str(artifact_path.name), "is_active": False,
        }
        client.table("model_runs").upsert(row, on_conflict="versi").execute()
        print(f"[train] Metadata model tersimpan ke Supabase (model_runs, versi={versi})")
        if not args.no_promote:
            promote_if_better(client, last, versi)
    else:
        print("[train] Supabase tidak terkonfigurasi -- metadata model hanya lokal.")

    print(f"\n[train] Selesai dalam {(time.time() - t0) / 60:.1f} menit.")


if __name__ == "__main__":
    main()
