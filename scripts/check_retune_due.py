"""
Exit 0 kalau retune penuh (Optuna+CV) sudah waktunya, exit 1 kalau belum.

Dipakai .github/workflows/weekly.yml untuk memutuskan antara
`train.py --n-trials 100` (retune, ~5 mingguan) vs `train.py --refit-only`
(retrain ringan, mingguan) TANPA perlu dua cron terpisah -- satu workflow
mingguan yang mengecek sendiri kapan terakhir kali retune penuh terjadi.

"Retune penuh" dibedakan dari "refit" lewat cv_detail: baris refit selalu
berisi key "refit_dari" (lihat radensa/train.py --refit-only), baris
retune penuh tidak.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radensa.supabase_client import require_client

RETUNE_INTERVAL_DAYS = 35  # ~5 minggu, menyamai horizon forecast


def main() -> int:
    client = require_client()
    rows = (client.table("model_runs").select("created_at,cv_detail")
            .order("created_at", desc=True).execute()).data

    full_retune = [
        r for r in rows
        if not (isinstance(r["cv_detail"], dict) and "refit_dari" in r["cv_detail"])
    ]
    if not full_retune:
        print("Belum pernah ada retune penuh sama sekali -- retune sekarang.")
        return 0

    last = full_retune[0]
    created = datetime.fromisoformat(last["created_at"].replace("Z", "+00:00"))
    days_since = (datetime.now(timezone.utc) - created).days
    due = days_since >= RETUNE_INTERVAL_DAYS
    print(f"Retune penuh terakhir: {days_since} hari lalu ({last['created_at']}). "
          f"{'SUDAH WAKTUNYA' if due else 'belum, refit ringan saja'}.")
    return 0 if due else 1


if __name__ == "__main__":
    sys.exit(main())
