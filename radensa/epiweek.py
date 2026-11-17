"""
Kalender Epidemiologi (MMWR) — sistem minggu KANONIK untuk seluruh pipeline Radensa.

Dinkes DKI Jakarta (surveilans-dinkes.jakarta.go.id) memakai minggu epidemiologi
MMWR: AHAD (Minggu) s/d SABTU. Ini BUKAN ISO week (Senin s/d Minggu).
Diverifikasi dari halaman umum_kalender.php pada 17 Nov 2026:
    "Selasa, 17 November 2026 | Minggu Kalender Epidemiologi: 46"
    -> epi week 46 2026 = Minggu 15 Nov s/d Sabtu 21 Nov 2026
    (ISO week 46 2026 = Senin 9 Nov s/d Minggu 15 Nov -> beda 6 hari)

Aturan MMWR week 1: minggu (Minggu-Sabtu) pertama yang punya >= 4 hari di Januari.

SELURUH komponen Radensa (kasus DBD, iklim GEE, fitur, forecast) WAJIB memakai
modul ini. Jangan pernah memakai date.isocalendar() / fromisocalendar() untuk
penanggalan minggu.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator, Tuple

__all__ = [
    "epi_year_start", "epi_weeks_in_year", "epi_week_start", "epi_week_range",
    "date_to_epi", "epi_to_date", "iter_epi_weeks", "add_epi_weeks",
    "current_epi_week", "last_complete_epi_week",
]

# Buffer keterlambatan pelaporan RS ke sistem Dinkes (dalam minggu).
# Horizon forecast 5 minggu dipilih supaya tetap menutupi 1 bulan meski ada lag ini.
REPORTING_LAG_WEEKS = 1


def epi_year_start(year: int) -> date:
    """Tanggal Minggu pertama dari epi week 1 tahun `year`."""
    jan1 = date(year, 1, 1)
    # Minggu pada atau sebelum 1 Januari. weekday(): Senin=0..Minggu=6
    sunday = jan1 - timedelta(days=(jan1.weekday() + 1) % 7)
    for cand in (sunday, sunday + timedelta(days=7)):
        days_in_jan = sum(
            1 for i in range(7)
            if (d := cand + timedelta(days=i)).month == 1 and d.year == year
        )
        if days_in_jan >= 4:
            return cand
    raise RuntimeError(f"Gagal menentukan awal epi year {year}")


def epi_weeks_in_year(year: int) -> int:
    """Jumlah epi week dalam satu tahun epi (52 atau 53)."""
    start, nxt = epi_year_start(year), epi_year_start(year + 1)
    return (nxt - start).days // 7


def epi_week_start(year: int, week: int) -> date:
    """Tanggal Minggu (hari pertama) dari epi week `week` tahun `year`."""
    n = epi_weeks_in_year(year)
    if not 1 <= week <= n:
        raise ValueError(f"epi week {week} di luar rentang tahun {year} (1..{n})")
    return epi_year_start(year) + timedelta(days=7 * (week - 1))


def epi_week_range(year: int, week: int) -> Tuple[date, date]:
    """(hari_pertama_inklusif, hari_terakhir_inklusif) -> Minggu s/d Sabtu."""
    start = epi_week_start(year, week)
    return start, start + timedelta(days=6)


def date_to_epi(d: date) -> Tuple[int, int]:
    """Petakan tanggal -> (epi_year, epi_week)."""
    for y in (d.year + 1, d.year, d.year - 1):
        start = epi_year_start(y)
        if start <= d < epi_year_start(y + 1):
            return y, (d - start).days // 7 + 1
    raise RuntimeError(f"Gagal memetakan {d} ke epi week")


def epi_to_date(year: int, week: int) -> date:
    """Alias epi_week_start — dipakai untuk kolom `tanggal` di dataset."""
    return epi_week_start(year, week)


def add_epi_weeks(year: int, week: int, n: int) -> Tuple[int, int]:
    """Geser n minggu (boleh negatif), menangani pergantian tahun 52/53 minggu."""
    return date_to_epi(epi_week_start(year, week) + timedelta(days=7 * n))


def iter_epi_weeks(start: Tuple[int, int], end: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
    """Iterasi (tahun, minggu) inklusif dari `start` s/d `end`."""
    cur, last = start, epi_week_start(*end)
    while epi_week_start(*cur) <= last:
        yield cur
        cur = add_epi_weeks(*cur, 1)


def current_epi_week(today: date | None = None) -> Tuple[int, int]:
    """Epi week yang SEDANG BERJALAN (belum lengkap)."""
    return date_to_epi(today or date.today())


def last_complete_epi_week(
    today: date | None = None, reporting_lag: int = REPORTING_LAG_WEEKS
) -> Tuple[int, int]:
    """
    Epi week terakhir yang boleh dianggap data final.

    Minggu berjalan selalu dibuang (baru sebagian hari terlapor), lalu mundur lagi
    `reporting_lag` minggu untuk menampung laporan RS yang masuk terlambat.

    Menggantikan `cut_to_last_obs()` di notebook lama yang memakai aturan "minggu
    terakhir dengan total kasus > 0" — aturan itu ikut menarik minggu berjalan yang
    masih parsial (terbukti: pada 17 Nov 2026 epi week 46 sudah berisi kasus
    walaupun baru berjalan 3 hari), sehingga kasus_DBD_lag1 jadi terlalu rendah.
    """
    return add_epi_weeks(*current_epi_week(today), -(1 + reporting_lag))
