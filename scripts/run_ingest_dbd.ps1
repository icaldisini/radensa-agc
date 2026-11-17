# Dipanggil oleh Windows Task Scheduler (lihat scripts/setup_task_scheduler.ps1).
#
# surveilans-dinkes.jakarta.go.id memblokir koneksi dari IP cloud (GitHub
# Actions) -- diverifikasi 17 Nov 2026 (Connection reset by peer tepat
# setelah TLS ClientHello). Scraping DBD karena itu dijalankan dari PC lokal
# ini, bukan dari workflow cloud. Jadwal: Rabu 06:00 WIB, 2 jam sebelum
# .github/workflows/weekly.yml jalan (Rabu 08:00 WIB) supaya data DBD sudah
# segar saat training+forecast cloud jalan.

$ErrorActionPreference = "Stop"
$ProjectDir = "C:\Users\Faishal Muflih\Documents\Radensa_AGC"
$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$LogFile = Join-Path $LogDir "ingest_dbd_$Timestamp.log"

Set-Location $ProjectDir
try {
    python -I radensa\ingest_dbd.py *> $LogFile
    $ExitCode = $LASTEXITCODE
} catch {
    $_ | Out-File -Append $LogFile
    $ExitCode = 1
}

if ($ExitCode -ne 0) {
    Add-Content $LogFile "`n[run_ingest_dbd.ps1] GAGAL, exit code $ExitCode"
}

# Simpan juga salinan log terbaru dengan nama tetap, supaya gampang dicek
# tanpa perlu cari file bertanggal.
Copy-Item $LogFile (Join-Path $LogDir "ingest_dbd_latest.log") -Force
exit $ExitCode
