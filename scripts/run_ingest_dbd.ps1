# Dipanggil oleh Windows Task Scheduler (lihat scripts/setup_task_scheduler.ps1).
#
# surveilans-dinkes.jakarta.go.id memblokir koneksi dari IP cloud (GitHub
# Actions) -- diverifikasi 17 Nov 2026 (Connection reset by peer tepat
# setelah TLS ClientHello). Scraping DBD karena itu dijalankan dari PC lokal
# ini, bukan dari workflow cloud.
#
# RANTAI EVENT, bukan dua jadwal terpisah: begitu scrape lokal ini SUKSES,
# langsung memicu .github/workflows/weekly.yml saat itu juga (gh workflow
# run) -- tidak menunggu jadwal cron cloud yang tetap. Ini supaya kalau
# task ini telat jalan (laptop baru nyala siang, bukan jam 06:00 WIB
# terjadwal), forecast ikut langsung diperbarui saat itu juga, bukan nunggu
# Rabu depan. Jadwal cron cloud (weekly.yml, Rabu 20:00 WIB) tetap ada
# sebagai JARING PENGAMAN kalau rantai ini gagal total (mis. laptop tidak
# pernah nyala sepanjang hari itu) -- bukan jalur utama lagi.

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
    Add-Content $LogFile "`n[run_ingest_dbd.ps1] GAGAL, exit code $ExitCode -- TIDAK memicu workflow cloud. "`
        + "weekly.yml akan jalan sendiri lewat cron fallback-nya memakai data lama."
} else {
    $GhExe = "C:\Program Files\GitHub CLI\gh.exe"
    try {
        & $GhExe workflow run weekly.yml --repo icaldisini/radensa-agc *>> $LogFile
        Add-Content $LogFile "`n[run_ingest_dbd.ps1] Scrape sukses -> workflow cloud dipicu langsung."
    } catch {
        Add-Content $LogFile "`n[run_ingest_dbd.ps1] Scrape sukses, TAPI gagal memicu workflow cloud: $_. "`
            + "weekly.yml tetap akan jalan lewat cron fallback-nya."
    }
}

# Simpan juga salinan log terbaru dengan nama tetap, supaya gampang dicek
# tanpa perlu cari file bertanggal.
Copy-Item $LogFile (Join-Path $LogDir "ingest_dbd_latest.log") -Force
exit $ExitCode
