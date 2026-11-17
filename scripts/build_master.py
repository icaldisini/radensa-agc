"""
Fase 0 - Bekukan data referensi spasial Radensa (sekali jalan, hasilnya di-commit).

Mengganti ketergantungan pada shapefile GADM 234 MB di Google Drive yang di notebook
lama diunduh ulang setiap kali run.

Memperbaiki dua cacat matriks ketetanggaan notebook lama:
  1. gdf punya 47 baris untuk 44 kecamatan (1 poligon 'DANAU SUNTER DLL' +
     fragmen duplikat KEBAYORAN LAMA & SETIABUDI, karena dissolve dilakukan
     SEBELUM normalisasi nama). neighbors_map adalah dict ber-key nama, sehingga
     fragmen kedua MENIMPA yang pertama -> Kebayoran Lama & Setiabudi memakai
     daftar tetangga yang terpotong.
  2. Poligon danau ikut masuk matriks bobot spasial.

Output (semuanya ke refdata/, di-commit ke repo):
  master_kecamatan.csv  - kode kanonik + semua varian nama + centroid
  neighbors.json        - Queen contiguity, 44 kecamatan bersih
  kecamatan_44.geojson  - geometri hasil dissolve
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REFDATA = ROOT / "refdata"
SRC_GEOJSON = ROOT / "data" / "jakarta.geojson"
DINKES_RAW = REFDATA / "dinkes_kecamatan_raw.txt"

# Endpoint cascading dropdown situs surveilans Dinkes DKI.
# kp=1 -> provinsi DKI Jakarta; kk=1..6 -> kabupaten/kota.
DINKES_KEC_URL = "https://surveilans-dinkes.jakarta.go.id/sarsbaru/zz_getkec.php"


def fetch_dinkes_raw() -> None:
    """Ambil daftar kecamatan resmi dari Dinkes dan simpan apa adanya (provenance)."""
    import requests

    lines = []
    for kab in range(1, 7):
        resp = requests.get(
            DINKES_KEC_URL, params={"kp": 1, "kk": kab}, timeout=60
        )
        resp.raise_for_status()
        lines += [f"KAB{kab}|{ln}" for ln in resp.text.splitlines() if ln.strip()]
    DINKES_RAW.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  diambil dari Dinkes -> {DINKES_RAW.name} ({len(lines)} baris)")

NOT_A_KECAMATAN = {"DANAU SUNTER DLL"}

KABUPATEN_BY_KODE = {
    1: "JAKARTA PUSAT",
    2: "JAKARTA UTARA",
    3: "JAKARTA BARAT",
    4: "JAKARTA SELATAN",
    5: "JAKARTA TIMUR",
    6: "KEPULAUAN SERIBU",
}

# nama Dinkes (hasil scrape) -> nama di shapefile GADM
DINKES_TO_SHP = {
    "GROGOL PETAMBURAN": "GROGOLPETAMBURAN",
    "PULO GADUNG": "PULOGADUNG",
    "PASAR REBO": "PASARREBO",
    "TANAH ABANG": "TANAHABANG",
    "KALI DERES": "KALIDERES",
    "KEBON JERUK": "KEBONJERUK",
    "KEBUN JERUK": "KEBONJERUK",
    "TAMAN SARI": "TAMANSARI",
    "KRAMAT JATI": "KRAMATJATI",
    "KEPULAUAN SERIBU UTARA": "KEPULAUAN SERIBU UTARA",
    "KEPULAUAN SERIBU SELATAN": "KEPULAUAN SERIBU SELATAN",
    "SETIA BUDI": "SETIABUDI",
    "KABAYORAN LAMA": "KEBAYORAN LAMA",
}


def norm(name: str) -> str:
    """Normalisasi nama kecamatan ke huruf kapital tanpa spasi ganda."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s.upper().strip())
    # \b di belakang KEP wajib: tanpa itu, "KEPULAUAN" ikut cocok dan jadi
    # "KEPULAUAN ULAUAN".
    s = re.sub(r"\bKEP\b\.?\s*", "KEPULAUAN ", s)
    return re.sub(r"\s+", " ", s).strip()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_dinkes() -> pd.DataFrame:
    """Baca daftar kecamatan resmi Dinkes beserta kode kab/kec-nya."""
    rows = []
    pattern = re.compile(
        r"KAB(\d+)\|kecOptions\.push\(new Option\('([^']*)',\s*'(\d+)'\)\)"
    )
    for line in DINKES_RAW.read_text(encoding="utf-8").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        kab_kode = int(m.group(1))
        nama = m.group(2).strip()
        kec_kode = int(m.group(3))
        if kec_kode == 0 or "PILIH" in nama.upper():
            continue
        rows.append(
            {
                "kab_kode": kab_kode,
                "kec_kode_dinkes": kec_kode,
                "nama_dinkes": nama,
                "nama_norm": norm(nama),
                "kabupaten": KABUPATEN_BY_KODE[kab_kode],
            }
        )
    df = pd.DataFrame(rows)
    if len(df) != 44:
        raise RuntimeError(f"Daftar kecamatan Dinkes harus 44, dapat {len(df)}")
    return df


def load_geometry() -> gpd.GeoDataFrame:
    """Muat geojson, buang poligon non-kecamatan, satukan fragmen duplikat."""
    gdf = gpd.read_file(SRC_GEOJSON)
    gdf["nama_shp"] = gdf["kecamatan_shp"].map(norm)

    dropped = sorted(set(gdf.loc[gdf["nama_shp"].isin(NOT_A_KECAMATAN), "nama_shp"]))
    gdf = gdf[~gdf["nama_shp"].isin(NOT_A_KECAMATAN)].copy()
    print(f"  poligon non-kecamatan dibuang : {dropped}")

    before = len(gdf)
    gdf = gdf.dissolve(by="nama_shp").reset_index()[["nama_shp", "geometry"]]
    print(f"  dissolve fragmen              : {before} -> {len(gdf)} poligon")
    if len(gdf) != 44:
        raise RuntimeError(f"Setelah dissolve harus 44 poligon, dapat {len(gdf)}")
    return gdf.set_crs(epsg=4326, allow_override=True)


def queen_neighbors(gdf: gpd.GeoDataFrame) -> dict:
    """
    Queen contiguity: bertetangga bila berbagi minimal satu titik batas.
    Dihitung pada CRS metrik (UTM 48S) agar buffer toleransi bermakna dalam meter.
    """
    proj = gdf.to_crs(epsg=32748)
    # buffer 1 m menutup celah topologi kecil antar poligon GADM
    shapes = proj.geometry.buffer(1.0)
    names = gdf["nama_shp"].tolist()
    sindex = shapes.sindex

    out = {}
    for i, name in enumerate(names):
        cand = [j for j in sindex.query(shapes.iloc[i]) if j != i]
        out[name] = sorted(
            names[j] for j in cand if shapes.iloc[i].intersects(shapes.iloc[j])
        )
    return out


def main() -> None:
    REFDATA.mkdir(exist_ok=True)

    print("[1/5] Daftar kecamatan resmi Dinkes")
    if not DINKES_RAW.exists():
        fetch_dinkes_raw()
    dinkes = parse_dinkes()
    dinkes["nama_shp"] = dinkes["nama_norm"].map(lambda n: DINKES_TO_SHP.get(n, n))
    print(f"  {len(dinkes)} kecamatan, {dinkes['kab_kode'].nunique()} kabupaten/kota")

    print("[2/5] Geometri kecamatan")
    gdf = load_geometry()

    missing = set(dinkes["nama_shp"]) - set(gdf["nama_shp"])
    extra = set(gdf["nama_shp"]) - set(dinkes["nama_shp"])
    if missing or extra:
        raise RuntimeError(
            f"Nama tidak cocok. Hilang di geojson: {missing}; berlebih: {extra}"
        )
    print("  seluruh 44 nama cocok persis (tanpa fuzzy matching)")

    print("[3/5] Queen contiguity")
    nbrs = queen_neighbors(gdf)
    counts = {k: len(v) for k, v in nbrs.items()}
    isolated = sorted(k for k, v in counts.items() if v == 0)
    print(
        f"  rata-rata tetangga: {sum(counts.values()) / len(counts):.1f}"
        f" | min {min(counts.values())} | max {max(counts.values())}"
    )
    print(f"  tanpa tetangga darat: {isolated if isolated else 'tidak ada'}")

    # Simetri wajib untuk Queen murni: jika A tetangga B maka B tetangga A.
    # Dicek SEBELUM fallback, karena fallback memang sengaja berarah.
    asym = [(a, b) for a, bs in nbrs.items() for b in bs if a not in nbrs[b]]
    if asym:
        raise RuntimeError(f"Queen contiguity tidak simetris: {asym[:5]}")

    # Pulau tanpa tetangga darat -> fallback 2 centroid terdekat, SATU ARAH.
    #
    # Dipisah dari matriks Queen dengan sengaja:
    #   - `queen`   : simetris, dipakai untuk statistik spasial (Moran's I, LISA).
    #                 Pulau tetap berdaftar kosong supaya bobot spasialnya jujur.
    #   - `feature` : dipakai HANYA untuk fitur spatial_kasus_tetangga di model.
    #                 Pulau meminjam sinyal dari kecamatan pelabuhan terdekat
    #                 (jalur importasi kasus via pelayaran); relasi ini sengaja
    #                 TIDAK dibalik, karena menyuntikkan kasus ~0 Kepulauan Seribu
    #                 ke rata-rata spasial kecamatan daratan akan menariknya turun
    #                 secara keliru.
    # Notebook lama memberi pulau nilai NaN -> 0 konstan untuk kedua fitur spasial.
    proj = gdf.to_crs(epsg=32748)
    cent_proj = proj.geometry.centroid
    names = gdf["nama_shp"].tolist()
    feature_nbrs = {k: list(v) for k, v in nbrs.items()}
    fallback = {}
    for name in isolated:
        i = names.index(name)
        dist = cent_proj.distance(cent_proj.iloc[i])
        near = sorted(names[j] for j in dist.sort_values().index[1:3])
        feature_nbrs[name] = near
        fallback[name] = near
        print(f"  fallback KNN (1 arah)  {name} -> {near}")

    print("[4/5] Centroid")
    cent = cent_proj.to_crs(epsg=4326)
    geo = pd.DataFrame(
        {"nama_shp": names, "lon": cent.x.round(6), "lat": cent.y.round(6)}
    )

    master = (
        dinkes.merge(geo, on="nama_shp", how="left")
        .sort_values(["kab_kode", "nama_shp"])
        .reset_index(drop=True)
    )
    master.insert(
        0,
        "kode_kec",
        master["kab_kode"].astype(str)
        + "."
        + master["kec_kode_dinkes"].astype(str).str.zfill(2),
    )
    master["slug"] = master["nama_shp"].map(slug)
    master["n_tetangga_queen"] = master["nama_shp"].map(lambda n: len(nbrs[n]))
    master["n_tetangga_fitur"] = master["nama_shp"].map(lambda n: len(feature_nbrs[n]))
    master["tetangga_fallback_knn"] = master["nama_shp"].isin(fallback)

    print("[5/5] Tulis refdata/")
    cols = [
        "kode_kec", "slug", "nama_shp", "nama_dinkes", "kabupaten", "kab_kode",
        "kec_kode_dinkes", "lon", "lat",
        "n_tetangga_queen", "n_tetangga_fitur", "tetangga_fallback_knn",
    ]
    master[cols].to_csv(REFDATA / "master_kecamatan.csv", index=False)

    payload = {
        "_catatan": (
            "queen = Queen contiguity simetris untuk statistik spasial "
            "(Moran's I, LISA). feature = daftar yang dipakai fitur model "
            "spatial_kasus_tetangga; untuk Kepulauan Seribu memakai fallback "
            "2 centroid terdekat dan sengaja berarah satu arah."
        ),
        "queen": dict(sorted(nbrs.items())),
        "feature": {k: sorted(v) for k, v in sorted(feature_nbrs.items())},
        "fallback_knn": dict(sorted(fallback.items())),
    }
    (REFDATA / "neighbors.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    out_geo = REFDATA / "kecamatan_44.geojson"
    if out_geo.exists():
        out_geo.unlink()
    gdf.to_file(out_geo, driver="GeoJSON")

    print(f"  OK  master_kecamatan.csv ({len(master)} baris)")
    print(f"  OK  neighbors.json (queen simetris: "
          f"{sum(len(v) for v in nbrs.values())} relasi; "
          f"feature: {sum(len(v) for v in feature_nbrs.values())} relasi)")
    print("  OK  kecamatan_44.geojson")


if __name__ == "__main__":
    main()
