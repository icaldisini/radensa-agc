-- =====================================================================
-- Radensa - skema Supabase (Fase 1)
-- Jalankan di Supabase Studio > SQL Editor > New query.
--
-- Konvensi minggu di SELURUH tabel: epi week MMWR (Minggu-Sabtu) milik
-- Dinkes DKI Jakarta. BUKAN ISO week. Lihat radensa/epiweek.py.
-- Satu tahun epi bisa punya 52 atau 53 minggu (2020 dan 2025 punya 53).
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Master kecamatan (seed dari refdata/master_kecamatan.csv)
-- ---------------------------------------------------------------------
create table if not exists kecamatan (
  kode_kec         text primary key,              -- '5.01' = kab 5, kec 01
  slug             text not null unique,           -- 'duren_sawit'
  nama             text not null,                  -- 'DUREN SAWIT' (kanonik/shapefile)
  nama_dinkes      text not null,                  -- 'Duren Sawit' (persis di situs)
  kabupaten        text not null,
  kab_kode         smallint not null,
  kec_kode_dinkes  smallint not null,
  lon              double precision not null,
  lat              double precision not null,
  unique (kab_kode, kec_kode_dinkes)
);

-- Ketetanggaan. tipe='queen' simetris untuk statistik spasial (Moran's I);
-- tipe='feature' berarah, dipakai fitur spatial_kasus_tetangga di model.
create table if not exists kecamatan_tetangga (
  kode_kec      text not null references kecamatan(kode_kec) on delete cascade,
  tetangga_kode text not null references kecamatan(kode_kec) on delete cascade,
  tipe          text not null check (tipe in ('queen', 'feature')),
  primary key (kode_kec, tetangga_kode, tipe)
);

-- ---------------------------------------------------------------------
-- 2. Data primer
-- ---------------------------------------------------------------------

-- Kasus DBD hasil scraping surveilans-dinkes.jakarta.go.id
-- is_final=false selama minggu masih bisa direvisi (laporan RS telat masuk).
-- Pipeline mingguan me-scrape ulang 8 minggu terakhir dan melakukan upsert,
-- jadi revisi Dinkes otomatis tertangkap.
create table if not exists dbd_weekly (
  kode_kec    text not null references kecamatan(kode_kec) on delete cascade,
  tahun       smallint not null,
  minggu      smallint not null check (minggu between 1 and 53),
  kasus       integer not null check (kasus >= 0),
  is_final    boolean not null default false,
  scraped_at  timestamptz not null default now(),
  primary key (kode_kec, tahun, minggu)
);
create index if not exists dbd_weekly_periode_idx on dbd_weekly (tahun, minggu);

-- Iklim dari Google Earth Engine, diagregasi pada batas EPI WEEK.
-- sumber_hujan: 'chirps_v2' | 'chirps_v3' | 'era5' (urutan fallback).
create table if not exists iklim_weekly (
  kode_kec                text not null references kecamatan(kode_kec) on delete cascade,
  tahun                   smallint not null,
  minggu                  smallint not null check (minggu between 1 and 53),
  curah_hujan_kumulatif   real not null,
  curah_hujan_max_harian  real not null,
  suhu_rata               real not null,
  kelembapan_rata         real not null,
  sumber_hujan            text not null,
  -- 'observasi' untuk minggu lampau, 'gfs' untuk ramalan minggu depan,
  -- 'klimatologi' bila terpaksa memakai rata-rata historis.
  jenis                   text not null default 'observasi'
                          check (jenis in ('observasi', 'gfs', 'klimatologi')),
  ingested_at             timestamptz not null default now(),
  primary key (kode_kec, tahun, minggu)
);
create index if not exists iklim_weekly_periode_idx on iklim_weekly (tahun, minggu);

-- ---------------------------------------------------------------------
-- 3. Fitur turunan (20 fitur X, hasil radensa/features.py)
-- Disimpan supaya training dan forecast memakai angka yang sama persis
-- dan hasilnya bisa diaudit ulang.
-- ---------------------------------------------------------------------
create table if not exists features_weekly (
  kode_kec                     text not null references kecamatan(kode_kec) on delete cascade,
  tahun                        smallint not null,
  minggu                       smallint not null,
  -- iklim current
  curah_hujan_kumulatif        real,
  suhu_rata                    real,
  kelembapan_rata              real,
  -- iklim lag
  curah_hujan_lag1             real,
  curah_hujan_lag2             real,
  suhu_rata_lag1               real,
  suhu_rata_lag2               real,
  kelembapan_rata_lag1         real,
  kelembapan_rata_lag2         real,
  -- epidemiologi lag
  kasus_dbd_lag1               real,
  kasus_dbd_lag2               real,
  kasus_dbd_rolling4           real,
  kasus_dbd_rolling8           real,
  -- spasial
  spatial_kasus_tetangga_lag1  real,
  spatial_kasus_tetangga_lag2  real,
  -- musiman & interaksi
  minggu_sin                   real,
  minggu_cos                   real,
  ch_suhu_interaksi            real,
  suhu_sq                      real,
  kecamatan_endemisitas        real,
  -- target (null bila minggu belum terlapor)
  kasus_dbd                    integer,
  built_at                     timestamptz not null default now(),
  primary key (kode_kec, tahun, minggu)
);

-- ---------------------------------------------------------------------
-- 4. Model & forecast
-- ---------------------------------------------------------------------
create table if not exists model_runs (
  id                bigserial primary key,
  versi             text not null unique,          -- 'lgb-2026w44'
  algoritma         text not null default 'LightGBM (Tweedie)',
  hyperparams       jsonb not null,
  r2                real, mae real, rmse real, mape real,
  cv_detail         jsonb,                          -- metrik per fold
  feature_importance jsonb,
  data_sampai_tahun  smallint not null,
  data_sampai_minggu smallint not null,
  artifact_path     text,                           -- path di Supabase Storage
  is_active         boolean not null default false, -- hanya satu yang aktif
  created_at        timestamptz not null default now()
);
create unique index if not exists model_runs_satu_aktif
  on model_runs (is_active) where is_active;

create table if not exists forecast_runs (
  id                 bigserial primary key,
  model_versi        text not null references model_runs(versi),
  basis_tahun        smallint not null,   -- minggu observasi terakhir yang dipakai
  basis_minggu       smallint not null,
  created_at         timestamptz not null default now()
);

create table if not exists forecast (
  run_id       bigint not null references forecast_runs(id) on delete cascade,
  kode_kec     text not null references kecamatan(kode_kec) on delete cascade,
  tahun        smallint not null,
  minggu       smallint not null,
  horizon      smallint not null check (horizon between 1 and 5),
  kasus_pred   real not null,
  ci_lower_95  real,
  ci_upper_95  real,
  zona_ewars   text check (zona_ewars in ('Hijau', 'Kuning', 'Merah')),
  zona_ec      text check (zona_ec    in ('Hijau', 'Kuning', 'Merah')),
  primary key (run_id, kode_kec, tahun, minggu)
);
create index if not exists forecast_lookup_idx on forecast (kode_kec, tahun, minggu);

-- ---------------------------------------------------------------------
-- 5. Ambang zonasi
-- ---------------------------------------------------------------------
create table if not exists threshold_kecamatan (
  kode_kec     text primary key references kecamatan(kode_kec) on delete cascade,
  p50 real not null, p75 real not null, p90 real,
  mean_kasus real, std_kasus real,
  basis_tahun_awal  smallint not null,   -- periode training sumber ambang
  basis_tahun_akhir smallint not null,
  updated_at timestamptz not null default now()
);

create table if not exists endemicity_channel (
  kode_kec        text not null references kecamatan(kode_kec) on delete cascade,
  minggu          smallint not null check (minggu between 1 and 53),
  ec_mean         real not null,
  ec_std          real not null,
  ec_upper_yellow real not null,
  ec_upper_red    real not null,
  updated_at      timestamptz not null default now(),
  primary key (kode_kec, minggu)
);

-- Rekomendasi aksi per zona - inilah yang tampil di aplikasi warga.
create table if not exists rekomendasi_aksi (
  id        bigserial primary key,
  zona      text not null check (zona in ('Hijau', 'Kuning', 'Merah')),
  audiens   text not null check (audiens in ('warga', 'dinkes')),
  urutan    smallint not null default 0,
  judul     text not null,
  deskripsi text not null,
  ikon      text
);

-- ---------------------------------------------------------------------
-- 6. Audit pipeline
-- ---------------------------------------------------------------------
create table if not exists ingest_log (
  id         bigserial primary key,
  tahap      text not null,       -- 'dbd' | 'iklim' | 'features' | 'forecast' | 'retrain'
  status     text not null check (status in ('sukses', 'gagal', 'sebagian')),
  baris      integer,
  pesan      text,
  durasi_ms  integer,
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- 7. Profil pengguna aplikasi warga (dipakai Fase 8)
-- ---------------------------------------------------------------------
create table if not exists profiles (
  id         uuid primary key references auth.users(id) on delete cascade,
  nama       text,
  kode_kec   text references kecamatan(kode_kec),
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- 8. Row Level Security
-- Data surveilans agregat per kecamatan = informasi publik; dibuka untuk
-- dibaca siapa pun (termasuk anon key di web & aplikasi warga). Penulisan
-- hanya lewat service_role key milik GitHub Actions, yang memang melewati RLS.
-- ---------------------------------------------------------------------
do $$
declare t text;
begin
  foreach t in array array[
    'kecamatan', 'kecamatan_tetangga', 'dbd_weekly', 'iklim_weekly',
    'features_weekly', 'model_runs', 'forecast_runs', 'forecast',
    'threshold_kecamatan', 'endemicity_channel', 'rekomendasi_aksi'
  ] loop
    execute format('alter table %I enable row level security', t);
    execute format(
      'drop policy if exists "baca publik" on %I', t);
    execute format(
      'create policy "baca publik" on %I for select using (true)', t);
  end loop;
end $$;

-- Profil bersifat privat: tiap pengguna hanya boleh menyentuh barisnya sendiri.
alter table profiles enable row level security;
drop policy if exists "profil sendiri" on profiles;
create policy "profil sendiri" on profiles
  for all using (auth.uid() = id) with check (auth.uid() = id);

alter table ingest_log enable row level security;  -- tanpa policy = tertutup

-- ---------------------------------------------------------------------
-- 9. View untuk aplikasi warga: forecast aktif terbaru per kecamatan
-- ---------------------------------------------------------------------
create or replace view forecast_terkini as
select k.kode_kec, k.nama, k.nama_dinkes, k.kabupaten, k.lon, k.lat,
       f.tahun, f.minggu, f.horizon,
       f.kasus_pred, f.ci_lower_95, f.ci_upper_95,
       f.zona_ewars, f.zona_ec,
       fr.created_at as dihitung_pada, fr.model_versi
from forecast f
join forecast_runs fr on fr.id = f.run_id
join kecamatan k on k.kode_kec = f.kode_kec
where fr.id = (select id from forecast_runs order by created_at desc limit 1);
