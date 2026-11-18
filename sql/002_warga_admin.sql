-- =====================================================================
-- Radensa - Fase 8: dukungan Radensa_Warga (apk HP) & Radensa_Admin
-- Jalankan di Supabase Studio > SQL Editor SETELAH 001_schema.sql.
-- Aman dijalankan ulang (idempotent).
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Profil warga dibuat otomatis saat mendaftar.
-- Konfirmasi email aktif di project ini, jadi saat signUp belum ada sesi
-- dan klien tidak bisa insert ke profiles (RLS). Trigger ini mengisi
-- profil dari metadata pendaftaran (nama, kode_kec) di sisi database.
-- kode_kec yang tidak valid jadi NULL, bukan menggagalkan pendaftaran.
-- ---------------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, nama, kode_kec)
  values (
    new.id,
    nullif(trim(new.raw_user_meta_data ->> 'nama'), ''),
    (select k.kode_kec from public.kecamatan k
      where k.kode_kec = new.raw_user_meta_data ->> 'kode_kec')
  )
  on conflict (id) do nothing;
  return new;
end $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------------
-- 2. Statistik agregat warga terdaftar per kecamatan (dashboard admin).
-- Hanya jumlah per kecamatan; tidak ada nama/email yang keluar.
-- ---------------------------------------------------------------------
create or replace function public.statistik_warga()
returns table (kode_kec text, jumlah bigint)
language sql
stable
security definer
set search_path = public
as $$
  select p.kode_kec, count(*)::bigint
  from public.profiles p
  where p.kode_kec is not null
  group by p.kode_kec;
$$;

revoke all on function public.statistik_warga() from public;
grant execute on function public.statistik_warga() to anon, authenticated;

-- ---------------------------------------------------------------------
-- 3. Rekomendasi aksi per zona (diisi sekali, hanya bila tabel kosong).
-- Teks warga ditulis dengan bahasa sehari-hari; teks dinkes operasional.
-- ---------------------------------------------------------------------
insert into public.rekomendasi_aksi (zona, audiens, urutan, judul, deskripsi, ikon)
select v.zona, v.audiens, v.urutan, v.judul, v.deskripsi, v.ikon
from (values
  ('Hijau',  'warga', 1, 'PSN 3M Plus seminggu sekali',
   'Kuras bak mandi, tutup tempat air, dan daur ulang barang bekas yang bisa menampung air.', 'droplet'),
  ('Hijau',  'warga', 2, 'Cek jentik di rumah',
   'Lihat bak mandi, vas bunga, tatakan dispenser, dan talang air. Jentik terlihat seperti cacing kecil yang bergerak.', 'search'),
  ('Hijau',  'warga', 3, 'Pantau info tiap minggu',
   'Risiko bisa berubah cepat saat musim hujan. Buka Radensa seminggu sekali.', 'calendar'),

  ('Kuning', 'warga', 1, 'PSN 3M Plus dua kali seminggu',
   'Nyamuk berkembang dari telur jadi dewasa dalam 7-10 hari. Menguras lebih sering memutus siklusnya.', 'droplet'),
  ('Kuning', 'warga', 2, 'Pakai losion antinyamuk',
   'Nyamuk DBD paling aktif pagi (08.00-10.00) dan sore (15.00-17.00). Lindungi terutama anak-anak.', 'shield'),
  ('Kuning', 'warga', 3, 'Ajak tetangga kerja bakti',
   'Nyamuk bisa terbang dari rumah tetangga. Bersihkan lingkungan RT bersama-sama.', 'users'),
  ('Kuning', 'warga', 4, 'Demam lebih dari 2 hari? Periksa',
   'Datang ke Puskesmas untuk tes darah. Jangan menunggu sampai lemas.', 'thermometer'),

  ('Merah',  'warga', 1, 'PSN serentak bersama RT/RW',
   'Kasus di kecamatanmu diperkirakan tinggi. Lakukan pembersihan sarang nyamuk serentak minggu ini.', 'users'),
  ('Merah',  'warga', 2, 'Lindungi anak dan lansia',
   'Gunakan kelambu saat tidur siang, baju lengan panjang, dan losion antinyamuk.', 'shield'),
  ('Merah',  'warga', 3, 'Demam tinggi mendadak? Segera ke Puskesmas',
   'Jangan tunggu. Penanganan cepat mencegah DBD menjadi berat.', 'thermometer'),
  ('Merah',  'warga', 4, 'Laporkan jentik dan kasus',
   'Hubungi kader Jumantik atau Puskesmas bila menemukan jentik atau ada tetangga sakit DBD.', 'megaphone'),

  ('Hijau',  'dinkes', 1, 'Surveilans rutin',
   'Pemeriksaan Jentik Berkala (PJB) dan pemantauan laporan RS/Puskesmas mingguan.', 'activity'),
  ('Hijau',  'dinkes', 2, 'Edukasi PSN 3M Plus',
   'Pertahankan gerakan 1 Rumah 1 Jumantik dan sosialisasi melalui sekolah.', 'megaphone'),

  ('Kuning', 'dinkes', 1, 'Intensifkan PJB',
   'Targetkan Angka Bebas Jentik (ABJ) >= 95%. Prioritaskan kelurahan dengan kasus terbanyak.', 'search'),
  ('Kuning', 'dinkes', 2, 'Siapkan logistik',
   'Pastikan stok larvasida, RDT NS1, dan cairan infus di Puskesmas kecamatan.', 'package'),
  ('Kuning', 'dinkes', 3, 'Komunikasi risiko',
   'Koordinasi lurah, camat, dan sekolah; sebarkan informasi waspada ke warga.', 'megaphone'),

  ('Merah',  'dinkes', 1, 'Penyelidikan Epidemiologi (PE)',
   'Lakukan PE untuk setiap kasus dalam 1x24 jam; tentukan perlu tidaknya fogging fokus.', 'activity'),
  ('Merah',  'dinkes', 2, 'PSN serentak & fogging fokus',
   'Gerakkan PSN serentak se-kecamatan; fogging hanya berdasarkan hasil PE.', 'users'),
  ('Merah',  'dinkes', 3, 'Siaga fasilitas rujukan',
   'Koordinasi RS rujukan untuk kesiapan tempat tidur dan trombosit.', 'hospital')
) as v(zona, audiens, urutan, judul, deskripsi, ikon)
where not exists (select 1 from public.rekomendasi_aksi);
