# Panduan Import Database SQL (Book Recommendation System)

Folder ini berisi skema DDL dan data dump SQL yang telah dibersihkan dan diformat dengan standar ANSI SQL (kompatibel dengan **MySQL**, **PostgreSQL**, dan **SQLite**).

---

## 📁 File yang Tersedia

1. **`01_schema.sql`** (Skema Tabel & Indeks):
   - Tabel `users` (`user_id`, `location`, `age`)
   - Tabel `books` (`isbn`, `book_title`, `book_author`, `year_of_publication`, `publisher`, `image_url_s`, `image_url_m`, `image_url_l`)
   - Tabel `ratings` (`user_id`, `isbn`, `book_rating`)
   - Primary Keys, Foreign Key considerations, dan Indeks pencarian/filter.
2. **`02_insert_users.sql`** (278.858 baris data pengguna)
3. **`03_insert_books.sql`** (271.360 baris data buku, sudah diperbaiki data baris yang korup/bergeser)
4. **`04_insert_ratings.sql`** (1.149.780 baris data interaksi rating)

---

## 🚀 Cara Import ke MySQL

### Langkah 1: Buat Database di MySQL
Buka MySQL CLI atau phpMyAdmin / DBeaver:
```sql
CREATE DATABASE book_recommendation CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE book_recommendation;
```

### Langkah 2: Jalankan Script SQL via Terminal / Command Prompt
Masuk ke folder `SQL` lalu jalankan perintah berikut secara berurutan:

```bash
# 1. Eksekusi Skema Tabel
mysql -u root -p book_recommendation < 01_schema.sql

# 2. Import Data Users
mysql -u root -p book_recommendation < 02_insert_users.sql

# 3. Import Data Books
mysql -u root -p book_recommendation < 03_insert_books.sql

# 4. Import Data Ratings
mysql -u root -p book_recommendation < 04_insert_ratings.sql
```

> **💡 Tips Performa MySQL:**
> Untuk dataset berukuran besar seperti `ratings` (1,14 juta baris), pastikan pengaturan buffer MySQL Anda memadai:
> ```sql
> SET GLOBAL max_allowed_packet = 1073741824; -- 1 GB
> SET GLOBAL net_buffer_length = 1000000;
> ```

---

## 🐘 Cara Import ke PostgreSQL

### Langkah 1: Buat Database di PostgreSQL
```bash
createdb -U postgres book_recommendation
```

### Langkah 2: Eksekusi File SQL via `psql`
```bash
# 1. Eksekusi Skema Tabel
psql -U postgres -d book_recommendation -f 01_schema.sql

# 2. Nonaktifkan foreign key checks saat bulk insert (opsional jika ada strict FK):
# psql -U postgres -d book_recommendation -c "SET session_replication_role = 'replica';"

# 3. Import Data
psql -U postgres -d book_recommendation -f 02_insert_users.sql
psql -U postgres -d book_recommendation -f 03_insert_books.sql
psql -U postgres -d book_recommendation -f 04_insert_ratings.sql
```

---

## ⚡ Opsi Cepat: SQLite (Tanpa Server / Standalone)
Jika Anda ingin langsung menggunakan database lokal `.db` untuk analisis di Python atau DBeaver:

```bash
# Jalankan di terminal/powershell:
python -c "
import sqlite3, glob
conn = sqlite3.connect('books.db')
for f in ['01_schema.sql', '02_insert_users.sql', '03_insert_books.sql', '04_insert_ratings.sql']:
    print(f'Executing {f}...')
    with open(f, 'r', encoding='utf-8') as sql_file:
        conn.executescript(sql_file.read())
print('Done! books.db is ready.')
"
```
