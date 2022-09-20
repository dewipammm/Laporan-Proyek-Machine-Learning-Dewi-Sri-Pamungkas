# Laporan-Proyek-Machine-Learning-Dewi-Sri-Pamungkas

### Project Overview

Domain yang saya pilih adalah dataset dari kaggle yang berjudul Netflix Movies and TV Shows

### Latar Belakang

Netflix adalah salah satu platform streaming media dan video paling populer. Mereka memiliki lebih dari 8000 film atau acara tv yang tersedia di platform mereka, pada pertengahan 2021, mereka memiliki lebih dari 200 juta Pelanggan secara global. Dataset tabular ini terdiri dari daftar semua film dan acara tv yang tersedia di Netflix, bersama dengan detail seperti - pemeran, sutradara, peringkat, tahun rilis, durasi, dll. 

Oleh karena itulah saya mengambil dataset ini untuk proyek akhir machine learning terapan menggunakan sistem rekomendasi.

### Business Understanding

#### Problem Statements

* Bagaimana cara membuat model Machine Learning untuk merekomendasikan judul Netflix kepada pengguna lain dengan teknik content-based filtering?

#### Goals

* Membuat model Machine Learning untuk menghasilkan sejumlah rekomendasi judul netflix kepada pengguna lain menggunakan teknik content-based filtering.
* Mengetahui tingkat akurasi model menggunakan metric Precision berdasarkan hasil rekomendasi yang diberikan.

#### Solution Statements

Untuk menyelesaikan masalah ini, saya menggunakan teknik content-based filtering. Berikut adalah penjelasan teknik yang akan digunakan untuk masalah ini :

* Content-Based Filtering : Merupakan cara untuk memberi rekomendasi berdasarkan fitur pada item yang disukai oleh pengguna. Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna.
* Kelebihan sistem rekomendasi dengan pendekatan content-based filtering memiliki kemampuan untuk merekomendasikan item (contoh: film, lagu, artikel dll) yang sifatnya baru bagi user, karena prinsip kerjanya yaitu dengan melihat diskripsi konten yang dikandung oleh item yang pernah diberi nilai rating tinggi sebelumnya oleh pengguna.
* Kelemahan Content-Based Filtering sulit untuk menghasilkan rekomendasi yang tidak terduga, karena semua informasi dipilih dan direkomendasikan berdasarkan konten dan teknik ini tidak reliable (handal) tidak adanya ide, pendapat serta masukkan dari pengguna sebelumya yang sulit dianalis oleh komputer sehingga sulit untuk pengguna pemula memakai sistem ini secara efektif.

### Data Understanding

Dataset yang digunakan pada proyek ini diambil dari website kaggle dengan Netflix Movies and TV Shows.

[https://www.kaggle.com/datasets/shivamb/netflix-shows]

Dataset yang dipakai adalah Netflix Movies and TV Shows yang memiliki 8807 baris dan 12 kolom. Adapun penjelasan dari masing-masing kolom dari dataset tersebut :
* show_id : ID unik untuk setiap Film/Acara TV
* type : Tanda Pengenal-Film atau Acara TV
* title : Judul Film/Acara TV
* director : Sutradara Film
* cast : Aktor yang terlibat dalam film/pertunjukan
* country : Negara tempat film/acara itu diproduksi
* date_added : Tanggal ditambahkan di Netflix
* release_year : Tahun Rilis aktual dari pergerakan / pertunjukan
* rating : Peringkat TV dari film / acara
* duration : Durasi Total dalam menit atau jumlah musim


