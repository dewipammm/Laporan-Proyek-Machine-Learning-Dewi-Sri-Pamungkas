# Laporan Proyek Machine Learning-Dewi Sri Pamungkas

### Project Overview

Domain yang saya pilih adalah dataset dari kaggle yang berjudul Netflix Movies and TV Shows

### Latar Belakang

Netflix adalah salah satu platform streaming media dan video paling populer. Mereka memiliki lebih dari 8000 film atau acara tv yang tersedia di platform mereka, pada pertengahan 2021, mereka memiliki lebih dari 200 juta Pelanggan secara global. Dataset tabular ini terdiri dari daftar semua film dan acara tv yang tersedia di Netflix, bersama dengan detail seperti - pemeran, sutradara, peringkat, tahun rilis, durasi, dll. 

Oleh karena itulah saya mengambil dataset ini untuk proyek akhir machine learning terapan menggunakan sistem rekomendasi.

### Business Understanding

#### Problem Statements

* Bagaimana cara membuat model *Machine Learning* untuk merekomendasikan judul Netflix kepada pengguna lain dengan teknik *content-based filtering*?
* Bagaimana mengetahui tingkat akurasi model menggunakan *Metrik Precision* berdasarkan hasil rekomendasi yang diberikan?

#### Goals

* Membuat model *Machine Learning* untuk menghasilkan sejumlah rekomendasi judul netflix kepada pengguna lain menggunakan teknik *content-based filtering*.
* Mengetahui tingkat akurasi model menggunakan *metric Precision* berdasarkan hasil rekomendasi yang diberikan.

#### Solution Statements

Untuk menyelesaikan masalah ini, saya menggunakan teknik *content-based filtering*. Berikut adalah penjelasan teknik yang akan digunakan untuk masalah ini :

* *Content-Based Filtering* : Merupakan cara untuk memberi rekomendasi berdasarkan fitur pada item yang disukai oleh pengguna. *Content-based filtering* mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna.
* Kelebihan sistem rekomendasi dengan pendekatan *content-based filtering* memiliki kemampuan untuk merekomendasikan item (contoh: film, lagu, artikel dll) yang sifatnya baru bagi user, karena prinsip kerjanya yaitu dengan melihat diskripsi konten yang dikandung oleh item yang pernah diberi nilai rating tinggi sebelumnya oleh pengguna.
* Kelemahan *Content-Based Filtering* sulit untuk menghasilkan rekomendasi yang tidak terduga, karena semua informasi dipilih dan direkomendasikan berdasarkan konten dan teknik ini tidak *reliable* (handal) tidak adanya ide, pendapat serta masukkan dari pengguna sebelumya yang sulit dianalis oleh komputer sehingga sulit untuk pengguna pemula memakai sistem ini secara efektif.

### Data Understanding

Dataset yang digunakan pada proyek ini diambil dari website *kaggle* dengan Netflix Movies and TV Shows.

<https://www.kaggle.com/datasets/shivamb/netflix-shows>

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

Data Loading sebagai berikut : 

<https://colab.research.google.com/drive/1XmSn_LqZjT8ATgF6yy1YUan3aUYcpfH6#scrollTo=QPlA-z7Pdh0t&line=1&uniqifier=1>

Berdasarkan output di atas, dapat diketahui bahwa file netflix_titles.csv memiliki 8807 entries. Dapat disimpulkan bahwa nilai rating bermacam-macam, yaitu: 

* PG-13 : Orangtua perlu sangat berhati-hati, beberapa materi mungkin tak pantas untuk anak di bawah 13 tahun
* TV-MA : Rating yang menunjukkan bahwa sebuah program ditujukan untuk orang dewasa
* R : Anak-anak berusia di bawah 17 tahun membutuhkan pendampingan orangtua atau orang dewasa saat menyaksikannya
* TV-Y7 : Untuk kalangan berusia 7-16 tahun
* PG : Parental Guidance Suggested yaitu perlu bimbingan orangtua
* TV-14 : untuk kalangan berusia 14 keatas jika dibawah 14 tahun maka dalam pengawasan ketat orang tua

Pada tahap ini, saya membersihkan data NaN. Jumlah NaN 4307 dari NaN yg ditemukan berdasarkan kolom :

jumlah NaN tiap Column
show_id            0
type               0
title              0
director        2634
cast             825
country          831
date_added        10
release_year       0
rating             4
duration           3
listed_in          0
description        0
dtype: int64

Ini adalah informasi yang didapatkan dari hasil eksplorasi pada variabel netflix.

#### Visualization Data

![image](https://user-images.githubusercontent.com/110523200/191176144-e9bc0bc1-d2d5-42f5-9de7-72ac36209848.png)

Insight yang saya dapatkan disni adalah:
Netflix Type "Movie" yang paling banyak dibandingkan Netflix Type "TV Show"

### Data Preparation

Sebelum membuat model, perlunya melakukan pada data preparation adalah menduplikasi variabel dan juga text cleaning agar dapat memberikan hasil rekomendasi yang baik

* Duplikasi variabel dataset melakukan duplikasi pada variabel netflix lalu data duplikasi ditampung pada variabel netflix_data sehingga dataset pada variabel netflix yang menampung dataset induk tidak terkontaminasi dan bisa digunakan kembali jika saya ingin mengembangkan model rekomendasi.
* *Text Cleaning* dilakukan pada kolom *title* untuk menghilangkan simbol atau teks yang tidak diperlukan dengan cara menggunakan teknik *Regex* agar membuat function yang bernama *text cleaning* dan mengaplikasikannya pada netflix_data.

### Modeling and Result

Pada Proyek yang dibuat, tahapan modelling yang digunakan dalam teknik sistem rekomendasi *Content Based Filtering*. Karena dapat merekomendasikan pengguna berdasarkan konten rating yang didapat dari film. Jadi saya membuat acuannya berdasarkan rating.

#### Content-based Filtering

* Saya menggunakan TF-IDF Vectorizer untuk menemukan representasi fitur penting dari setiap rating netflix. Fungsi yang saya gunakan adalah tfidfvectorizer() dari library sklearn. Berikut sebagian outputnya :

(https://colab.research.google.com/drive/1XmSn_LqZjT8ATgF6yy1YUan3aUYcpfH6#scrollTo=fXLVQ_ZYx5S8&line=1&uniqifier=1)

* Selanjutnya saya menetapkan 1 sebagai tanda judul netflix yang direkomendasikan dan 0 sebagai judul netflix yang tidak direkomendasikan. Dengan begitu saya menggunakan kernel sigmoid karena paling cocok untuk hasil binary. Berikut outputnya :
        
(https://colab.research.google.com/drive/1XmSn_LqZjT8ATgF6yy1YUan3aUYcpfH6#scrollTo=Nw8yKJjqzJxc&line=1&uniqifier=1)

* Selanjutnya saya menggunakan argpartition untuk mengambil sejumlah nilai k tertinggi dari similarity data kemudian mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah.

Kemudian saya menguji akurasi dari sistem rekomendasi ini untuk menemukan rekomendasi netflix yang mirip dengan judul "A Cinderella Story". Berikut adalah detail informasi judul netflix "A Cinderella Story" :

(https://colab.research.google.com/drive/1XmSn_LqZjT8ATgF6yy1YUan3aUYcpfH6#scrollTo=vdUjh7bZ5vKi&line=1&uniqifier=1)

Berdasarkan output di atas, dapat dilihat bahwa netflix dengan judul "A Cinderella Story" memiliki rating "PG" dengan negara asal United States, Canada. Rekomendasi yang diharapkan adalah judul netflix dengan rating yang serupa.

Berikut adalah rekomendasi yang diberikan oleh model yang telah dibuat :

(https://colab.research.google.com/drive/1XmSn_LqZjT8ATgF6yy1YUan3aUYcpfH6#scrollTo=4CGhQO0_6Edu&line=1&uniqifier=1)

Model berhasil memberikan rekomendasi 10 judul netflix dengan rating yang serupa.

### Evaluation

Pada tahap ini, saya menggunakan metriks precision. Precision Adalah sebuah metrics yang digunakan untuk mengukur berapa jumlah prediksi benar yang telah dibuat. Kelebihannya yaitu sangat baik untuk klasifikasi, dokumen yang dipilih secara acak dari kumpulan dokumen yang diambil adalah relevan, precision bagus untuk kasus di mana kelasnya seimbang. Namun kekurangan dari metrik precision ini yaitu tidak baik untuk data yang Imbalance dan hanya hasil teratas yang dikembalikan oleh sistem.

Untuk mengevaluasi model adalah menampung terlebih dahulu data netflix yang akan menjadi data uji coba, dalam kasus ini saya mencoba untuk menampung data netflix yang mempunyai judul "A Cinderella Story" dan saya tampung pada variabel feature. Lalu selanjutnya saya menampung rating yang ada pada data uji coba untuk selanjutnya dipakai untuk evaluasi model.

Dan langkah terakhir yang saya lakukan adalah membuat perulangan berdasarkan rating pada data uji coba dan melakukan implementasi dari formula precision. Berikut adalah hasil keluaran dari implementasi formula precision : 

(https://colab.research.google.com/drive/1XmSn_LqZjT8ATgF6yy1YUan3aUYcpfH6#scrollTo=nT_ELbAY6Ea8&line=1&uniqifier=1)

Output tersebut memberikan hasil yang cukup baik dan memiliki akurasi sebesar 100% sehingga dari sini saya bisa mengetahui bahwa model yang saya kembangkan berjalan sesuai yang diharapkan. 
 
### Kesimpulan

Saya sudah cukup paham bagaimana cara menyelesaikan proyek ini. Namun masih banyak pula hal yang perlu saya perbaiki dan pelajari agar saya paham sepenuhnya.
