# Laporan Proyek Machine Learning - Dewi Sri Pamungkas

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

![image](https://user-images.githubusercontent.com/110523200/191248931-a6ac9c5f-9e47-41dd-b65d-52bf3eedeab3.png)

Berdasarkan output di atas, dapat diketahui bahwa file netflix_titles.csv memiliki 8807 entries. Dapat disimpulkan bahwa nilai rating bermacam-macam, yaitu: 

* PG-13 : Orangtua perlu sangat berhati-hati, beberapa materi mungkin tak pantas untuk anak di bawah 13 tahun
* TV-MA : Rating yang menunjukkan bahwa sebuah program ditujukan untuk orang dewasa
* R : Anak-anak berusia di bawah 17 tahun membutuhkan pendampingan orangtua atau orang dewasa saat menyaksikannya
* TV-Y7 : Untuk kalangan berusia 7-16 tahun
* PG : *Parental Guidance Suggested* yaitu perlu bimbingan orangtua
* TV-14 : untuk kalangan berusia 14 keatas jika dibawah 14 tahun maka dalam pengawasan ketat orang tua

Pada tahap ini, saya membersihkan data NaN. Jumlah NaN 4307 dari NaN yg ditemukan berdasarkan kolom :

![image](https://user-images.githubusercontent.com/110523200/191249055-109fc2cc-da2c-4d76-81e3-cb2c005cacc3.png)

Ini adalah informasi yang didapatkan dari hasil eksplorasi pada variabel netflix.

#### Visualization Data

![image](https://user-images.githubusercontent.com/110523200/191176144-e9bc0bc1-d2d5-42f5-9de7-72ac36209848.png)

Insight yang saya dapatkan disni adalah:
Netflix Type "Movie" yang paling banyak dibandingkan Netflix Type "TV Show"

### Data Preparation

Sebelum membuat model, perlunya melakukan pada data preparation adalah menduplikasi variabel dan juga text cleaning agar dapat memberikan hasil rekomendasi yang baik

* Melakukan duplikasi pada variabel netflix, fungsi yang saya gunakan adalah copy(). Lalu data duplikasi ditampung pada variabel netflix_data sehingga dataset pada variabel netflix yang menampung dataset induk tidak terkontaminasi dan bisa digunakan kembali jika saya ingin mengembangkan model rekomendasi.
* *Data Cleaning* dilakukan menggunakan fungsi data.isna().any() untuk mendeteksi adanya error atau corrupt pada data. Karena data memiliki NaN maka saya membuang data yang mempunyai NaN
* *Text Cleaning* dilakukan pada kolom *title* untuk menghilangkan simbol atau teks yang tidak diperlukan dengan cara menggunakan teknik *Regex* agar membuat function yang bernama *text cleaning* dan mengaplikasikannya pada netflix_data.

### Modeling and Result

Pada Proyek yang dibuat, tahapan modelling yang digunakan dalam teknik sistem rekomendasi *Content Based Filtering*. Karena dapat merekomendasikan pengguna berdasarkan konten rating yang didapat dari film. Jadi saya membuat acuannya berdasarkan rating.

#### Content-based Filtering

* Saya menggunakan TF-IDF Vectorizer untuk menemukan representasi fitur penting dari setiap rating netflix. Fungsi yang saya gunakan adalah tfidfvectorizer() dari library sklearn. Berikut sebagian outputnya :

![image](https://user-images.githubusercontent.com/110523200/191249618-9f749a40-a1c5-435f-a420-9029f8271819.png)

* Selanjutnya saya menetapkan 1 sebagai tanda judul netflix yang direkomendasikan dan 0 sebagai judul netflix yang tidak direkomendasikan. Dengan begitu saya menggunakan kernel sigmoid karena paling cocok untuk hasil *binary*. Berikut outputnya :
        
![image](https://user-images.githubusercontent.com/110523200/191249781-42ac3757-1b4b-4f38-a188-311b1968cef6.png)

* Selanjutnya saya menggunakan *argpartition* untuk mengambil sejumlah nilai k tertinggi dari *similarity* data kemudian mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah.

Kemudian saya menguji akurasi dari sistem rekomendasi ini untuk menemukan rekomendasi netflix yang mirip dengan judul "A Cinderella Story". Berikut adalah detail informasi judul netflix "A Cinderella Story" :

![image](https://user-images.githubusercontent.com/110523200/191250230-66c3cc87-4eba-47f0-a7e0-203b194813df.png)

Berdasarkan output di atas, dapat dilihat bahwa netflix dengan judul "A Cinderella Story" memiliki rating "PG" dengan negara asal United States, Canada. Rekomendasi yang diharapkan adalah judul netflix dengan rating yang serupa.

Berikut adalah rekomendasi yang diberikan oleh model yang telah dibuat :

![image](https://user-images.githubusercontent.com/110523200/191250106-17da077c-f6cf-417b-abf2-024120ef0c88.png)

Model berhasil memberikan rekomendasi 10 judul netflix dengan rating yang serupa.

### Evaluation

Pada tahap ini, saya menggunakan metriks *precision*. *Precision* adalah sebuah metrics yang digunakan untuk mengukur berapa jumlah prediksi benar yang telah dibuat. Kelebihannya yaitu sangat baik untuk klasifikasi, dokumen yang dipilih secara acak dari kumpulan dokumen yang diambil adalah relevan, *precision* bagus untuk kasus di mana kelasnya seimbang. Namun kekurangan dari metrik *precision* ini yaitu tidak baik untuk data yang Imbalance dan hanya hasil teratas yang dikembalikan oleh sistem.

Untuk mengevaluasi model adalah menampung terlebih dahulu data netflix yang akan menjadi data uji coba, dalam kasus ini saya mencoba untuk menampung data netflix yang mempunyai judul "A Cinderella Story" dan saya tampung pada variabel feature. Lalu selanjutnya saya menampung rating yang ada pada data uji coba untuk selanjutnya dipakai untuk evaluasi model.

Dan langkah terakhir yang saya lakukan adalah membuat perulangan berdasarkan rating pada data uji coba dan melakukan implementasi dari formula *precision*. Berikut adalah hasil keluaran dari implementasi formula precision : 

![image](https://user-images.githubusercontent.com/110523200/191246646-4c9b5285-0a2c-4fc7-a268-50e4a48f4618.png)

Output tersebut memberikan hasil yang cukup baik dan memiliki akurasi sebesar 100% sehingga dari sini saya bisa mengetahui bahwa model yang saya kembangkan berjalan sesuai yang diharapkan. 
 
### Kesimpulan

Saya belum sepenuhnya paham bagaimana cara menyelesaikan proyek sistem rekomendasi secara baik dan benar, namun sedikit banyaknya saya mengerti apa yang harus saya lakukan untuk mengerjakan proyek ini. Dan masih banyak pula hal yang perlu saya perbaiki dan pelajari agar saya paham sepenuhnya.
