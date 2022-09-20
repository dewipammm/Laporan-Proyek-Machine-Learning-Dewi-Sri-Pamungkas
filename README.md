# Laporan Proyek Machine Learning - Dewi Sri Pamungkas

### Project Overview

Domain yang saya pilih adalah dataset dari kaggle yang berjudul Netflix Movies and TV Shows

### Latar Belakang

Netflix adalah salah satu platform streaming media dan video paling populer. Mereka memiliki lebih dari 8000 film atau acara tv yang tersedia di platform mereka, pada pertengahan 2021, mereka memiliki lebih dari 200 juta Pelanggan secara global. Dataset tabular ini terdiri dari daftar semua film dan acara tv yang tersedia di Netflix, bersama dengan detail seperti - pemeran, sutradara, peringkat, tahun rilis, durasi, dll. 

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

| # |   Column   |  Non-Null  Count | Dtype   |                 
|---|------------|------------------|---------|
| 0 |show_id     |  8807 non-null   | object  |   
| 1 |type        |  8807 non-null   | object  |   
| 2 |title       |  8807 non-null   | object  |   
| 3 |director    |  6173 non-null   | object  |   
| 4 |cast        |  7982 non-null   | object  |   
| 5 |country     |  7976 non-null   | object  |   
| 6 |date_added  |  8797 non-null   | object  |   
| 7 |release_year|  8807 non-null   | int64   |   
| 8 |rating      |  8803 non-null   | object  |   
| 9 |duration    |  8804 non-null   | object  |   
|10 |listed_in   |  8807 non-null   | object  |   
|11 |description |  8807 non-null   | object  |  

|       | release_year |
|-------|--------------|
| count | 8807.000000  | 
| mean  | 2014.180198  | 
|  std  |    8.819312  | 
|  min  | 1925.000000  | 
|  25%  | 2013.000000  | 
|  50%  | 2017.000000  |
|  75%  | 2019.000000  | 
|  max  | 2021.000000  | 

Berdasarkan output di atas, dapat diketahui bahwa file netflix_titles.csv memiliki 8807 entries. Dapat disimpulkan bahwa nilai rating bermacam-macam, yaitu: 

* PG-13 : Orangtua perlu sangat berhati-hati, beberapa materi mungkin tak pantas untuk anak di bawah 13 tahun
* TV-MA : Rating yang menunjukkan bahwa sebuah program ditujukan untuk orang dewasa
* R : Anak-anak berusia di bawah 17 tahun membutuhkan pendampingan orangtua atau orang dewasa saat menyaksikannya
* TV-Y7 : Untuk kalangan berusia 7-16 tahun
* PG : *Parental Guidance Suggested* yaitu perlu bimbingan orangtua
* TV-14 : untuk kalangan berusia 14 keatas jika dibawah 14 tahun maka dalam pengawasan ketat orang tua

Jumlah NaN 4307 dari NaN yg ditemukan. Berikut jumlah NaN tiap Column :

| show_id | type | title | director | cast | country | date_added | release_year | rating | duration | listed_in | description |
|---------|------|-------|----------|------|---------|------------|--------------|--------|----------|-----------|-------------|
|    0    |  0   |   0   |   2634   | 825  |   831   |     10     |      0       |    4   |    3     |     0     |      0      |

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

|   7   |   8   |   9   |  12   |  24   | ... |  8801 | 8802 | 8804 | 8805 |  8806 |
|-------|-------|-------|-------|-------|-----|-------|------|------|------|-------|
| TV-MA | TV-14 | PG-13 | TV-MA | TV-14 | ... | TV-MA |  R   |  R   |  PG  | TV-14 | 

* Selanjutnya saya menetapkan 1 sebagai tanda judul netflix yang direkomendasikan dan 0 sebagai judul netflix yang tidak direkomendasikan. Dengan begitu saya menggunakan kernel sigmoid karena paling cocok untuk hasil *binary*. 

* Selanjutnya saya menggunakan *argpartition* untuk mengambil sejumlah nilai k tertinggi dari *similarity* data kemudian mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah.

Kemudian saya menguji akurasi dari sistem rekomendasi ini untuk menemukan rekomendasi netflix yang mirip dengan judul "A Cinderella Story". Berikut adalah detail informasi judul netflix "A Cinderella Story" :

|     | show_id |  type  |       title        |  director   |                       cast                         | 
|-----|---------|--------|--------------------|-------------|----------------------------------------------------|
| 127 |  s128   |  Movie | A Cinderella Story | Mark Rosman | Hilary Duff, Chad Michael Murray, Jennifer Coo...  | 
 
|        country        |  date_added  | release_year | rating | duration |             listed_in              |                   description                     |
|-----------------------|--------------|--------------|--------|----------|------------------------------------|---------------------------------------------------|
| United States, Canada | Sept 1, 2021 |     2004     |   PG   |  95 min  | Children & Family Movies, Comedies | Teen Sam meets the boy of her dreams at a danc... |

Berdasarkan output di atas, dapat dilihat bahwa netflix dengan judul "A Cinderella Story" memiliki rating "PG" dengan negara asal United States, Canada. Rekomendasi yang diharapkan adalah judul netflix dengan rating yang serupa.

Berikut adalah rekomendasi yang diberikan oleh model yang telah dibuat :

|   |        Netflix Title        | Rating |	  	         Country                      |
|---|-----------------------------|--------|--------------------------------------------------|
| 0 | Safe House                  |    R   | South Africa, United States, Japan               |
| 1 | Training Day                |    R   | United States                                    |
| 2 | Kate	                  |    R   | United States                                    |
| 3 | In the Cut                  |    R   | United Kingdom, Australia, France, United States |
| 4 | Blade Runner: The Final Cut |    R   | United States                                    |
| 5 | Chappie                     |    R   | South Africa, United States                      |
| 6 | Cliffhanger	          |    R   | United States, Italy, France, Japan              |
| 7 | Cold Mountain	          |    R   | United States, Italy, Romania, United Kingdom    |
| 8 | Do the Right Thing          |    R   | United States                                    |
| 9 | Extraction	          |    R   | United States, United Kingdom, Canada            |
|10 | House Party                 |    R   | United States                                    |

Model berhasil memberikan rekomendasi 10 judul netflix dengan rating yang serupa.

### Evaluation

Pada tahap ini, saya menggunakan metriks *precision*. *Precision* adalah sebuah metrics yang digunakan untuk mengukur berapa jumlah prediksi benar yang telah dibuat. Kelebihannya yaitu sangat baik untuk klasifikasi, dokumen yang dipilih secara acak dari kumpulan dokumen yang diambil adalah relevan, *precision* bagus untuk kasus di mana kelasnya seimbang. Namun kekurangan dari metrik *precision* ini yaitu tidak baik untuk data yang Imbalance dan hanya hasil teratas yang dikembalikan oleh sistem.

Untuk mengevaluasi model adalah menampung terlebih dahulu data netflix yang akan menjadi data uji coba, dalam kasus ini saya mencoba untuk menampung data netflix yang mempunyai judul "A Cinderella Story" dan saya tampung pada variabel feature. Lalu selanjutnya saya menampung rating yang ada pada data uji coba untuk selanjutnya dipakai untuk evaluasi model.

Dan langkah terakhir yang saya lakukan adalah membuat perulangan berdasarkan rating pada data uji coba dan melakukan implementasi dari formula *precision*. Berikut adalah hasil keluaran dari implementasi formula precision : 

PG: 100.0

Output tersebut memberikan hasil yang cukup baik dan memiliki akurasi sebesar 100% sehingga dari sini saya bisa mengetahui bahwa model yang saya kembangkan berjalan sesuai yang diharapkan. 
 
### Kesimpulan

Proyek diatas sudah memenuhi goals yang diinginkan sesuai dengan Problem Statements yang dibahas. Sedikit banyaknya saya mengerti apa yang harus saya lakukan untuk mengerjakan proyek ini. Namun masih banyak pula hal yang perlu saya perbaiki dan pelajari agar hasil dari proyek sempurna.
