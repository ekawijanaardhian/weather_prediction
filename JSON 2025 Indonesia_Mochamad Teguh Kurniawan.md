# Estimasi Suhu Rata-Rata Harian Menggunakan Algoritma Random Forest dan Linear Regression Berdasarkan Observasi Meteorologi Multistasiun

**M. T. Kurniawan¹**, **Ardhian Ekawijana²\***

¹Fakultas Rekayasa Industri, Sistem Informasi, Universitas Telkom, Bandung, Indonesia  
²Jurusan Teknik Elektro, Politeknik Negeri Bandung, Bandung, Indonesia  

Email: ¹teguhkurniawan@telkomuniversity.ac.id, ²\*ardhian.ekawijana@polban.ac.id  
Email Penulis Korespondensi: ardhian.ekawijana@polban.ac.id  

Submitted: 99/99/9999; Accepted: 99/99/9999; Published: 99/99/9999

---

### **Abstrak**
Estimasi suhu udara rata-rata harian yang akurat dan andal sangat krusial bagi berbagai sektor di Indonesia, mulai dari pemantauan agroklimatologi, manajemen sumber daya air, hingga mitigasi risiko lingkungan. Penelitian ini bertujuan mengembangkan model estimasi suhu rata-rata harian (*same-day average temperature estimation*) menggunakan algoritma *Random Forest* (RF) dan *Linear Regression* (LR) berbasis data observasi meteorologi dari sembilan stasiun di berbagai wilayah Indonesia. Dataset mencakup 3.285 rekaman data harian sepanjang tahun dengan 19 fitur yang merepresentasikan parameter cuaca hari yang sama ($T_{\max}$, $T_{\min}$, kelembapan, kecepatan angin, curah hujan), indeks termal, serta fitur temporal siklikal dan encoding geografis. Pada evaluasi *random split* (80/20), *Random Forest* menunjukkan kemampuan interpolasi non-linear yang sangat baik dengan $R^2 = 0,9508$ ($\text{RMSE} = 0,4286^\circ\text{C}$), mengungguli *Linear Regression* ($R^2 = 0,9272$, $\text{RMSE} = 0,5213^\circ\text{C}$) dengan perbedaan yang signifikan secara statistik ($p < 0,001$). Analisis *feature importance* membuktikan bahwa *region encoding* (62,53%) dan parameter suhu ekstrem ($T_{\max}$ 18,78%, $T_{\min}$ 9,07%) menjadi kontributor utama, sementara studi ablasi menegaskan *heat index* tanpa *target leakage* tidak mengubah akurasi secara signifikan. Namun, pengujian validasi temporal menunjukkan dinamika berbeda: pada *split* kronologis ke depan (20 Okt–31 Des), *Linear Regression* lebih tangguh ($R^2 = 0,9123$, $\text{RMSE} = 0,5713^\circ\text{C}$) dibanding *Random Forest* ($R^2 = 0,8967$, $\text{RMSE} = 0,6200^\circ\text{C}$), yang dikonfirmasi oleh uji Diebold–Mariano ($p < 0,01$), sedangkan pada *rolling-origin* bulanan performa keduanya seimbang ($R^2 \approx 0,920$). Pada eksperimen prakiraan 1 hari ke depan ($t-1$), pemodelan selisih residual ($\Delta T = T_t - T_{t-1}$) memungkinkan Random Forest mencapai $R^2 = 0,8726$ ($\text{RMSE} = 0,6885^\circ\text{C}$), melampaui *baseline persistensi* ($R^2 = 0,8391$). Hasil ini menegaskan bahwa model ini dirancang optimal sebagai sistem estimasi dan rekonstruksi data observasi harian terdistribusi (*same-day estimation*), dengan fleksibilitas prakiraan *1-day ahead* melalui formulasi selisih suhu.

**Kata Kunci**: Estimasi Suhu Harian, Machine Learning, Random Forest, Linear Regression, Validasi Temporal, Meteorologi Indonesia

---

### **Abstract**
Accurate and reliable estimation of daily average temperature is crucial for various sectors in Indonesia, including agroclimatological monitoring, water resource management, and environmental risk mitigation. This study aims to develop a same-day daily average temperature estimation framework using Random Forest (RF) and Linear Regression (LR) algorithms based on meteorological observation records from nine multi-regional stations across Indonesia. The dataset comprises 3,285 daily records throughout a full annual cycle with 19 engineered features capturing same-day weather parameters ($T_{\max}$, $T_{\min}$, humidity, wind speed, precipitation), thermal indices, cyclical temporal markers, and geographic encodings. Under standard randomized 80/20 evaluation, Random Forest demonstrated strong non-linear interpolation capability with $R^2 = 0.9508$ ($\text{RMSE} = 0.4286^\circ\text{C}$), outperforming Linear Regression ($R^2 = 0.9272$, $\text{RMSE} = 0.5213^\circ\text{C}$) with statistically significant margins ($p < 0.001$). Feature importance analysis revealed that regional encoding (62.53%) and extreme temperatures ($T_{\max}$ 18.78%, $T_{\min}$ 9.07%) were the primary drivers, while ablation verified that leakage-free heat index had negligible impact on performance. However, temporal validation revealed contrasting dynamics: in forward chronological splitting (Oct 20–Dec 31), Linear Regression exhibited superior stability ($R^2 = 0.9123$, $\text{RMSE} = 0.5713^\circ\text{C}$) over Random Forest ($R^2 = 0.8967$, $\text{RMSE} = 0.6200^\circ\text{C}$), corroborated by the Diebold–Mariano test ($p < 0.01$), while monthly rolling-origin evaluations showed virtually identical mean performance ($R^2 \approx 0.920$). In a 1-day ahead forecasting setup ($t-1$), a delta residual formulation ($\Delta T = T_t - T_{t-1}$) allowed Random Forest to achieve $R^2 = 0.8726$ ($\text{RMSE} = 0.6885^\circ\text{C}$), outperforming the persistence baseline ($R^2 = 0.8391$). These findings confirm that the proposed framework is optimally suited for same-day climate observation estimation and historical gap-filling, while also enabling 1-day-ahead prediction through delta target modeling.

**Keywords**: Daily Temperature Estimation, Machine Learning, Random Forest, Linear Regression, Temporal Validation, Indonesian Climatology

---

## 1. PENDAHULUAN

Indonesia sebagai negara kepulauan tropis yang terletak di antara dua benua dan dua samudera memiliki variabilitas iklim lokal yang dinamis [1]. Parameter suhu udara rata-rata harian (*daily mean temperature*) merupakan variabel kunci dalam pemodelan agroklimatologi, evaluasi kenyamanan termal perkotaan, analisis hidrologi, hingga pemantauan perubahan iklim regional [2]. Meskipun Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) mengoperasikan jaringan stasiun pengamatan di berbagai provinsi, tantangan berupa keterbatasan instrumen pengukur suhu kontinu, kekosongan data (*missing values*), serta disparitas kerapatan sensor otomatis menuntut adanya pendekatan komputasi yang mampu mengestimasi dan merekonstruksi suhu rata-rata harian secara akurat dari parameter observasi cuaca dasar yang tersedia [1, 3].

Pendekatan fisik tradisional seperti *Numerical Weather Prediction* (NWP) memerlukan sumber daya komputasi skala superkomputer dan data asimilasi atmosfer yang sangat padat [4]. Di sisi lain, perkembangan teknologi *machine learning* menawarkan alternatif yang efisien untuk memodelkan hubungan non-linear antara variabel observasi permukaan dengan target suhu rata-rata harian [5, 6]. Berbagai algoritma pembelajaran mesin telah dieksplorasi dalam domain meteorologi di Asia Tenggara [7, 8].

Rahman et al. [9] mengembangkan model klasifikasi curah hujan menggunakan *Support Vector Machine* (SVM) pada 15 stasiun di Pulau Jawa dengan akurasi 87,3%, menunjukkan bahwa suhu dan kelembaban berkorelasi kuat dengan dinamika atmosfer lokal [10]. Sari dan Wibowo [11] menerapkan *Artificial Neural Network* (ANN) untuk estimasi suhu di Jakarta dengan mencapai RMSE 1,20°C [12], namun evaluasi tersebut terpusat pada satu lokasi perkotaan homogen. Kusuma et al. [14] memanfaatkan *Random Forest* dan klastering K-Means untuk mendeteksi pola monsun di Sumatera dengan akurasi 92,1% [7]. Sementara itu, Santoso et al. [16] menguji arsitektur LSTM untuk *time-series* cuaca di Bali dengan MAPE 8,3% [13].

Meskipun kajian terdahulu telah memperlihatkan potensi *machine learning*, terdapat sejumlah celah metodologis penting yang perlu diselesaikan [17]:
1. **Definisi Tugas Pemodelan:** Banyak penelitian mencampuradukkan antara peramalan masa depan (*future forecasting*) dengan estimasi observasi hari yang sama (*same-day interpolation/estimation*), sehingga evaluasi acak (*random split*) menghasilkan akurasi yang *over-optimistic* akibat autokorelasi serial antar-hari berdekatan.
2. **Pencegahan Target Leakage:** Penggunaan fitur turunan seperti *Heat Index* kerap kali secara tidak sengaja mengikutsertakan variabel target dalam formulanya, sehingga mendistorsi analisis *feature importance*.
3. **Validasi Temporal dan Spasial yang Ketat:** Belum banyak studi di Indonesia yang menguji konsistensi model lintas skema validasi temporal (seperti *chronological split*, *rolling-origin*, dan perbandingan *lag-1* terhadap *persistence baseline*) serta validasi spasial *Leave-One-Region-Out*.

Penelitian ini bertujuan merancang dan mengevaluasi sistem estimasi suhu rata-rata harian (*same-day daily average temperature estimation*) menggunakan algoritma *Random Forest* dan *Linear Regression* berdasarkan data observasi meteorologi multistasiun di Indonesia. Kontribusi utama penelitian ini mencakup:
1. Rekonstruksi data bersih dari 9 stasiun meteorologi representatif dengan imputasi spasial koridor Jawa untuk stasiun dengan data penakar hujan kosong.
2. *Feature engineering* bebas kebocoran target (*leakage-free heat index* berbasis $T_{\max}$ dan kelembapan) serta analisis *feature importance* yang terverifikasi melalui studi ablasi.
3. Evaluasi komprehensif yang membandingkan performa *random split*, *5-fold cross-validation*, validasi spasial regional, serta validasi temporal mendalam (*chronological split*, *rolling-origin*, dan *1-day ahead forecasting* versus *persistence baseline*).

---

## 2. METODOLOGI PENELITIAN

### 2.1 Tahapan Penelitian
Alur penelitian dilaksanakan secara sistematis melalui tahapan: pengumpulan data historis 9 stasiun meteorologi, pembersihan data dan konversi unit, imputasi spasial curah hujan, rekayasa fitur bebas kebocoran target, pelatihan model *Random Forest* dan *Linear Regression*, pengujian signifikansi statistik, validasi spasial *cross-regional*, serta pengujian validasi temporal berbasis urutan waktu (*chronological* dan *rolling-origin*).

### 2.2 Pengumpulan Data dan Preprocessing
Data dikumpulkan dari 9 stasiun meteorologi yang mewakili sebaran geografis Indonesia: Stasiun Kualanamu Medan dan Pinangsori Sibolga (Sumatera Utara), Banda Aceh (Aceh), Husein Sastranegara Bandung (Jawa Barat), Soekarno-Hatta Tangerang (Banten), Ahmad Yani Semarang (Jawa Tengah), Adisutjipto Yogyakarta (DI Yogyakarta), Lombok/Mataram (NTB), dan Sultan Hasanuddin Makassar (Sulawesi Selatan) [1, 3]. Dataset mencakup 3.285 rekaman data harian sepanjang 365 hari kalender.

Tahapan pra-pemrosesan meliputi:
1. **Konversi Satuan:** Kecepatan angin dikonversi secara presisi dari satuan km/jam menjadi meter per detik (m/s) ($\text{kecepatan} / 3{,}6$).
2. **Penanganan Missing Value & Data Kosong Yogyakarta:** Pada data mentah Meteostat, Stasiun Adisutjipto Yogyakarta (96853) tidak memiliki rekaman sensor curah hujan (berstatus *missing value* / *NaN*). Untuk menjaga integritas data tanpa merusak sebaran multivariat, nilai curah hujan Yogyakarta diimputasi menggunakan rata-rata spasial bulanan dari tiga stasiun lain di Pulau Jawa (Semarang, Bandung, dan Tangerang). Parameter lain ($T_{\max}$, $T_{\min}$, $RH$, kecepatan angin, dan suhu target) tidak memiliki nilai kosong; kolom *wind gust* dan *sunshine* yang kosong seluruhnya tidak digunakan dalam pemodelan.
3. **Validasi Fisik Ekstrem:** Nilai suhu minimum terendah $10{,}6^\circ\text{C}$ di Stasiun Bandung pada 14 September divalidasi sebagai fenomena iklim dataran tinggi yang riil (*bediding*) selama periode monsun dingin Australia di musim kemarau.

Karakteristik statistik dataset observasi setelah imputasi disajikan pada **Tabel 1**.

**Tabel 1. Karakteristik Dataset Observasi Meteorologi**

| Parameter | Satuan | Nilai Minimum | Nilai Maksimum | Mean | Standar Deviasi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Curah Hujan (*Rainfall*) | mm | 0,0 | 115,1 | 5,9 | 7,8 |
| Suhu Maksimum ($T_{\max}$) | °C | 22,2 | 38,0 | 31,7 | 2,1 |
| Suhu Minimum ($T_{\min}$) | °C | 10,6 | 28,0 | 23,3 | 2,3 |
| Suhu Rata-Rata ($T_{\text{avg}}$ - Target) | °C | 20,6 | 31,5 | 27,1 | 1,9 |
| Kelembapan Relatif ($RH$) | % | 46,0 | 97,0 | 79,4 | 8,4 |
| Kecepatan Angin | m/s | 0,9 | 5,6 | 2,3 | 0,6 |

---

### 2.3 Feature Engineering Bebas Target Leakage
Sebanyak 19 fitur prediktor dikembangkan tanpa menyertakan variabel target ($T_{\text{avg}}$):
1. **Fitur Termal & Atmosfer:** $T_{\max}$, $T_{\min}$, rentang suhu diurnal ($\Delta T = T_{\max} - T_{\min}$), kelembapan relatif ($RH$), kecepatan angin (m/s), dan presipitasi (mm).
2. **Heat Index Bebas Kebocoran:** Dihitung dari variabel prediktor $T_{\max}$ dan $RH$. Untuk kondisi sejuk di mana rata-rata indeks awal $< 80^\circ\text{F}$ (tercatat pada 35 rekaman: 31 di Bandung dan 4 di Makassar), digunakan formula Steadman sederhana:
   $$\text{HI}_{\text{simple}} = 0{,}5 \times \left(T + 61{,}0 + (T - 68{,}0) \times 1{,}2 + RH \times 0{,}094\right)$$
   Sementara pada kondisi panas ($\frac{\text{HI}_{\text{simple}} + T}{2} \ge 80^\circ\text{F}$), digunakan persamaan regresi Rothfusz penuh [21]:
   $$\begin{aligned}
   HI = &-42{,}379 + 2{,}04901523\,T + 10{,}14333127\,RH - 0{,}22475541\,T\,RH \\
   &- 6{,}83783\times 10^{-3}\,T^2 - 5{,}481717\times 10^{-2}\,RH^2 + 1{,}22874\times 10^{-3}\,T^2\,RH \\
   &+ 8{,}5282\times 10^{-4}\,T\,RH^2 - 1{,}99\times 10^{-6}\,T^2\,RH^2
   \end{aligned}$$
   di mana $T$ adalah $T_{\max}$ dalam derajat Fahrenheit (°F) dan $RH$ dalam persen.
3. **Fitur Temporal & Siklikal:** `month`, `quarter`, `day_of_year`, `day`, `year`, `season_encoded` (Musim Hujan: Nov–Mar, Kemarau: Jun–Sep, Transisi: Apr–Mei, Okt), serta transformasi trigonometrik siklikal: $\sin(2\pi \cdot \text{month}/12)$, $\cos(2\pi \cdot \text{month}/12)$, $\sin(2\pi \cdot \text{day}/365{,}25)$, dan $\cos(2\pi \cdot \text{day}/365{,}25)$.
4. **Fitur Geografis:** `station_encoded` dan `region_encoded` untuk menangkap *spatial baseline* dan mikroklimat wilayah masing-masing stasiun.

---

### 2.4 Konfigurasi Model Pembelajaran Mesin
1. **Random Forest Regressor:** Model *ensemble* non-parametrik dikonfigurasi dengan $n_{\text{estimators}} = 100$, kriteria *squared error*, dan `random_state = 42`.
2. **Linear Regression:** Model parametrik linier standar yang dilatih pada data terstandardisasi (*StandardScaler*, mean = 0, std = 1) sebagai pembanding yang efisien dan memiliki interpretibilitas tinggi.

---

### 2.5 Skema Evaluasi dan Validasi Komprehensif
Evaluasi model dirancang melalui empat skema pengujian:
1. **Random Split (80/20) & 5-Fold Cross Validation:** Evaluasi interpolasi acak standar menggunakan metrik $R^2$, RMSE, dan MAE, dilengkapi 5-Fold CV non-stratifikasi untuk mengukur stabilitas rata-rata dan deviasi standar model.
2. **Uji Signifikansi Statistik & Analisis Residual:** Penerapan *Paired t-test* dan *Wilcoxon Signed-Rank test* pada galat absolut data uji interpolasi, serta *Shapiro-Wilk test* untuk menguji normalitas residual. Untuk data berbasis waktu pada *split* kronologis, diterapkan uji **Diebold–Mariano** dengan lag autokorelasi Newey–West ($h = 7$) [22] guna menguji signifikansi perbedaan galat tanpa asumsi independensi observasi.
3. **Validasi Spasial Cross-Regional (Leave-One-Region-Out):** Pengujian ketahanan model ketika dilatih pada 7 region dan diuji pada 1 region yang sama sekali belum pernah dilihat.
4. **Validasi Temporal (Chronological Split, TimeSeriesSplit, Rolling-Origin, & Enhanced 1-Day Ahead Forecasting):** 
   - *Chronological Split:* Latih pada periode 1 Jan – 19 Okt, uji pada 20 Okt – 31 Des (80/20 berdasarkan urutan waktu).
   - *TimeSeriesSplit (5-Fold):* Evaluasi *expanding window* sekuensial lintas waktu untuk menguji ketahanan model pada data deret waktu murni.
   - *Rolling-Origin Bulanan:* Evaluasi jendela ekspansi bulanan untuk menguji performa bertahap pada bulan Juli hingga Desember.
   - *Eksperimen Prakiraan 1 Hari ke Depan ($t-1$):* Memprediksi suhu rata-rata hari $t$ menggunakan fitur observasi hari sebelumnya ($t-1$). Seluruh fitur lag dan statistik bergerak dihitung secara independen per stasiun melalui fungsi pengelompokan `df.groupby('station_id')[col].shift(1)` guna menjamin tidak adanya kebocoran data antar-stasiun. Kumpulan fitur diperkaya dengan lag 1–3 hari ($T_{\text{avg}}$, $T_{\max}$, $T_{\min}$, $RH$, angin, hujan), rata-rata bergerak (*rolling mean* 3 dan 7 hari), serta tren suhu ($T_{t-1} - T_{t-2}$). Model dievaluasi dalam dua skema target: (a) *direct target prediction* ($T_t$), dan (b) *delta residual modeling* ($\Delta T = T_t - T_{t-1}$), lalu dibandingkan langsung dengan *persistence baseline* ($T_t = T_{t-1}$), *Ridge Regression*, dan *HistGradientBoosting*.

---

## 3. HASIL DAN PEMBAHASAN

### 3.1 Evaluasi Model pada Random Split dan Studi Ablasi
Pada pengujian *random split* (80% latih, 20% uji), *Random Forest* mencapai performa interpolasi non-linear yang sangat tinggi dengan $R^2 = 0,9508$ dan $\text{RMSE} = 0,4286^\circ\text{C}$. *Linear Regression* memberikan performa yang juga solid dengan $R^2 = 0,9272$ dan $\text{RMSE} = 0,5213^\circ\text{C}$ (**Tabel 2**).

Uji hipotesis statistik mengonfirmasi bahwa perbedaan performa antara RF dan LR pada *random split* ini signifikan secara statistik (*Paired t-test*: $t = -6{,}2229$, $p = 8{,}70 \times 10^{-10} < 0{,}001$; *Wilcoxon test*: $W = 82269{,}0$, $p = 1{,}14 \times 10^{-7} < 0{,}001$). 

Studi ablasi (*ablation study*) yang mengecualikan fitur *Heat Index* membuktikan bahwa performa RF ($R^2 = 0,9506$, $\text{RMSE} = 0,4294^\circ\text{C}$) dan LR ($R^2 = 0,9273$, $\text{RMSE} = 0,5209^\circ\text{C}$) praktis tidak mengalami perubahan. Hal ini mengonfirmasi bahwa setelah *target leakage* dieliminasi, *heat index* bertindak sebagai fitur komplementer dan tidak mendominasi proses pembelajaran model secara artifisial.

**Tabel 2. Hasil Evaluasi Model dan Studi Ablasi (Random Split 80/20)**

| Konfigurasi Model | $R^2$ Score (Test) | RMSE (°C) | 5-Fold CV $R^2$ (Mean ± Std) | Waktu Latih (s) | Waktu Prediksi (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (19 Fitur Lengkap)** | **0,9508** | **0,4286** | **0,9470 ± 0,0056** | **1,38** | **0,022** |
| **Linear Regression (19 Fitur Lengkap)** | **0,9272** | **0,5213** | **0,9249 ± 0,0057** | **0,00** | **< 0,001** |
| *Ablasi: Random Forest (Tanpa Heat Index)* | 0,9506 | 0,4294 | — | — | — |
| *Ablasi: Linear Regression (Tanpa Heat Index)* | 0,9273 | 0,5209 | — | — | — |

---

### 3.2 Analisis Feature Importance
Analisis tingkat kepentingan fitur (*Mean Decrease in Impurity / MDI berbasis reduksi variansi*) pada model *Random Forest* menunjukkan pergeseran struktural yang logis secara fisik (**Tabel 3**).

**Tabel 3. Analisis Feature Importance (Top 10 Random Forest)**

| Peringkat | Fitur Prediktor | Nilai Importance | Kumulatif (%) |
| :---: | :--- | :---: | :---: |
| 1 | **Region Encoded** | 0,6253 | 62,53% |
| 2 | **Suhu Maksimum ($T_{\max}$)** | 0,1878 | 81,31% |
| 3 | **Suhu Minimum ($T_{\min}$)** | 0,0907 | 90,39% |
| 4 | **Kelembapan Relatif ($RH$)** | 0,0275 | 93,14% |
| 5 | **Heat Index** | 0,0167 | 94,81% |
| 6 | **Day of Year** | 0,0097 | 95,78% |
| 7 | **Kecepatan Angin (m/s)** | 0,0074 | 96,53% |
| 8 | **Curah Hujan (mm)** | 0,0065 | 97,18% |
| 9 | **Day Cos (Siklikal)** | 0,0063 | 97,81% |
| 10 | **Rentang Suhu Diurnal ($\Delta T$)** | 0,0050 | 98,31% |

Fitur `region_encoded` berkontribusi sebesar 62,53%, merefleksikan bahwa *spatial baseline* dan karakteristik geografis lokal antar-wilayah Indonesia merupakan penentu utama rentang suhu harian. Parameter observasi suhu ekstrem ($T_{\max}$ 18,78% dan $T_{\min}$ 9,07%) menjadi prediktor langsung batas rentang suhu rata-rata, sedangkan *heat index* hanya menyumbang 1,67%.

---

### 3.3 Analisis Pola Regional
Statistik observasi regional menyajikan variasi iklim tropis yang heterogen di seluruh stasiun pengamatan (**Tabel 4**).

**Tabel 4. Statistik Cuaca Regional Berdasarkan Observasi Stasiun**

| Wilayah (*Region*) | Curah Hujan (mm/hari) | Suhu Rata-Rata (°C) | Kelembapan (%) | Jumlah Stasiun |
| :--- | :---: | :---: | :---: | :---: |
| **Sumatera Utara** | 9,7 | 27,1 | 87,1 | 2 |
| **Sulawesi Selatan** | 7,7 | 27,2 | 75,8 | 1 |
| **Aceh** | 5,4 | 27,5 | 77,7 | 1 |
| **Jawa Barat (Bandung)** | 4,4 | 22,8 | 78,3 | 1 |
| **Banten (Tangerang)** | 4,2 | 28,3 | 78,5 | 1 |
| **DI Yogyakarta** *(Imputasi Spasial)* | 4,1 | 28,1 | 75,3 | 1 |
| **NTB (Mataram)** | 3,9 | 26,9 | 81,1 | 1 |
| **Jawa Tengah (Semarang)** | 3,6 | 28,6 | 73,6 | 1 |

Wilayah Sumatera Utara dan Sulawesi Selatan mencatat intensitas curah hujan harian tertinggi, sedangkan Jawa Barat (Stasiun Bandung di dataran tinggi) mencatat rata-rata suhu harian paling sejuk ($22{,}8^\circ\text{C}$).

---

### 3.4 Analisis Pola Musiman
Pola presipitasi bulanan hasil observasi dan imputasi spasial konsisten merefleksikan pergerakan monsun Indonesia (**Tabel 5**). Puncak musim hujan terjadi pada periode November–Maret dengan curah hujan bulanan berkisar antara 237,8 mm hingga 275,1 mm (rata-rata harian 7,7–9,8 mm), sementara musim kemarau berlangsung dari Juni hingga September dengan curah hujan bulanan di bawah 99 mm (90,8–98,7 mm).

**Tabel 5. Pola Presipitasi Musiman**

| Bulan | Curah Hujan Bulanan (mm) | Rata-Rata Harian (mm) | Klasifikasi Musim |
| :---: | :---: | :---: | :--- |
| Jan | 237,8 | 7,7 | Musim Hujan (*Rainy Season*) |
| Feb | 275,1 | 9,8 | Musim Hujan (*Rainy Season*) |
| Mar | 251,2 | 8,1 | Musim Hujan (*Rainy Season*) |
| Apr | 192,1 | 6,4 | Peralihan (*Transition*) |
| Mei | 142,1 | 4,6 | Peralihan (*Transition*) |
| Jun | 98,7 | 3,3 | Musim Kemarau (*Dry Season*) |
| Jul | 93,7 | 3,0 | Musim Kemarau (*Dry Season*) |
| Agu | 90,8 | 2,9 | Musim Kemarau (*Dry Season*) |
| Sep | 96,0 | 3,2 | Musim Kemarau (*Dry Season*) |
| Okt | 160,5 | 5,2 | Peralihan (*Transition*) |
| Nov | 243,0 | 8,1 | Musim Hujan (*Rainy Season*) |
| Des | 255,1 | 8,2 | Musim Hujan (*Rainy Season*) |

---

### 3.5 Validasi Prediksi pada Skenario Riil
Uji coba prediksi pada tiga skenario kondisi riil disajikan pada **Tabel 6**. 

**Tabel 6. Validasi Prediksi pada Skenario Riil**

| Skenario Pengujian | Kondisi Input Parameter | Prediksi RF | Prediksi LR | Rata-Rata Prediksi |
| :--- | :--- | :---: | :---: | :---: |
| **Jakarta Musim Hujan** | Bulan: 1, Kelembapan: 80%, $T_{\max}$: 32°C | 27,27°C | 27,48°C | 27,37°C |
| **Medan Musim Kemarau** | Bulan: 7, Kelembapan: 65%, $T_{\max}$: 35°C | 28,71°C | 29,60°C | 29,15°C |
| **Bandung Kondisi Normal**| Bulan: 4, Kelembapan: 70%, $T_{\max}$: 28°C | 24,36°C | 24,13°C | 24,24°C |

Hasil prediksi selaras dengan dinamika meteorologi tropis: skenario Jakarta musim hujan mengestimasi suhu rata-rata $27{,}37^\circ\text{C}$ (RF: 27,27°C, LR: 27,48°C) dan Medan musim kemarau $29{,}15^\circ\text{C}$ (RF: 28,71°C, LR: 29,60°C). Pada skenario Bandung dengan input $T_{\max} = 28^\circ\text{C}$, model mengestimasi suhu moderat $24{,}24^\circ\text{C}$ (RF: 24,36°C, LR: 24,13°C), yang berada di bawah $T_{\max}$ sehingga masuk akal secara fisik. Hal ini membuktikan bahwa model berhasil merefleksikan **baseline suhu spesifik per stasiun yang dipelajari secara efektif melalui station dan region encoding**, tanpa memerlukan parameter elevasi fisik eksplisit.

---

### 3.6 Perbandingan dengan Penelitian Terkait
Meskipun perbandingan langsung dibatasi oleh perbedaan karakteristik dan cakupan dataset antar-studi, hasil evaluasi pada dataset sembilan stasiun nasional ini memberikan gambaran performa komparatif terhadap pemodelan meteorologi berbasis *machine learning* di Indonesia, sebagaimana dirangkum pada **Tabel 7**.

**Tabel 7. Perbandingan Performa dengan Penelitian Terkait**

| Studi Penelitian | Metode / Algoritma | Metrik Evaluasi | Hasil Performa | Cakupan Wilayah |
| :--- | :--- | :---: | :---: | :--- |
| Sari & Wibowo [11] | Artificial Neural Network | RMSE | 1,20°C | Suhu Jakarta (Lokal) |
| Rahman et al. [9] | Support Vector Machine | Akurasi | 87,30% | Curah Hujan Jawa |
| Kusuma et al. [14] | K-Means + Random Forest | Akurasi | 92,10% | Monsun Sumatera |
| **Penelitian Ini** | **Linear Regression** | **$R^2$ Score / RMSE** | **92,72% / 0,5213°C** | **Multistasiun Nasional** |
| **Penelitian Ini** | **Random Forest** | **$R^2$ Score / RMSE** | **95,08% / 0,4286°C** | **Multistasiun Nasional** |

---

### 3.7 Kinerja Komputasi dan Analisis Sumber Daya
Pengukuran profil komputasi menunjukkan efisiensi praktis (*practical efficiency*) kedua algoritma untuk kebutuhan pemrosesan data meteorologi harian (**Tabel 8**). *Linear Regression* memiliki waktu pelatihan instan (<0,001 detik), waktu inferensi sangat cepat (< 0,001 detik), serta ukuran file model pada disk hanya 0,8 KB. *Random Forest* membutuhkan waktu latih 1,38 detik dan ukuran file model 18,93 MB, yang sangat layak untuk diimplementasikan pada perangkat keras berspesifikasi standar.

**Tabel 8. Profil Kinerja Komputasi dan Ukuran Model**

| Aspek Komputasi | Random Forest | Linear Regression |
| :--- | :---: | :---: |
| Waktu Pelatihan (*Training Time*) | 1,38 detik | 0,00 detik |
| Waktu Inferensi (*Batch Prediction*) | 0,022 detik | < 0,001 detik |
| Ukuran Berkas Model di Disk | 18,93 MB | 0,8 KB |

---

### 3.8 Analisis Distribusi Galat dan Residual
Analisis distribusi galat menunjukkan rata-rata galat mendekati nol pada kedua algoritma (**Tabel 9**). Sebanyak 78,7% prediksi RF memiliki galat di bawah $0{,}5^\circ\text{C}$, dan 97,7% berada di bawah $1{,}0^\circ\text{C}$. Rata-rata galat yang mendekati nol ($\approx -0{,}01^\circ\text{C}$) menandakan tidak adanya bias sistematis arah pada model. Namun, uji normalitas *Shapiro-Wilk* pada residual ($W = 0{,}9671$, $p = 3{,}81 \times 10^{-9} < 0{,}001$) mengindikasikan bahwa distribusi residual menyimpang dari kurva normal murni secara statistik, yang utamanya disebabkan oleh beberapa galat yang lebih besar pada ekor distribusi (*heavy tails*) saat kondisi cuaca ekstrem lokal.

**Tabel 9. Analisis Distribusi Galat Prediksi**

| Metrik Galat | Random Forest | Linear Regression |
| :--- | :---: | :---: |
| Rata-Rata Galat (*Mean Error*) | -0,0100°C | -0,0224°C |
| Standar Deviasi Galat (*Std Error*) | 0,4288°C | 0,5212°C |
| Galat Maksimum (*Max Error*) | +1,9010°C | +2,3863°C |
| Galat Minimum (*Min Error*) | -1,4210°C | -1,5879°C |
| Proporsi Galat < 0,5°C (%) | 78,7% | 72,5% |
| Proporsi Galat < 1,0°C (%) | 97,7% | 94,1% |

---

### 3.9 Validasi Cross-Regional (Leave-One-Region-Out)
Untuk menguji generalisasi model terhadap stasiun di wilayah geografis baru yang tidak disertakan dalam data latih, dilakukan validasi *Leave-One-Region-Out* pada 8 wilayah (**Tabel 10**). Penurunan performa dihitung terhadap baseline model $R^2_{\text{baseline}} = 0,9508$ melalui formula:
$$\text{Performance Drop (\%)} = \frac{R^2_{\text{region}} - R^2_{\text{baseline}}}{R^2_{\text{baseline}}} \times 100\%$$

**Tabel 10. Hasil Validasi Cross-Regional (Leave-One-Region-Out)**

| Wilayah Uji (*Test Region*) | $R^2$ Score | RMSE (°C) | Penurunan Performa (%) |
| :--- | :---: | :---: | :---: |
| DI Yogyakarta | 0,8932 | 0,3177 | -6,06% |
| Jawa Tengah (Semarang) | 0,8193 | 0,4792 | -13,83% |
| Banten (Tangerang) | 0,7394 | 0,5450 | -22,23% |
| NTB (Mataram) | 0,6372 | 0,6463 | -32,98% |
| Sumatera Utara (Medan/Sibolga) | 0,6099 | 0,6169 | -35,85% |
| Aceh (Banda Aceh) | 0,5656 | 0,8455 | -40,51% |
| Sulawesi Selatan (Makassar) | 0,3900 | 0,9330 | -58,98% |
| Jawa Barat (*Dataran Tinggi Bandung*)\* | -5,6308 | 2,2330 | -692,22% |
| **Rata-Rata Wilayah Dataran Rendah (7 Region)** | **0,6649** | **0,6262** | **-30,06%** |

Model mempertahankan akurasi yang baik pada wilayah dataran rendah dengan rata-rata $R^2 = 0,6649$, $\text{RMSE} = 0,6262^\circ\text{C}$, dan penurunan performa rata-rata $-30,06\%$. Penurunan tajam pada wilayah Jawa Barat terjadi karena Bandung merupakan satu-satunya stasiun dataran tinggi ($>700$ mdpl) dalam dataset, sehingga saat dikeluarkan dari data latih, model tidak memiliki referensi termal dataran tinggi.

---

### 3.10 Validasi Temporal dan Evaluasi Kelayakan Peramalan (*Temporal Validation*)
Untuk mengevaluasi ketahanan model terhadap autokorelasi serial serta memverifikasi kesesuaian tugas pemodelan (*same-day estimation* vs *future forecasting*), dilakukan empat pengujian temporal berbasis urutan waktu:

#### 1. Split Kronologis (Chronological Split 80/20) dan Uji Signifikansi Deret Waktu
Model dilatih pada data 1 Januari hingga 19 Oktober (80% awal), kemudian diuji untuk mengestimasi suhu pada periode 20 Oktober hingga 31 Desember (20% akhir tahun) (**Tabel 11**).

**Tabel 11. Hasil Evaluasi Split Kronologis (Uji: 20 Okt – 31 Des)**

| Model Algoritma | $R^2$ Score | RMSE (°C) | MAE (°C) | Karakteristik Performa |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression (LR)** | **0,9123** | **0,5713** | **0,4443** | **Lebih tangguh dalam ekstrapolasi tren data baru** |
| **Random Forest (RF)** | **0,8967** | **0,6200** | **0,4805** | Terdegradasi akibat sifat *piecewise-constant* pohon regresi |

Pada skema kronologis ini, **Linear Regression ($R^2 = 0,9123$) mengungguli Random Forest ($R^2 = 0,8967$)**. Untuk menguji signifikansi keunggulan ini tanpa melanggar asumsi independensi galat berurutan, diterapkan uji **Diebold–Mariano ($h = 7$ hari autokorelasi)** yang menghasilkan statistik $DM = +3{,}1860$ dengan nilai $p = 1{,}44 \times 10^{-3} < 0{,}01$. Hal ini membuktikan secara statistik bahwa *loss* galat kuadrat Linear Regression lebih rendah secara signifikan dibanding Random Forest pada horizon data waktu ke depan.

#### 2. TimeSeriesSplit Cross-Validation (5 Folds)
Evaluasi *TimeSeriesSplit* (5-Fold *expanding window*) pada data sekuensial menghasilkan $R^2 = 0{,}8771 \pm 0{,}0487$ ($\text{RMSE} = 0{,}6529 \pm 0{,}1105^\circ\text{C}$) untuk Random Forest. Sebagai pembanding, Linear Regression standar menghasilkan rata-rata $R^2 = 0{,}4911 \pm 0{,}8206$ ($\text{RMSE} = 1{,}0102 \pm 0{,}7984^\circ\text{C}$), yang utamanya dipengaruhi oleh sensitivitas terhadap multikolinearitas musiman pada Fold 1 saat jendela data latih masih sangat sempit di awal tahun ($R^2 = -1{,}1500$), sebelum meningkat stabil pada Fold 2–5 ($R^2 = 0{,}8922$ hingga $0{,}9153$). Sementara itu, *Ridge Regression* ($\alpha = 10$) dengan regularisasi $L_2$ mencapai performa yang sangat konsisten di seluruh fold dengan $R^2 = 0{,}9126 \pm 0{,}0110$ ($\text{RMSE} = 0{,}5638 \pm 0{,}0309^\circ\text{C}$).

#### 3. Evaluasi Rolling-Origin Bulanan (Juli – Desember)
Pengujian *rolling-origin* menggunakan *expanding window* bulanan dilakukan dari bulan Juli hingga Desember (**Tabel 12**).

**Tabel 12. Hasil Evaluasi Rolling-Origin Bulanan**

| Bulan Pengujian | RF $R^2$ Score | RF RMSE (°C) | LR $R^2$ Score | LR RMSE (°C) |
| :---: | :---: | :---: | :---: | :---: |
| **Juli** | 0,9288 | 0,5626 | 0,9277 | 0,5670 |
| **Agustus** | 0,9412 | 0,4650 | 0,9340 | 0,4928 |
| **September** | 0,9116 | 0,5760 | 0,9068 | 0,5912 |
| **Oktober** | 0,8817 | 0,6466 | 0,9184 | 0,5371 |
| **November** | 0,9274 | 0,5310 | 0,9033 | 0,6130 |
| **Desember** | 0,9334 | 0,4829 | 0,9280 | 0,5022 |
| **Rata-Rata Evaluasi** | **0,9207** | **0,5440** | **0,9197** | **0,5505** |

Hasil *rolling-origin* menunjukkan bahwa rata-rata performa RF ($R^2 = 0,9207$) dan LR ($R^2 = 0,9197$) **praktis seimbang**. Temuan ini membuktikan bahwa klaim keunggulan mutlak RF pada *random split* ($R^2 \approx 0,95$) sebagian dipengaruhi oleh kemiripan kondisi cuaca pada hari-hari yang berdekatan (*temporal autocorrelation*).

#### 4. Eksperimen Prakiraan 1 Hari ke Depan (Fitur Lag $t-1 \rightarrow$ Target Suhu $t$)
Untuk menguji batas kemampuan peramalan masa depan (*forecasting*), dievaluasi beberapa arsitektur model menggunakan kumpulan fitur multi-lag (lag 1–3 hari, *rolling mean* 3 dan 7 hari, serta tren suhu $T_{t-1} - T_{t-2}$) pada skema *direct prediction* dan *delta modeling* ($\Delta T = T_t - T_{t-1}$), dibandingkan terhadap *Persistence Baseline* ($T_t = T_{t-1}$) (**Tabel 13**).

**Tabel 13. Perbandingan Eksperimen Prakiraan 1 Hari ke Depan**

| Skema / Arsitektur Model Prediksi | $R^2$ Score | RMSE (°C) | MAE (°C) | Evaluasi Metodologis |
| :--- | :---: | :---: | :---: | :--- |
| **Persistence Baseline ($T_t = T_{t-1}$)** | 0,8391 | 0,7737 | 0,5633 | *Benchmark* acuan suhu hari sebelumnya |
| **HistGradientBoosting (Direct Target)** | 0,8560 | 0,7320 | 0,5671 | Melampaui baseline persistensi |
| **HistGradientBoosting ($\Delta T$ Delta Model)** | 0,8622 | 0,7160 | 0,5541 | Performa meningkat dengan target selisih |
| **Linear Regression (Rich Lags Direct)** | 0,8674 | 0,7023 | 0,5209 | Model linier multivariat lag tangguh |
| **Ridge Regression ($\alpha=10$, Direct)** | 0,8694 | 0,6971 | 0,5171 | Regularisasi $L_2$ meningkatkan stabilitas |
| **Random Forest ($\Delta T = T_t - T_{t-1}$ Delta Model)** | **0,8726** | **0,6885** | **0,5229** | **Tertinggi; melampaui baseline, setara dengan Ridge** |

Temuan eksperimen ini menunjukkan bahwa:
1. Ketika memprediksi suhu mutlak $T(t)$ secara langsung, Random Forest standar tanpa rekayasa selisih rentan terdegradasi ($R^2 = 0,8019$). Namun, ketika diformulasikan sebagai **model estimasi selisih suhu harian ($\Delta T = T_t - T_{t-1}$)**, Random Forest berhasil mencapai $R^2 = 0,8726$ ($\text{RMSE } 0,6885^\circ\text{C}$), melampaui *persistence baseline* ($R^2 = 0,8391$), model linier reguler ($R^2 = 0,8674$), serta praktis setara dengan Ridge Regression ($R^2 = 0,8694$).
2. Pada domain estimasi observasi hari yang sama (*same-day estimation*), *Random Forest* mencapai akurasi optimal $R^2 = 0,9508$ ($\text{RMSE } 0,4286^\circ\text{C}$), sedangkan untuk kebutuhan prakiraan 1 hari ke depan (*1-day ahead forecasting*), pemodelan berbasis *delta target* menjadi solusi paling efektif.

---

## 4. KESIMPULAN

Penelitian ini telah berhasil mengembangkan dan mengevaluasi kerangka kerja komputasi untuk estimasi suhu rata-rata harian (*same-day daily average temperature estimation*) berbasis algoritma *Random Forest* dan *Linear Regression* menggunakan data observasi meteorologi multistasiun di Indonesia.

Melalui pra-pemrosesan yang ketat, eliminasi kebocoran target pada *heat index*, dan analisis empiris komprehensif, diperoleh kesimpulan utama sebagai berikut:
1. **Performa Interpolasi vs Ekstrapolasi Temporal:** Pada pengujian *random split*, *Random Forest* unggul dalam menangkap relasi non-linear ($R^2 = 0,9508$, $\text{RMSE} = 0,4286^\circ\text{C}$) dibandingkan *Linear Regression* ($R^2 = 0,9272$, $\text{RMSE} = 0,5213^\circ\text{C}$). Namun, pada validasi kronologis murni ke depan, *Linear Regression* terbukti lebih tangguh ($R^2 = 0,9123$ vs $0,8967$) dan seimbang pada evaluasi *rolling-origin* ($R^2 \approx 0,920$).
2. **Struktur Kontribusi Fitur:** *Region encoding* (62,53%) serta suhu ekstrem harian ($T_{\max}$ 18,78% dan $T_{\min}$ 9,07%) merupakan faktor penentu utama estimasi suhu harian, sedangkan studi ablasi membuktikan *heat index* tanpa *target leakage* tidak mengubah akurasi secara signifikan.
3. **Generalisasi Spasial dan Temporal:** Validasi *Leave-One-Region-Out* membuktikan model mampu mengestimasi suhu di stasiun dataran rendah baru dengan rata-rata $R^2 = 0,6649$ (penurunan performa rata-rata $-30,06\%$). Pengujian prakiraan 1 hari ke depan menegaskan bahwa model ini paling tepat diposisikan sebagai sistem estimasi dan pengisian data kosong (*gap-filling/quality control*) observasi hari yang sama, bukan peramalan masa depan operasional.

Pengembangan di masa mendatang disarankan untuk mengintegrasikan model deret waktu (*time-series deep learning* seperti PatchTST atau TiDE), data reanalisis satelit (ERA5), dan variabel elevasi topografi eksplisit untuk meningkatkan kemampuan prakiraan masa depan *multi-step ahead*.

---

## REFERENCES

[1] BMKG, "Climate Data and Information of Indonesia," *Badan Meteorologi Klimatologi dan Geofisika*, Jakarta, Indonesia, 2023.  
[2] J. L. Monteith and M. H. Unsworth, *Principles of Environmental Physics: Plants, Animals, and the Atmosphere*, 4th ed. Academic Press, 2013.  
[3] World Meteorological Organization, "Guidelines on ensemble prediction systems and forecasting," *WMO Technical Document*, WMO-No. 1091, 2012.  
[4] P. Bauer, A. Thorpe, and G. Brunet, "The quiet revolution of numerical weather prediction," *Nature*, vol. 525, no. 7567, pp. 47-55, 2015.  
[5] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed. New York: Springer, 2009.  
[6] S. Raschka and V. Mirjalili, *Python Machine Learning: Machine Learning and Deep Learning with Python*, 3rd ed. Packt Publishing, 2019.  
[7] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.  
[8] A. Géron, *Hands-On Machine Learning with Scikit-Learn and TensorFlow*, 2nd ed. O'Reilly Media, 2019.  
[9] A. Rahman, B. Sartono, and I. Fahmi, "Rainfall prediction using support vector machine in weather forecasting for Indonesian regions," *International Journal of Advanced Computer Science and Applications*, vol. 8, no. 8, pp. 143-152, 2017.  
[10] N. Cristianini and J. Shawe-Taylor, *An Introduction to Support Vector Machines and Other Kernel-based Learning Methods*. Cambridge University Press, 2000.  
[11] M. Sari and A. Wibowo, "Temperature forecasting using artificial neural network for Jakarta weather prediction," *Journal of Physics: Conference Series*, vol. 1367, no. 1, p. 012087, 2019.  
[12] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436-444, 2015.  
[13] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.  
[14] I. K. Kusuma, N. P. Sastra, and G. M. Wibawa, "Rainfall pattern prediction using K-means clustering and random forest algorithm in Sumatra," *International Journal of Artificial Intelligence Research*, vol. 4, no. 2, pp. 102-113, 2020.  
[15] D. Wijaya and R. Pratama, "Extreme weather prediction using ensemble machine learning with satellite data integration," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 59, no. 4, pp. 3247-3258, 2021.  
[16] B. Santoso, F. Herawan, and S. Defit, "Long short-term memory for weather forecasting: A case study in Bali Indonesia," *Journal of King Saud University - Computer and Information Sciences*, vol. 34, no. 6, pp. 3482-3493, 2022.  
[17] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021.  
[18] C. M. Bishop, *Pattern Recognition and Machine Learning*. New York: Springer, 2006.  
[19] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.  
[20] G. James, D. Witten, T. Hastie, and R. Tibshirani, *An Introduction to Statistical Learning with Applications in R*, 2nd ed. New York: Springer, 2021.  
[21] L. P. Rothfusz, "The heat index 'equation' (or, more than you ever wanted to know about heat index)," *NWS Southern Region Technical Attachment SR 90-23*, 1990.  
[22] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253-263, 1995.
