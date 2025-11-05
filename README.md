# 🤖 Chatbot Jaga Mata

**Jaga Mata** adalah chatbot berbasis kecerdasan buatan (AI) yang membantu pengguna menjaga kesehatan mata, mendeteksi potensi penyakit mata, dan memberikan edukasi seputar perawatan mata berkelanjutan.

Chatbot ini dilatih menggunakan dataset percakapan berbahasa Indonesia dan memanfaatkan pendekatan **klasifikasi intent (intent classification)** untuk memahami maksud pengguna.

---

## 🧠 Konsep Dasar

Chatbot ini bekerja dengan prinsip **Natural Language Processing (NLP)**, khususnya dalam mengklasifikasikan teks.

1. **Input pengguna** → teks pertanyaan atau keluhan (contoh: *"mata saya merah dan perih"*)
2. **Model NLP** → memprediksi intent (contoh: `treatment`)
3. **Response generator** → mengambil jawaban yang sesuai dari dataset

Alur sederhana:
```
User Input → Intent Classification → Response Selection → Chatbot Reply
```


---

## 📂 Struktur Dataset

Dataset utama disimpan dalam file [`jaga_mata_chatbot.csv`](./jaga_mata_chatbot.csv)

| intent | question | answer |
|--------|-----------|--------|
| greeting | Halo | Halo! 👋 Saya JagaBot, asisten kesehatan mata Anda. Ada yang bisa saya bantu hari ini? |
| check_symptom | Apa tanda-tanda mata minus | Tanda-tanda mata minus bisa meliputi: penglihatan jauh menjadi buram... |
| treatment | Bagaimana mengatasi mata kering | Gunakan tetes mata buatan, minum air cukup, dan kurangi paparan AC. |

**Kolom penjelasan:**
- `intent` → kategori atau maksud dari pertanyaan pengguna  
- `question` → contoh kalimat dari pengguna  
- `answer` → respons yang diberikan chatbot  

Dataset ini berisi **ratusan baris data** yang mencakup berbagai topik:
- 👋 *greeting* – sapaan awal
- 😷 *check_symptom* – pengecekan gejala mata
- 💊 *treatment* – saran pengobatan
- 🧑‍⚕️ *prevention* – tips pencegahan
- 💻 *vision_care* – edukasi tentang kebiasaan layar
- ⚠️ *emergency* – penanganan darurat

---

## ⚙️ Model Klasifikasi

Model menggunakan pendekatan **klasifikasi teks (Text Classification)** dengan pipeline sederhana:

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd

# Load dataset
df = pd.read_csv("jaga_mata_chatbot.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["question"], df["intent"], test_size=0.2, random_state=42)

# Buat model klasifikasi teks
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Tes prediksi intent
print(model.predict(["mata saya merah dan perih"]))

```

Output:
```
['treatment']
```

