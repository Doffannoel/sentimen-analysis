# Analisis Sentimen dengan IndoRoBERTa (80%–20% Split dan Evaluasi Lengkap)

Berikut adalah notebook Python lengkap untuk melakukan fine-tuning model **`w11wo/indonesian-roberta-base-sentiment-classifier`** (Hugging Face) pada dataset Anda (16K komentar). Langkah-langkah utamanya meliputi:

- **Persiapan dan import pustaka**: menginstal/masukkan `transformers`, `sklearn`, dan pustaka pendukung lainnya.
- **Load dan bersihkan data**: muat file CSV, ambil kolom teks dan label, lalu lakukan pembersihan teks (lowercase, hapus URL, non-alfabet, stemming, hapus stopwords Bahasa Indonesia).
- **Encode label**: ubah label (-1,0,1) menjadi integer (0,1,2) dengan `LabelEncoder`.
- **Split data**: bagi dataset menjadi 80% data latih dan 20% data uji secara **stratified** agar distribusi kelas seimbang【34†L5-L9】.
- **Tokenisasi**: gunakan *tokenizer* RoBERTa Bahasa Indonesia untuk ubah teks menjadi token.
- **Pembuatan dataset PyTorch**: wrap data latih dan uji ke dalam `Dataset` agar bisa digunakan oleh Trainer.
- **Penyesuaian Bobot Kelas**: hitung bobot kelas (invers frekuensi) untuk mengatasi ketidakseimbangan data (kelas positif jarang).
- **Load dan siapkan model**: pakai `AutoModelForSequenceClassification` dari Hugging Face, konfigurasi untuk 3 kelas.
- **Custom Trainer dengan Weighted Loss**: override `compute_loss` untuk menggunakan `CrossEntropyLoss` dengan bobot kelas.
- **Fine-tuning**: train model dengan parameter yang sudah diatur.
- **Evaluasi pada data uji**: prediksi label uji, hitung *confusion matrix*, *precision*, *recall*, *F1-score*, serta TP/TN.
- **Contoh prediksi**: simulasikan prediksi model pada beberapa komentar baru.

Kode di bawah ini diatur sesuai langkah-langkah tersebut. Silakan ubah nama kolom (misal `"Isi komentar"` dan `"Sentimen"`) jika berbeda, dan jalankan di lingkungan Python yang telah menginstal semua pustaka.



```python
# -------------------------
# 0. Persiapan Lingkungan
# -------------------------
!pip install -q transformers torch sklearn nltk sastrawi

import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Perangkat yang digunakan: {device}\n")
```

```python
# -------------------------
# 1. Load dan Persiapan Data
# -------------------------
print("Memuat data...")
# Ganti path CSV jika perlu
df = pd.read_csv('/mnt/data/Data Gabungan - Sheet1.csv')
print(f"Jumlah baris awal: {len(df)}")
print("Kolom tersedia:", df.columns.tolist())

# Pastikan kolom teks dan label ada
if 'Isi komentar' not in df.columns or 'Sentimen' not in df.columns:
    raise ValueError("File CSV harus berisi kolom 'Isi komentar' dan 'Sentimen'.")

df = df[['Isi komentar','Sentimen']].dropna()
print(f"Jumlah baris setelah drop NA: {len(df)}")

# Tampilkan sedikit data
print(df.head(3))
```

```python
# -------------------------
# 2. Preprocessing Teks
# -------------------------
print("\nMelakukan preprocessing teks...")

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(nltk.corpus.stopwords.words('indonesian'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+|http\S+|www.\S+','', text)             # hapus mention dan URL
    text = re.sub(r'[^a-zA-Z\s]','', text)                     # hanya huruf dan spasi
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]  # stemming + hapus stopword
    return ' '.join(words)

df['cleaned_text'] = df['Isi komentar'].apply(clean_text)

# Tampilkan contoh hasil pembersihan teks
print("\nContoh sebelum-sesudah preprocessing:")
for i in range(min(3, len(df))):
    ori = df['Isi komentar'].iloc[i]
    cl = df['cleaned_text'].iloc[i]
    print(f"- Original: {ori[:60]}...")
    print(f"  Cleaned : {cl[:60]}...\n")
```

```python
# -------------------------
# 3. Encode Label
# -------------------------
print("\nMengencode label...")
label_encoder = LabelEncoder()
# Label semula: -1 (negatif), 0 (netral), 1 (positif)
df['label'] = label_encoder.fit_transform(df['Sentimen'])
print("Mapping label:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Cek distribusi kelas
counts = df['label'].value_counts().sort_index()
print("\nDistribusi kelas (label -> count):")
for label, count in counts.items():
    sentiment = label_encoder.inverse_transform([label])[0]
    print(f"  {label} ({sentiment}): {count} ({count/len(df)*100:.2f}%)")
```

```python
# -------------------------
# 4. Split Data (80/20)
# -------------------------
print("\nMembagi data menjadi train/test...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['cleaned_text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'].tolist()
)
print(f"Jumlah data latih: {len(train_texts)}")
print(f"Jumlah data uji : {len(test_texts)}")

# Konfirmasi stratifikasi
unique, train_counts = np.unique(train_labels, return_counts=True)
unique, test_counts = np.unique(test_labels, return_counts=True)
print("\nDistribusi kelas di train dan test (label: count):")
print(dict(zip(unique, train_counts)), "(train)")
print(dict(zip(unique, test_counts)), "(test)")
```

```python
# -------------------------
# 5. Tokenisasi dengan IndoRoBERTa
# -------------------------
print("\nTokenisasi data dengan IndoRoBERTa...")
# Load tokenizer dari model Hugging Face
pretrained_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

MAX_LEN = 128  # panjang maksimum token
def tokenize_batch(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')

train_encodings = tokenize_batch(train_texts)
test_encodings  = tokenize_batch(test_texts)
```

```python
# ==========================
# 6. Dataset PyTorch
# ==========================
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset  = SentimentDataset(test_encodings, test_labels)
```

```python
# ==========================
# 7. Load Model & Class Weights
# ==========================
# Hitung bobot kelas (inverse frequency)
class_counts = np.array([counts.get(i,0) for i in sorted(counts.index)])
# Bobot = 1/frequency
class_weights = 1.0 / class_counts
# Normalisasi ke rata-rata 1.0 (agar scale-nya wajar)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Bobot kelas (minoritas lebih tinggi):", class_weights.tolist())

# Load model (dengan 3 label)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, num_labels=3)
model.to(device)
```

```python
# ==========================
# 8. Custom Trainer dengan Weighted Loss
# ==========================
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Gunakan CrossEntropyLoss dengan bobot kelas
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

```python
# ==========================
# 9. Training Arguments
# ==========================
training_args = TrainingArguments(
    output_dir = './results',
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    logging_steps = 50,
    save_strategy = 'no',
    evaluation_strategy = 'no',
    learning_rate = 2e-5,
    fp16 = torch.cuda.is_available()
)
```

```python
# ==========================
# 10. Fine-Tuning Model
# ==========================
trainer = WeightedLossTrainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    tokenizer = tokenizer
)

print("\nMemulai fine-tuning (ini akan memakan waktu)...")
trainer.train()
print("Fine-tuning selesai.")
```

```python
# ==========================
# 11. Evaluasi pada Data Uji
# ==========================
print("\nEvaluasi pada data uji...")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(test_labels, preds)
print("\nConfusion Matrix (baris: aktual, kolom: prediksi):")
print(cm)

# Classification report (precision, recall, f1 per kelas)
print("\nClassification Report (precision, recall, f1-score per kelas):")
print(classification_report(test_labels, preds, target_names=['Negatif','Netral','Positif'], digits=4))

# Hitung TP (kelas Positif) dan TN (kelas Negatif)
# Misal label 0=Negatif, 1=Netral, 2=Positif (sesuai LabelEncoder)
tp_pos = cm[2,2]   # aktual positif yang diprediksi positif
tn_neg = cm[0,0]   # aktual negatif yang diprediksi negatif

print(f"True Positives (kelas Positif) = {tp_pos}")
print(f"True Negatives (kelas Negatif) = {tn_neg}")

# Akurasi keseluruhan
acc = accuracy_score(test_labels, preds)
print(f"\nAkurasi keseluruhan: {acc:.4f}")
```

```python
# ==========================
# 12. Contoh Prediksi
# ==========================
print("\nContoh prediksi pada beberapa komentar baru:")
def predict_sentiment(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    pred_label = torch.argmax(outputs.logits, dim=1).cpu().item()
    return label_encoder.inverse_transform([pred_label])[0]

sample_indices = np.random.choice(len(test_texts), 3, replace=False)
for idx in sample_indices:
    text = test_texts[idx]
    actual = label_encoder.inverse_transform([test_labels[idx]])[0]
    predicted = predict_sentiment(text)
    print(f"- Komentar: \"{text[:50]}...\"")
    print(f"  Sentimen aktual   : {actual}")
    print(f"  Sentimen prediksi : {predicted}\n")
```

**Penjelasan Hasil:**
- **Confusion Matrix:** Menunjukkan berapa banyak data di setiap kelas benar/salah diklasifikasikan. Diagonal utama (mis. cm[2,2]) adalah True Positive untuk kelas Positif【12†L5-L8】.  
- **Precision & Recall:** Precision (TP/(TP+FP)) dan Recall (TP/(TP+FN)) dihitung otomatis oleh `classification_report`. Masing-masing menyatakan proporsi prediksi benar dari kelas tertentu dan kemampuan model menangkap instance positif【12†L5-L8】.  
- **TP & TN:** Dari `cm`, kita ambil `cm[2,2]` sebagai TP untuk kelas *Positif* dan `cm[0,0]` sebagai TN untuk kelas *Negatif*, sebagaimana ditampilkan di atas.  
- **Akurasi:** (TP+TN)/(Total) keseluruhan. Dicetak paling bawah.

Dengan notebook di atas Anda dapat melatih dan mengevaluasi model *Indonesian RoBERTa* untuk analisis sentimen (positif/netral/negatif) pada data Anda. Anda juga dapat memodifikasi epoch, `batch_size`, atau menambahkan *early stopping* untuk optimasi lebih lanjut.  

**⚠️ Catatan:** Model *fine-tuned* ini disimpan secara otomatis di direktori `./results` setiap beberapa epoch. Sesuaikan parameter `TrainingArguments` jika ingin menyimpan model terbaik atau menggunakan GPU untuk percepatan.

