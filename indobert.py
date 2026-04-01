# -*- coding: utf-8 -*-
"""
Analisis Sentimen Komentar Instagram Menggunakan IndoBERT (Optimized)
--------------------------------------------------------------------
Target: Precision, Recall, F1 > 90% untuk semua kelas
"""

import pandas as pd
import numpy as np
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, WeightedRandomSampler
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import warnings
warnings.filterwarnings('ignore')

# ==========================
# 1. Persiapan Lingkungan
# ==========================
nltk.download('stopwords')
nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan perangkat: {device}")

# ==========================
# 2. Load dan Persiapan Data
# ==========================
print("\nMemuat data...")
df = pd.read_excel('dataset/DataGabungan.xlsx')  # Sesuaikan path file
print(f"Jumlah data: {len(df)}")
print("Kolom yang tersedia:", df.columns.tolist())

# Pastikan kolom yang diperlukan ada
required_cols = ['Isi komentar', 'Sentimen']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"File Excel harus memiliki kolom: {required_cols}")

# Ambil data yang diperlukan, hapus baris dengan nilai kosong
df = df[required_cols].dropna()

# ==========================
# 3. Preprocessing Teks
# ==========================
print("\nMelakukan preprocessing teks...")

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(nltk.corpus.stopwords.words('indonesian'))

def clean_text(text):
    """Membersihkan teks komentar untuk pelatihan."""
    text = str(text).lower()
    text = re.sub(r'@\w+|http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned_text'] = df['Isi komentar'].apply(clean_text)

# Tampilkan contoh hasil cleaning
print("\nContoh hasil preprocessing:")
for i in range(min(3, len(df))):
    print(f"Original : {df['Isi komentar'].iloc[i][:100]}...")
    print(f"Cleaned  : {df['cleaned_text'].iloc[i][:100]}...")
    print()

# ==========================
# 4. Encode Label
# ==========================
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Sentimen'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Pemetaan sentimen ke label integer:", label_mapping)

# Cek distribusi kelas
class_counts = df['label'].value_counts().sort_index()
print("\nDistribusi kelas:")
for i, count in class_counts.items():
    print(f"Kelas {i} ({label_encoder.inverse_transform([i])[0]}): {count} ({count/len(df)*100:.2f}%)")

# ==========================
# 5. Split Data (Stratified)
# ==========================
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['cleaned_text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'].tolist()
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"\nUkuran data: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

# ==========================
# 6. Tokenisasi dengan IndoBERT
# ==========================
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
MAX_LEN = 128  # Dapat ditingkatkan jika diperlukan (misal 256)

def tokenize(texts, labels):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    return encodings, labels

train_encodings, train_labels = tokenize(train_texts, train_labels)
val_encodings, val_labels = tokenize(val_texts, val_labels)
test_encodings, test_labels = tokenize(test_texts, test_labels)

# ==========================
# 7. Dataset PyTorch
# ==========================
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# ==========================
# 8. Load Model IndoBERT dengan Class Weights
# ==========================
# Hitung class weights (inverse frequency)
class_weights = 1.0 / class_counts.values
class_weights = class_weights / class_weights.sum() * 3  # normalisasi ke 3 kelas
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print("\nClass weights yang digunakan:", class_weights)

# Override loss function untuk menggunakan weighted loss
model = BertForSequenceClassification.from_pretrained(
    'indobenchmark/indobert-base-p1',
    num_labels=3
)

# Ganti loss function dengan weighted cross-entropy
model.classifier = torch.nn.Linear(model.config.hidden_size, 3)
model.to(device)

# Custom Trainer untuk weighted loss
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==========================
# 9. Training Arguments (Optimized)
# ==========================
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,                # Lebih banyak epoch
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,                  # Warmup untuk stabilitas
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',  # Evaluasi berdasarkan akurasi pada val set
    greater_is_better=True,
    save_total_limit=2,
    learning_rate=2e-5,                # Learning rate lebih kecil untuk fine-tuning
    fp16=True if torch.cuda.is_available() else False,  # Mixed precision jika GPU
)

# ==========================
# 10. Compute Metrics
# ==========================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    
    # Format metrics untuk setiap kelas
    metrics = {
        'accuracy': acc,
        'precision_neg': precision[0],
        'recall_neg': recall[0],
        'f1_neg': f1[0],
        'precision_net': precision[1],
        'recall_net': recall[1],
        'f1_net': f1[1],
        'precision_pos': precision[2],
        'recall_pos': recall[2],
        'f1_pos': f1[2],
    }
    return metrics

# ==========================
# 11. Trainer dengan Weighted Loss
# ==========================
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ==========================
# 12. Fine-Tuning Model
# ==========================
print("\nMemulai fine-tuning...")
trainer.train()

# ==========================
# 13. Evaluasi pada Data Uji
# ==========================
print("\nEvaluasi model pada data uji...")
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

# Confusion matrix dan classification report
cm = confusion_matrix(test_labels, preds)
print("\nConfusion Matrix:")
print(cm)

# Visualisasi confusion matrix (opsional)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negatif', 'Netral', 'Positif'],
            yticklabels=['Negatif', 'Netral', 'Positif'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

print("\nClassification Report (per kelas):")
print(classification_report(
    test_labels,
    preds,
    target_names=['Negatif', 'Netral', 'Positif'],
    digits=4
))

# Hitung akurasi
acc = accuracy_score(test_labels, preds)
print(f"Akurasi keseluruhan: {acc:.4f}")

# ==========================
# 14. Simpan Model Terbaik
# ==========================
model_save_path = './indoBERT_sentiment_model'
tokenizer.save_pretrained(model_save_path)
model.save_pretrained(model_save_path)
print(f"\nModel dan tokenizer disimpan di: {model_save_path}")

# ==========================
# 15. Contoh Prediksi untuk Komentar Baru
# ==========================
def predict_sentiment(text):
    """Memprediksi sentimen dari teks komentar."""
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1).cpu().item()
    return label_encoder.inverse_transform([pred])[0]

print("\nContoh prediksi pada komentar dari dataset test:")
# Ambil 5 sampel dari data test untuk ditampilkan
test_indices = np.random.choice(len(test_texts), min(5, len(test_texts)), replace=False)
for idx in test_indices:
    komentar = test_texts[idx]
    actual = label_encoder.inverse_transform([test_labels[idx]])[0]
    predicted = predict_sentiment(komentar)
    print(f"Komentar: {komentar[:80]}...")
    print(f"Sentimen Asli  : {actual}")
    print(f"Sentimen Prediksi: {predicted}")
    print()

print("Selesai!")