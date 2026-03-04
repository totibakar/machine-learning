import pandas as pd
import numpy as np
import re
import string
import pickle

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# 1. LOAD DATA (MAX 1000)
# =========================
print("Loading dataset...")
df = pd.read_csv("dataset.csv", sep=';', encoding='latin-1')

# Ambil maksimal 1000 data
df = df.sample(n=1000, random_state=42)

print("Jumlah data awal:", len(df))

# =========================
# 2. LABELING
# =========================
def rating_to_sentiment(rating):
    if rating >= 4:
        return 1   # Positif
    elif rating <= 2:
        return 0   # Negatif
    else:
        return np.nan  # Rating 3 dihapus

df['sentiment'] = df['rating'].apply(rating_to_sentiment)
df = df.dropna()

print("Jumlah data setelah hapus rating 3:", len(df))

# =========================
# 3. PREPROCESSING
# =========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = str(text).lower()  # Case folding
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus angka & tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    text = stemmer.stem(text)  # Stemming
    return text

print("Preprocessing text...")
df['clean_review'] = df['review'].apply(clean_text)

# =========================
# 4. CEK IMBALANCE
# =========================
print("\nDistribusi Sentimen:")
print(df['sentiment'].value_counts())

# =========================
# 5. SPLIT DATA
# =========================
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. TF-IDF
# =========================
print("Vectorizing...")
tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# 7. MODEL TRAINING
# =========================
print("Training model...")
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# =========================
# 8. EVALUATION
# =========================
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== HASIL EVALUASI ===")
print("Akurasi:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 9. PERSENTASE AKHIR
# =========================
positif = (df['sentiment'] == 1).sum()
negatif = (df['sentiment'] == 0).sum()
total = len(df)

print("\n=== PERSENTASE DATA ===")
print("Positif:", round((positif/total)*100, 2), "%")
print("Negatif:", round((negatif/total)*100, 2), "%")

# =========================
# 10. SIMPAN MODEL
# =========================
with open("model_sentiment.pkl", "wb") as f:
    pickle.dump((model, tfidf), f)

print("\nModel berhasil disimpan sebagai model_sentiment.pkl")

# =========================
# 11. PREDIKSI MANUAL
# =========================
def predict_sentiment(text):
    text_clean = clean_text(text)
    vector = tfidf.transform([text_clean])
    prediction = model.predict(vector)[0]
    return "Positif" if prediction == 1 else "Negatif"

print("\nContoh prediksi manual:")
print(predict_sentiment("Produk ini sangat bagus dan saya suka sekali"))
print(predict_sentiment("Barangnya jelek dan sangat mengecewakan"))