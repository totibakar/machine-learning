import pandas as pd
import numpy as np
import re
import string
import pickle
import matplotlib.pyplot as plt  # Ditambahkan untuk visualisasi
import seaborn as sns            # Ditambahkan untuk visualisasi

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load 1000 Data
print("Loading dataset")
df = pd.read_csv(r"Sample_Data\dataset.csv", sep=';', encoding='latin-1')

# Ambil maksimal 1000 data
df = df.sample(n=1000, random_state=42)

print("Jumlah data awal:", len(df))

# Labeling Sentimen
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

# Preprocessing
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Cache untuk mempercepat proses stemming
stem_cache = {}

def clean_text(text):
    text = str(text).lower()  # Case folding
    text = re.sub(r'[^a-z\s]', '', text)  # Hapus angka & tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    
    # Stemming
    stemmed_words = []
    for word in text.split():
        if word not in stem_cache:
            stem_cache[word] = stemmer.stem(word)  # Stemming
        stemmed_words.append(stem_cache[word])
    text = ' '.join(stemmed_words)
    
    return text

print("Preprocessing text...")
df['clean_review'] = df['review'].apply(clean_text)

# Tampilkan 5 data pertama untuk mengecek hasil cleaning
df[['review', 'clean_review', 'sentiment']].head().to_csv("data_preview.csv", index=False)
print("\nPreview data berhasil disimpan sebagai 'data_preview.csv'")

# Cek Imbalance
print("\nDistribusi Sentimen:")
print(df['sentiment'].value_counts())

# Split Data
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
print("Vectorizing...")
tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Training
print("Training model...")
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Evaluation 
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nHASIL EVALUASi")
print("Akurasi:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualisasi Distribusi Sentimen
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df)
plt.title('Distribusi Sentimen (0 = Negatif, 1 = Positif)')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Data')
plt.show()


# Visualisasi Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negatif', 'Positif'], 
            yticklabels=['Negatif', 'Positif'])
plt.title('Confusion Matrix')
plt.xlabel('Prediksi Model')
plt.ylabel('Data Asli')
plt.show()

# Presentase Akhir
positif = (df['sentiment'] == 1).sum()
negatif = (df['sentiment'] == 0).sum()
total = len(df)

print("\nPERSENTASE DATA")
print("Positif:", round((positif/total)*100, 2), "%")
print("Negatif:", round((negatif/total)*100, 2), "%")

# Simpan Model
with open("model_sentiment.pkl", "wb") as f:
    pickle.dump((model, tfidf), f)

print("\nModel berhasil disimpan sebagai model_sentiment.pkl")

# Prediksi Manual
def predict_sentiment(text):
    text_clean = clean_text(text)
    vector = tfidf.transform([text_clean])
    prediction = model.predict(vector)[0]
    return "Positif" if prediction == 1 else "Negatif"

print("\nPrediksi manual:")
print(predict_sentiment("Produk ini sangat bagus dan saya suka sekali"))
print(predict_sentiment("Barangnya jelek dan sangat mengecewakan"))