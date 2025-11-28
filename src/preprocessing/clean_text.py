import pandas as pd
import re
import os
from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords 

# NLTK ve STOP_WORDS indirme/tanımlama kısmı aynı kaldı.
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

HUMAN_PATH = "dataset/human/human_augmented.csv"
AI_PATH = "dataset/ai/merged.csv"
OUTPUT_PATH = "dataset/cleaned/cleaned_dataset_new.csv"


def clean_text(text: str) -> str:
    """Metin temizleme fonksiyonu (Son Agresif Versiyon)."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    
    # Rakamlar, noktalamalar, özel karakterler temizlenir.
    text = re.sub(r'[^a-z\s]', ' ', text) 

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    text = " ".join(tokens)
    
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def load_and_clean(path, label):
    # 🚩 KRİTİK NOKTA 1: Sadece 'text' kolonunu okuyarak olası diğer kolon sızıntılarını engelle
    # Eğer ham CSV'lerinizde başka kolonlar varsa, onları görmezden gelir.
    df = pd.read_csv(path, usecols=['text'])
    
    # text kolonunu temizle
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label"] = label

    # boş veya çok kısa metinleri at
    df = df[df["text"].str.len() > 30]

    print(f"Loaded {label} data: {len(df)} rows.")
    return df


def main():
    print("📥 Importing raw datasets...")
    human_df = load_and_clean(HUMAN_PATH, "human")
    ai_df = load_and_clean(AI_PATH, "ai")

    print("🔄 Merging...")
    full_df = pd.concat([human_df, ai_df], ignore_index=True)
    
    # 🚩 KRİTİK NOKTA 2: Tüm kolonları kontrol et (Sadece 'text' ve 'label' kalmalı)
    if list(full_df.columns) != ['text', 'label']:
        print(f"⚠️ DİKKAT: DataFrame'de beklenmedik kolonlar var: {list(full_df.columns)}")
        # Sadece gerekli kolonları tutarak sızıntı kaynağını ele (Örn. eski bir index kolonu)
        full_df = full_df[['text', 'label']]
        print("Kolonlar sadece 'text' ve 'label' olarak filtrelendi.")


    print("🔀 Shuffling...")
    # KRİTİK NOKTA 3: shuffle sonrası index'leri sıfırlamak
    full_df = shuffle(full_df).reset_index(drop=True)

    print("📁 Saving cleaned dataset...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # index=False ile kaydederek gereksiz index kolonlarının oluşmasını engelle
    full_df.to_csv(OUTPUT_PATH, index=False)

    print("\n🎉 SUCCESS! Cleaned dataset created:")
    print(f"  → {OUTPUT_PATH}")
    print(f"  Total rows: {len(full_df)}")
    print(f"  Final columns: {list(full_df.columns)}") # Son kontrol

if __name__ == "__main__":
    main()