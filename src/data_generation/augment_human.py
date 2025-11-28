import pandas as pd
import random
import re
import os

INPUT_PATH = "dataset/human/human_raw.csv"
OUTPUT_PATH = "dataset/human/human_augmented.csv"

# HUMAN için synonym değişimleri — teknikliği biraz düşürür
SYNONYMS = {
    "approach": ["method", "technique", "way"],
    "demonstrate": ["show", "indicate", "suggest"],
    "significant": ["notable", "important", "meaningful"],
    "results": ["findings", "outcomes"],
    "analysis": ["study", "examination"],
    "performance": ["effectiveness", "output"],
    "model": ["system", "architecture"],
}

# Filler words — insanlaştırır
FILLERS = [
    "to some extent", "in practice", "in reality",
    "broadly speaking", "as it happens", "in many cases"
]

# Grammar slips — arXiv’in mükemmelliğini kırmak için
GRAMMAR_SLIPS = [
    lambda s: s.replace("results", "results shows"),
    lambda s: s.replace("models", "models seem to"),
    lambda s: s.replace("method", "method show"),
]

def synonym_replace(text):
    for word, syns in SYNONYMS.items():
        if word in text and random.random() < 0.25:
            text = text.replace(word, random.choice(syns))
    return text

def shuffle_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) > 2:
        random.shuffle(sentences)
    return " ".join(sentences)

def split_merge(text):
    sentences = re.split(r'(?<=[.!?]) +', text)

    # split random
    if random.random() < 0.4:
        idx = random.randint(0, len(sentences)-1)
        words = sentences[idx].split()
        if len(words) > 8:
            cut = random.randint(3, len(words)-4)
            sentences[idx] = " ".join(words[:cut]) + ". " + " ".join(words[cut:])

    # merge random
    if len(sentences) > 2 and random.random() < 0.3:
        i = random.randint(0, len(sentences)-2)
        sentences[i] = sentences[i] + " " + sentences[i+1]
        del sentences[i+1]

    return " ".join(sentences)

def inject_noise(text):
    if random.random() < 0.35:
        text += " " + random.choice(FILLERS)
    if random.random() < 0.2:
        text = random.choice(GRAMMAR_SLIPS)(text)
    if random.random() < 0.15:
        text = text.replace(",", " ,")
    return text

def augment(text):
    text = synonym_replace(text)
    text = shuffle_sentences(text)
    text = split_merge(text)
    text = inject_noise(text)
    return text

def main():
    print("📥 Loading HUMAN dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("🔧 Augmenting human samples...")
    augmented = []

    for i, row in df.iterrows():
        print(f"➡️ {i+1}/{len(df)}")
        augmented.append(augment(row["text"]))

    new_df = pd.DataFrame({
        "text": augmented,
        "label": ["human"] * len(augmented)
    })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    new_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\n🎉 HUMAN AUGMENTATION COMPLETE!")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
