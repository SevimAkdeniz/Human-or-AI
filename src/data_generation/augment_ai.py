import pandas as pd
import random
import re
import os

INPUT_PATH = "dataset/ai/ai_raw_original.csv"
OUTPUT_PATH = "dataset/ai/ai_augmented.csv"

FILLERS = [
    "broadly speaking", "in practice", "to some extent", "notably",
    "in particular", "surprisingly", "in many respects", "from a practical standpoint"
]

HEDGES = [
    "may indicate", "could suggest", "possibly", "it appears that",
    "it seems plausible that", "might imply"
]

ACADEMIC_NOISE = [
    "This interpretation is not without limitations.",
    "Although this perspective remains debatable, it offers valuable insight.",
    "Further clarifications may refine the significance of these findings.",
    "This aspect warrants additional examination in future work.",
]

GRAMMAR_SLIPS = [
    lambda s: s.replace("results", "results indicates"),
    lambda s: s.replace("method", "method show"),
    lambda s: s.replace("models", "models seem"),
]

SYNONYMS = {
    "approach": ["method", "strategy", "framework", "procedure"],
    "experiment": ["evaluation", "analysis", "test"],
    "results": ["findings", "outcomes", "observations"],
    "significant": ["substantial", "notable", "considerable"],
    "performance": ["effectiveness", " capability", "quality"],
    "data": ["dataset", "information", "corpus"],
}


def synonym_replace(sentence):
    for word, syns in SYNONYMS.items():
        if word in sentence and random.random() < 0.25:
            sentence = sentence.replace(word, random.choice(syns))
    return sentence


def shuffle_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    random.shuffle(sentences)
    return " ".join(sentences)


def merge_split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)

    # RANDOM split
    if len(sentences) >= 2 and random.random() < 0.3:
        idx = random.randint(0, len(sentences)-1)
        s = sentences[idx]
        words = s.split()
        if len(words) > 6:
            cut = random.randint(3, len(words)-3)
            sentences[idx] = " ".join(words[:cut]) + ". " + " ".join(words[cut:])

    # RANDOM merge
    if len(sentences) >= 3 and random.random() < 0.3:
        i = random.randint(0, len(sentences)-2)
        sentences[i] = sentences[i] + " " + sentences[i+1]
        del sentences[i+1]

    return " ".join(sentences)


def inject_noise(text):
    # Add fillers
    if random.random() < 0.35:
        text += " " + random.choice(FILLERS)

    # Add hedge
    if random.random() < 0.35:
        text += " " + random.choice(HEDGES)

    # Academic noise sentence
    if random.random() < 0.50:
        text += " " + random.choice(ACADEMIC_NOISE)

    # Grammar slips
    if random.random() < 0.25:
        text = random.choice(GRAMMAR_SLIPS)(text)

    # Small punctuation noise
    if random.random() < 0.20:
        text = text.replace(",", " ,")

    return text


def augment_text(text):
    text = synonym_replace(text)
    text = shuffle_sentences(text)
    text = merge_split_sentences(text)
    text = inject_noise(text)
    return text


def main():
    print("📥 Loading original AI dataset...")
    df = pd.read_csv(INPUT_PATH)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    augmented_texts = []
    print("🔧 Augmenting samples...")

    for i, row in df.iterrows():
        print(f"➡️ {i+1}/{len(df)} augmenting...")
        augmented_texts.append(augment_text(row["text"]))

    new_df = pd.DataFrame({
        "text": augmented_texts,
        "label": ["ai"] * len(df)
    })

    new_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\n🎉 AI AUGMENTATION COMPLETED!")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Total: {len(new_df)} rows")


if __name__ == "__main__":
    main()
