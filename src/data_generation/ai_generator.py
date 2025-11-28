import csv
import os
import random

OUTPUT_PATH = "dataset/ai/ai_raw.csv"
N_SAMPLES = 3000

DOMAINS = [
    "machine learning", "natural language processing", "computer vision",
    "reinforcement learning", "data mining", "cybersecurity",
    "bioinformatics", "quantum computing", "autonomous systems"
]

PROBLEMS = [
    "handling noisy data", "improving generalization",
    "scaling models to large datasets", "capturing long-range dependencies",
    "reducing model complexity", "ensuring robustness under shifts"
]

METHODS = [
    "a transformer-inspired model", "a graph-based framework",
    "a hybrid optimization strategy", "a meta-learning system",
    "a self-supervised representation method"
]

DATASETS = [
    "real-world experimental data", "benchmark corpora",
    "multilingual datasets", "heterogeneous multimodal data"
]

RESULTS = [
    "achieves competitive performance", 
    "outperforms existing baselines",
    "shows promising improvements",
    "reduces error across multiple metrics"
]

# HUMAN-LIKE NOISE SETS
FILLERS = [
    "broadly speaking", "in practice", "to some extent", 
    "notably", "in particular", "surprisingly"
]

HEDGES = [
    "may indicate", "could suggest", "possibly", 
    "it appears that", "it seems plausible that"
]

ACADEMIC_NOISE = [
    "This aspect remains debatable.",
    "Further clarifications are needed.",
    "Future studies may refine this direction.",
    "Interpretation of results should be cautious.",
]

def inject_noise(sentence: str) -> str:
    """Add small human-like noise to a sentence."""
    # 1) Maybe add a filler
    if random.random() < 0.25:
        sentence += " " + random.choice(FILLERS)

    # 2) Maybe add a hedge
    if random.random() < 0.25:
        sentence += " " + random.choice(HEDGES)

    # 3) Maybe grammar slip
    if random.random() < 0.12:
        sentence = sentence.replace("results", "results indicates")
        sentence = sentence.replace("method", "method show")

    return sentence


def generate_abstract():
    domain = random.choice(DOMAINS)
    problem = random.choice(PROBLEMS)
    method = random.choice(METHODS)
    dataset = random.choice(DATASETS)
    result = random.choice(RESULTS)

    # Create base abstract structure
    sentences = [
        f"This study examines {problem} within the broader field of {domain}.",
        f"To address these challenges, we propose {method} as a potential direction.",
        f"The model is evaluated using {dataset}, representing varied real-world characteristics.",
        f"The empirical analysis {result}, although certain limitations remain.",
        f"Overall, these findings contribute to the growing discussion in {domain} research."
    ]

    # Inject noise
    noisy_sentences = [inject_noise(s) for s in sentences]

    # Add a random academic noise sentence
    if random.random() < 0.6:
        noisy_sentences.append(random.choice(ACADEMIC_NOISE))

    return " ".join(noisy_sentences)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for i in range(N_SAMPLES):
            print(f"➡️ {i+1}/{N_SAMPLES} generating with noise...")
            abstract = generate_abstract()
            writer.writerow([abstract, "ai"])

    print("\n🎉 AI dataset (with noise) successfully generated!")
    print(f"Saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
