import csv
import os
import random
from openai import OpenAI
from dotenv import load_dotenv

# .env'den API key yükle
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_PATH = "dataset/ai/ai_advanced.csv"
N_SAMPLES = 400  # maliyeti düşük tutmak için ~400 yeterli

TOPICS = [
    "deep learning optimization",
    "natural language processing",
    "large language models",
    "computer vision for medical imaging",
    "graph neural networks",
    "reinforcement learning for control",
    "federated learning and privacy",
    "time-series forecasting in finance",
    "anomaly detection in cybersecurity",
    "bioinformatics sequence modeling",
    "multimodal learning with text and images",
    "self-supervised learning on large datasets",
]


def build_prompt(topic: str) -> str:
    return f"""
You are an expert researcher writing a conference paper abstract.

Write a fully ORIGINAL scientific abstract (about 180–230 words)
on the topic: "{topic}".

MUST:
- sound like a real human-written arXiv abstract
- include: motivation → research gap → proposed method → experimental setup → results → conclusion
- use technical terms, dataset names (e.g. ImageNet, CIFAR-10, WikiText-103, etc.), or realistic synthetic datasets
- mention at least ONE concrete number (e.g. 92.3%, 10^5 samples, 32 layers, etc.)
- optionally include formula-like fragments (e.g. O(n^2), x^2, λ, σ)
- optionally include citation-style tokens like [1], [2] or (2021)
- vary sentence lengths; mix long and short sentences
- avoid typical generic AI phrases like "In this paper, we explore..." or "The results demonstrate the effectiveness..."

DO NOT:
- copy from real existing papers
- be too polished and robotic; allow slightly messy, human-like style
- use bullet points or lists; write ONE single paragraph.
"""


def generate_abstract(topic: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": build_prompt(topic)}],
        temperature=0.95,
        max_tokens=380,
    )
    return response.choices[0].message.content


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for i in range(N_SAMPLES):
            topic = random.choice(TOPICS)
            print(f"➡️ {i+1}/{N_SAMPLES} generating advanced abstract for: {topic}")
            abstract = generate_abstract(topic)
            writer.writerow([abstract, "ai"])

    print("\n🎉 ADVANCED AI DATASET generated successfully!")
    print(f"📁 Saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
