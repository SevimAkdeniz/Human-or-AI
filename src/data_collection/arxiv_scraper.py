import arxiv
import csv
import time

# Kategoriler ve her biri için alınacak özet sayısı
CATEGORIES = {
    "cs.AI": 500,
    "cs.CL": 500,
    "cs.LG": 500,
    "stat.ML": 500,
    "physics.comp-ph": 500,
    "math.OC": 500
}

OUTPUT_PATH = "dataset/human/human_raw.csv"


def fetch_abstracts():
    total = 0

    # CSV dosyasını hazırlıyoruz
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])

        # Kategorileri sırayla çek
        for category, amount in CATEGORIES.items():
            print(f"\n➡️ {category} kategorisinden {amount} özet çekiliyor...")

            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=amount,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            for result in search.results():
                summary = result.summary.replace("\n", " ").strip()
                writer.writerow([summary, "human"])
                total += 1

            time.sleep(2)   # API'yı korumak için

    print(f"\n✅ Toplam {total} HUMAN özeti başarıyla kaydedildi!")
    print(f"📁 Kaydedilen dosya: {OUTPUT_PATH}")


if __name__ == "__main__":
    fetch_abstracts()
