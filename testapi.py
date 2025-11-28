import pandas as pd

df1 = pd.read_csv("dataset/ai/ai_raw_original.csv")
df2 = pd.read_csv("dataset/ai/ai_advanced.csv")

merged = pd.concat([df1, df2], ignore_index=True)

merged.to_csv("merged.csv", index=False)

print("Birleştirme tamamlandı!")
