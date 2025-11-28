import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# ==========================
# PATH AYARLARI
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("📁 BASE_DIR:", BASE_DIR)
print("📁 MODEL_DIR:", MODEL_DIR)


# ==========================
# TF-IDF YÜKLE
# ==========================
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)


# ==========================
# TÜM MODELLERİ YÜKLE (5 MODEL)
# ==========================
model_paths = {
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "Naive Bayes": os.path.join(MODEL_DIR, "naive_bayes.pkl"),
    "Linear SVC": os.path.join(MODEL_DIR, "linear_svc.pkl"),
    "LightGBM": os.path.join(MODEL_DIR, "lightgbm.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
}

loaded_models = {}
for name, path in model_paths.items():
    try:
        with open(path, "rb") as f:
            loaded_models[name] = pickle.load(f)
        print(f"✔ Model yüklendi: {name}")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {name} — {e}")


# ==========================
# PROBABILITY HESAPLAMA
# ==========================
def compute_probability(model, X):

    # Linear SVC → decision_function (sigmoid)
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        ai_prob = 1 / (1 + np.exp(-score))
        return ai_prob, 1 - ai_prob

    # Diğerleri predict_proba destekler
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        ai_index = list(model.classes_).index("ai")
        ai_prob = proba[ai_index]
        return ai_prob, 1 - ai_prob

    return None, None


# ==========================
# ANA ROUTE
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():
    results = {}

    if request.method == "POST":
        text = request.form["input_text"]
        X = vectorizer.transform([text])

        # Tüm modellerden tahmin al
        for model_name, model in loaded_models.items():
            ai_prob, human_prob = compute_probability(model, X)
            pred = model.predict(X)[0]

            results[model_name] = {
                "prediction": pred,
                "ai_prob": round(ai_prob * 100, 2),
                "human_prob": round(human_prob * 100, 2)
            }

    return render_template("index.html", results=results)


# ==========================
# APP START
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
