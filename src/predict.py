import pickle
import os
import math

MODEL_DIR = "models/"

def load_models():

    # Load vectorizer
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    models = {}

    # Logistic Regression
    with open(os.path.join(MODEL_DIR, "logistic_regression.pkl"), "rb") as f:
        models["Logistic Regression"] = pickle.load(f)

    # Naive Bayes
    with open(os.path.join(MODEL_DIR, "naive_bayes.pkl"), "rb") as f:
        models["Naive Bayes"] = pickle.load(f)

    # Linear SVC
    with open(os.path.join(MODEL_DIR, "linear_svc.pkl"), "rb") as f:
        models["Linear SVC"] = pickle.load(f)

    # LightGBM
    with open(os.path.join(MODEL_DIR, "lightgbm.pkl"), "rb") as f:
        models["LightGBM"] = pickle.load(f)

    # Random Forest
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
        models["Random Forest"] = pickle.load(f)

    return vectorizer, models


def calibrate(human_prob, ai_prob):

    ai_prob = ai_prob ** 0.45
    human_prob = human_prob ** 1.3

    total = ai_prob + human_prob
    ai_prob /= total
    human_prob /= total

    return human_prob, ai_prob


def predict_all(text):

    vectorizer, models = load_models()
    X = vectorizer.transform([text])

    results = {}

    for name, model in models.items():

        # SVC probability yok → manual hesap
        if name == "Linear SVC":
            decision = model.decision_function(X)[0]
            ai_prob = 1 / (1 + math.exp(-decision))
            human_prob = 1 - ai_prob

        else:
            raw = model.predict_proba(X)[0]
            human_prob = raw[0]
            ai_prob = raw[1]

        # calibrate
        human_prob, ai_prob = calibrate(human_prob, ai_prob)

        label = "ai" if ai_prob > 0.5 else "human"

        results[name] = {
            "prediction": label,
            "ai_prob": round(ai_prob * 100, 2),
            "human_prob": round(human_prob * 100, 2)
        }

    return results


# Test usage
if __name__ == "__main__":
    text = input("Metin: ")
    print(predict_all(text))
