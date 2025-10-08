import pandas as pd, json
from tensorflow.keras.models import load_model

def simulate_detection(model_path, input_csv, output_json):
    model = load_model(model_path)
    df = pd.read_csv(input_csv)
    X = df.drop('label', axis=1).values
    preds = model.predict(X)
    labels, probs = preds.argmax(axis=1), preds.max(axis=1)
    results = [{"id": i, "predicted_class": int(labels[i]), "confidence": float(probs[i])} for i in range(len(labels))]
    with open(output_json, "w") as f: json.dump(results, f, indent=2)
    print(f"✅ Alerts generated and saved to {output_json}")
