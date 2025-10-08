from sklearn.feature_selection import mutual_info_classif
import pandas as pd, numpy as np, os

def select_top_features(data_path, label_col, k, output_path):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[label_col]), df[label_col]
    mi_scores = mutual_info_classif(X, y, random_state=42)
    top_k = np.argsort(mi_scores)[-k:]
    X_top = X.iloc[:, top_k]
    X_top[label_col] = y
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "features_selected.csv")
    X_top.to_csv(out_file, index=False)
    print(f"✅ Top {k} features saved to {out_file}")
    return out_file
