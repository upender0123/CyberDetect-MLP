from sklearn.model_selection import train_test_split
import pandas as pd, os

def stratified_split(data_path, output_path):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=['label']), df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
    os.makedirs(output_path, exist_ok=True)
    X_train.assign(label=y_train).to_csv(f"{output_path}/train.csv", index=False)
    X_val.assign(label=y_val).to_csv(f"{output_path}/val.csv", index=False)
    X_test.assign(label=y_test).to_csv(f"{output_path}/test.csv", index=False)
    print("✅ Train/Val/Test sets created.")
    return (f"{output_path}/train.csv", f"{output_path}/val.csv", f"{output_path}/test.csv")
