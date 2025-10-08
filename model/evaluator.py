from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd, tensorflow as tf

def evaluate(model, test_csv, label_encoder):
    df = pd.read_csv(test_csv)
    X, y_true = df.drop('label', axis=1).values, label_encoder.transform(df['label'])
    y_pred_prob = model.predict(X)
    y_pred = y_pred_prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    roc = roc_auc_score(tf.keras.utils.to_categorical(y_true), y_pred_prob, average='macro', multi_class='ovr')
    print(f"✅ Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} ROC-AUC={roc:.4f}")
    return acc, prec, rec, f1, roc
