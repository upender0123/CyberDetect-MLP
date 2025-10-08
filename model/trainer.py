from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
import pandas as pd, math, os
from .cyberdetect_mlp import build_model

def cosine_annealing(epoch, lr_min=1e-5, lr_max=1e-3, T=25):
    return lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(math.pi*epoch/T))

def train_model(train_csv, val_csv, model_path):
    train_df, val_df = pd.read_csv(train_csv), pd.read_csv(val_csv)
    X_train, y_train = train_df.drop('label', axis=1).values, train_df['label'].values
    X_val, y_val = val_df.drop('label', axis=1).values, val_df['label'].values
    le = LabelEncoder(); y_train = le.fit_transform(y_train); y_val = le.transform(y_val)
    num_classes = len(le.classes_)
    model = build_model(X_train.shape[1], num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lr_sched = LearningRateScheduler(lambda e: cosine_annealing(e))
    early = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=25, batch_size=64, callbacks=[lr_sched,early], verbose=2)
    os.makedirs(model_path, exist_ok=True)
    model.save(f"{model_path}/cyberdetect_mlp.h5")
    print("✅ Model trained and saved.")
    return model, le
