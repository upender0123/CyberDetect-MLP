from preprocessing.spark_preprocessor import preprocess_data
from preprocessing.feature_selector import select_top_features
from preprocessing.dataset_splitter import stratified_split
from model.trainer import train_model
from model.evaluator import evaluate
from model.explainability import explain
from pipeline.realtime_detector import simulate_detection
from config import settings

def main():
    preprocessed = preprocess_data("data/raw/ton_iot.csv", "data/processed")
    selected = select_top_features(preprocessed, "label", settings.TOP_K_FEATURES, "data/features")
    train_csv, val_csv, test_csv = stratified_split(selected, "data/features")
    model, le = train_model(train_csv, val_csv, "saved_models")
    evaluate(model, test_csv, le)
    explain(model, test_csv)
    simulate_detection("saved_models/cyberdetect_mlp.h5", test_csv, "alerts.json")

if __name__ == "__main__":
    main()
