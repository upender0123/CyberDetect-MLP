import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "raw")
PROC_PATH = os.path.join(DATA_DIR, "processed")
FEATURE_PATH = os.path.join(DATA_DIR, "features")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models")

SPARK_APP = "CyberDetectMLP"
SPARK_MASTER = "local[*]"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
TOP_K_FEATURES = 40
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
