# CyberDetect-MLP
A Big Data-Enabled Optimized Deep Learning Framework for Scalable
Cyberattack Detection in IoT Environments
CyberDetect-MLP is a modular, big-data-driven deep learning framework that integrates Apache
Spark and TensorFlow for distributed preprocessing, optimized training, and explainable cyberattack
detection in IoT networks.
It supports scalable data ingestion, feature selection, model optimization using cosine-annealing,
SHAP-based interpretability, and real-time JSON alert generation.

## Features
Spark-based preprocessing and mutual-information feature selection
Optimized MLP model for large-scale IoT attack detection
Explainability through SHAP visualizations
JSON-based real-time alert generation
Modular and extensible architecture

⚙️ Installation
git clone https://github.com/upender0123/CyberDetect-MLP.git
cd CyberDetectMLP
pip install -r requirements.txt

## Usage
1. Place your dataset file as data/raw/ton_iot.csv
2. Run the complete workflow:
3. python main.py
4. Outputs include:
o Processed data: data/processed/preprocessed.csv
o Selected features: data/features/features_selected.csv
o Trained model: saved_models/cyberdetect_mlp.h5
o Explainability plot: shap_summary.png
o Alerts: alerts.json

## Citation
If you use this framework in your research, please cite:

@article{CyberDetectMLP2025,
title={CyberDetect-MLP: A Big Data-Enabled Optimized Deep Learning Framework for Scalable
Cyberattack Detection in IoT Environments},
author={Talluri Upender , Dr. M. Neelakantappa , Dr. C. Prakasa Rao , Dr. Jaideep Gera  , Vuyyuru Lakshma
Reddy , Nagendar Yamsani },
year={2025},
journal={Under Review, SCI Journal, Scientific Reports}
}

## License
Released under the MIT License.
Keywords: cybersecurity · IoT · deep learning · big data · explainable AI
