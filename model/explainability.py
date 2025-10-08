import shap, pandas as pd, matplotlib.pyplot as plt

def explain(model, sample_csv):
    df = pd.read_csv(sample_csv)
    X = df.drop('label', axis=1)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X.sample(200))
    shap.summary_plot(shap_values, X.sample(200), show=False)
    plt.savefig("shap_summary.png", bbox_inches='tight')
    print("✅ SHAP summary plot saved as shap_summary.png")
