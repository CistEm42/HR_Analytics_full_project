import pandas as pd
from pathlib import Path
import joblib

class Predictions():
    def __init__(self, model_path: Path = Path("artifacts/model.pkl"),
                 processor_path: Path = Path("artifacts/preprocessor.pkl")):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(processor_path)

    def prediction(self, data: pd.DataFrame):
        X_processed = self.preprocessor.transform(data)

        predic = self.model.predict(X_processed)
        probs = self.model.predict_proba(X_processed)[:, 1]

        pred_labels = ["Yes" if p == 1 else "No" for p in predic]

        return pd.DataFrame({
            "Attrition_Prediction": pred_labels,
            "Attrition_Probability": probs
        })