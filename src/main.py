from src.data_loader import DataIngestionConfig, ingest_data
from pathlib import Path
from src.preprocessing import Preprocessing
from src.data_loader import ingest_data,DataIngestionConfig
from src.model_trainer import ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    data_path = Path("data/HR_Analytics.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"Dataset loaded: {data.shape}")

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["Attrition"]
    )

    print(f"Train size is: {train_data.shape}")
    print(f"Test size is: {test_data.shape}")

    preprocessing = Preprocessing()

    X_train, y_train = preprocessing.fit_transform(train_data)
    X_test, y_test = preprocessing.transform(test_data)

    model_trainer = ModelTrainer()
    best_model, best_model_name = model_trainer.model_selection(
        X_train, y_train, X_test, y_test
    )

    print(" Model training completed")
    print(f" Best model: {best_model_name}")
    print(f" Best model: {best_model}")

    print("Pipeline finished successfully")

   
   
   
   
   
    # config = DataIngestionConfig()

    # ingest_data(config=config, source_path=Path("data/HR_Analytics.csv"))

# def transformed(data):
#     X_processed, y = Preprocessing.fit_transform(data)

#     print("Preprocessing completed")
#     print("X shape:", X_processed.shape)
#     print("y distribution:")
#     print(y.value_counts())



if __name__ == "__main__":
    main()