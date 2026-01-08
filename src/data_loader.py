import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split

# import seaborn as sns
# import matplotlib.pyplot as plt

@dataclass(frozen=True)
class DataIngestionConfig():
    artifacts_dir: Path = Path("artifacts")
    raw_data_path: Path = artifacts_dir / "raw.csv"
    train_data_path: Path = artifacts_dir / "train.csv"
    test_data_path: Path = artifacts_dir / "test.csv"

def ingest_data(config: DataIngestionConfig, source_path: Path) -> None:
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(source_path)
        data.to_csv(config.raw_data_path, index=False)
        
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Attrition'])

        train_data.to_csv(config.train_data_path, index=False)
        test_data.to_csv(config.test_data_path, index=False)

        
    
    

