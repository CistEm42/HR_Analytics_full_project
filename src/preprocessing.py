import os
import sys
from pathlib import Path
import joblib
from dataclasses import dataclass
from src.data_loader import DataIngestionConfig,ingest_data
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer



@dataclass(frozen=True)
class DataProcessingConfig():
    preprocessor_obj_file_path: Path = Path("artifacts", "preprocessor.pkl")
    target_column: str = "Attrition"
    numerical_features: tuple = (
        "Age", "DailyRate", "DistanceFromHome", "Education",
        "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
        "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
        "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StockOptionLevel",
        "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
        "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    )
    categorical_features: tuple = (
        "BusinessTravel", "Department", "EducationField",
        "Gender", "JobRole", "MaritalStatus", "OverTime"
    )

class Preprocessing:
    def __init__(self):
        self.config=DataProcessingConfig()
        self.label_encoder = LabelEncoder()

    def do_preprocessing(self):

        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())

        ])

        categorical_pipeline = Pipeline (steps=[
            ("imputer", SimpleImputer (strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])


        return ColumnTransformer(transformers=[
            ("num", numerical_pipeline, self.config.numerical_features),
            ("cat", categorical_pipeline, self.config.categorical_features)
        ])
    
    def _get_feature_names(self):
        try:
            return self.preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = []

            num_features = self.config.numerical_features
            cat_encoder = self.preprocessor.named_transformers_["cat"] \
                .named_steps["one_hot_encoder"]

            cat_features = cat_encoder.get_feature_names(self.config.categorical_features)

            feature_names.extend(num_features)
            feature_names.extend(cat_features)

            return feature_names
    
    def fit_transform(self, data: pd.DataFrame):

        X = data.drop(columns=[self.config.target_column])
        y = data[self.config.target_column]

        label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        self.preprocessor = self.do_preprocessing()
        X_processed = self.preprocessor.fit_transform(X)


        self.config.preprocessor_obj_file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, self.config.preprocessor_obj_file_path)

        feature_names = self._get_feature_names()
        X_processed = pd.DataFrame(X_processed, columns=feature_names)

        return X_processed, y
    
    def transform(self, data: pd.DataFrame):
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform() first.")

        X = data.drop(columns=[self.config.target_column])
        y = data[self.config.target_column]

        y = self.label_encoder.transform(y)

        X_processed = self.preprocessor.transform(X)
        feature_names = self._get_feature_names()
        X_processed = pd.DataFrame(X_processed, columns=feature_names)

        return X_processed, y




        # train_var = pd.read_csv(train_path)
        # test_var = pd.read_csv(test_path)

        # preprocessing_obj = self.do_preprocessing()
        # Target_column = "Attrition"

        