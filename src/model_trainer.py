import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from pathlib import Path
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import f1_score, r2_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelConfig():
    trained_model_path: Path = Path("artifacts/model.pkl")

class ModelTrainer():
    def __init__(self):
        self.config = ModelConfig()

    def initiate_model_trainer(self, y_true, y_pred, y_prob):
        return {
            "f1_score": f1_score(y_true, y_pred),
            "Recall Score": recall_score(y_true, y_pred),
            "R2-Score": r2_score(y_true, y_pred),
            "Roc_auc": roc_auc_score(y_true, y_prob)
        }
    
    def model_selection(self, X_train, y_train, X_test, y_test):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        models = {
            "Random Forest": {"model": RandomForestClassifier( 
                     class_weight="balanced",  random_state=42),
                     "params": {
                         "n_estimators": [100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                     }
            },


            "Decision Classifier":{"model": DecisionTreeClassifier(criterion='gini', 
            splitter="best", max_depth=8, class_weight="balanced", random_state=42),
            "params":{
                "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
            }
            },

            "LogisticRegression": {"model":LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42)
            ,
            "params":{
                "C": [0.01,0.1,1,10],
                "penalty":['elasticnet', 'l2'],
                "solver": ["lbfgs"]
            }
         }
        }



        best_model = None
        best_score = -1
        best_model_name = None

        for name, mp in models.items():
            print(f"\nRunning GridSearchCV for {name}...")

            grid = GridSearchCV(
                estimator=mp["model"],
                param_grid=mp["params"],
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                verbose=1
            )

            grid.fit(X_train, y_train)

            print(f"Best CV F1-score: {grid.best_score_:.4f}")
            print(f"Best params: {grid.best_params_}")

            best_estimator = grid.best_estimator_

            y_pred = best_estimator.predict(X_test)
            y_prob = best_estimator.predict_proba(X_test)[:, 1]

            metrics = self.initiate_model_trainer(y_test, y_pred, y_prob)
            score = metrics["f1_score"]

            print("Test set metrics:")

            
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            if score > best_score:
                best_score = score
                best_model = best_estimator
                best_model_name = name

        self.config.trained_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, self.config.trained_model_path)

        

        print("\n===================================")
        print(f"Best model overall: {best_model_name}")
        print(f"Test F1-score    : {best_score:.4f}")
        print(f"Saved to         : {self.config.trained_model_path}")
       

        return best_model, best_model_name

                      



        