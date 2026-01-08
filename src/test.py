from src.predictor import Predictions
import pandas as pd

# Example input (1 employee)
sample = pd.DataFrame([{
    "Age": 35,
    "DailyRate": 1100,
    "DistanceFromHome": 10,
    "Education": 3,
    "EnvironmentSatisfaction": 3,
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 4,
    "MonthlyIncome": 6000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 12,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 4,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Sales",
    "EducationField": "Life Sciences",
    "Gender": "Male",
    "JobRole": "Sales Executive",
    "MaritalStatus": "Single",
    "OverTime": "Yes"
}])

predictor = Predictions()
result = predictor.prediction(sample)

print(result)