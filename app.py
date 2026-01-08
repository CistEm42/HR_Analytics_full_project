import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="HR Attrition Prediction",
    layout="centered"
)

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

model, preprocessor = load_artifacts()

# -----------------------------
# UI
# -----------------------------
st.title(" HR Attrition Prediction App")
st.markdown("Predict whether an employee is likely to **leave or stay**.")

st.divider()

# -----------------------------
# USER INPUTS
# -----------------------------
with st.form("employee_form"):
    st.subheader("Employee Information")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", 18, 65, 30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Department = st.selectbox(
            "Department",
            ["Sales", "Research & Development", "Human Resources"]
        )
        JobRole = st.selectbox(
            "Job Role",
            [
                "Sales Executive", "Research Scientist",
                "Laboratory Technician", "Manufacturing Director",
                "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director",
                "Human Resources"
            ]
        )
        MaritalStatus = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        )

    with col2:
        OverTime = st.selectbox("OverTime", ["Yes", "No"])
        BusinessTravel = st.selectbox(
            "Business Travel",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        )
        DistanceFromHome = st.slider("Distance From Home", 1, 50, 10)
        Education = st.slider("Education (1–5)", 1, 5, 3)
        EducationField = st.selectbox(
            "Education Field",
            [
                "Life Sciences", "Medical", "Marketing",
                "Technical Degree", "Human Resources", "Other"
            ]
        )

    st.subheader("Work & Compensation")

    MonthlyIncome = st.number_input("Monthly Income", 1000, 30000, 5000)
    JobLevel = st.slider("Job Level", 1, 5, 2)
    StockOptionLevel = st.slider("Stock Option Level", 0, 3, 1)
    PercentSalaryHike = st.slider("Percent Salary Hike", 0, 30, 13)

    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 8)
    YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
    YearsInCurrentRole = st.slider("Years in Current Role", 0, 18, 3)
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    YearsWithCurrManager = st.slider("Years With Current Manager", 0, 17, 3)

    st.subheader("Satisfaction Scores")

    JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    WorkLifeBalance = st.slider("Work-Life Balance", 1, 4, 3)
    JobInvolvement = st.slider("Job Involvement", 1, 4, 3)

    PerformanceRating = st.selectbox("Performance Rating", [3, 4])
    TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, 3)
    NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 10, 2)

    submitted = st.form_submit_button("Predict Attrition")

# -----------------------------
# PREDICTION
# -----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Age": Age,
        "DailyRate": 800,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "HourlyRate": 65,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobSatisfaction": JobSatisfaction,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": NumCompaniesWorked,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "BusinessTravel": BusinessTravel,
        "Department": Department,
        "EducationField": EducationField,
        "Gender": Gender,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "OverTime": OverTime
    }])

    X_processed = preprocessor.transform(input_data)

    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]

    st.divider()

    if prediction == 1 or prediction == "Yes":
        st.error(f" High Attrition Risk")
    else:
        st.success(f" Low Attrition Risk")

    st.metric(
        label="Probability of Attrition",
        value=f"{probability * 100:.2f}%"
    )
