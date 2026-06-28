# ==========================================
# HR Employee Attrition Prediction System
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------
# Configure Page
# ---------------------------------------

st.set_page_config(
    page_title="HR Employee Attrition Prediction",
    page_icon="🏢",
    layout="wide"
)
st.markdown("""
<style>
section[data-testid="stSidebar"]{
    width:230px !important;
}
</style>
""", unsafe_allow_html=True) 

# ---------------------------------------
# Load Trained Model
# ---------------------------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# =======================================
# Sidebar
# =======================================

st.sidebar.image(
    "https://img.icons8.com/color/96/office.png",
    width=70
)
st.sidebar.title("📌 Project Information")

with st.sidebar.expander("ℹ About Project", expanded=True):

    st.write("**Machine Learning Model**")
    st.write("Random Forest Classifier")

    st.write("**Dataset**")
    st.write("IBM HR Employee Attrition Dataset")

    st.write("**Developer**")
    st.write("Mezan Akhtar")

st.sidebar.markdown("---")

st.sidebar.metric("🎯 Model Accuracy", "87.07%")

st.sidebar.metric("👥 Dataset Size", "1,470s")

st.sidebar.metric("📊 Features Used", "27")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🛠 Built With")

st.sidebar.markdown("""
- Python
- Streamlit
- Scikit-Learn
- Random Forest
""")

# =======================================
# Main Page
# =======================================

st.title("🏢 HR Employee Attrition Prediction System")

st.write(
    "Predict employee attrition risk using a Machine Learning model trained on the IBM HR Employee Attrition Dataset."
)

st.divider()

# ============================================
# Two Column Layout
# ============================================

left_col, right_col = st.columns([1, 1])

# --------------------------
# LEFT SIDE
# --------------------------

with left_col:

    st.subheader("📝 Employee Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=35
    )

    distance = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
        value=5
    )

    income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=20000,
        value=5000
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        3
    )

    job_level = st.selectbox(
        "Job Level",
        [1,2,3,4,5]
    )

    work_life = st.slider(
        "Work Life Balance",
        1,
        4,
        3
    )

    years_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    overtime = st.radio(
        "OverTime",
        ["No","Yes"]
    )

    department = st.selectbox(
        "Department",
        [
            "Research & Development",
            "Sales",
            "Human Resources"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

# --------------------------
# RIGHT SIDE
# --------------------------

with right_col:

    st.subheader("📊 Prediction Result")

    predict = st.button(
        "🚀 Predict Employee Attrition",
        use_container_width=True
    )

    reset = st.button(
        "🔄 Reset Form",
        use_container_width=True
    )
    
    result_placeholder = st.empty()

# =====================================
# Machine Learning Pipeline
# =====================================
if predict:
    
    input_data = {
    
        "Age": age,
        "DistanceFromHome": distance,
        "EnvironmentSatisfaction": 3,
        "JobInvolvement": 3,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": income,
        "PercentSalaryHike": 15,
        "TotalWorkingYears": years_company,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1
    
    }

    # ======================================
    # Department
    # ======================================
    
    input_data["Department_Research & Development"] = 0
    input_data["Department_Sales"] = 0
    
    if department == "Research & Development":
        input_data["Department_Research & Development"] = 1
    
    elif department == "Sales":
        input_data["Department_Sales"] = 1
    
    
    # ======================================
    # Gender
    # ======================================
    
    input_data["Gender_Male"] = 1 if gender == "Male" else 0
    
    
    # ======================================
    # Marital Status
    # ======================================
    
    input_data["MaritalStatus_Married"] = 1 if marital == "Married" else 0
    
    input_data["MaritalStatus_Single"] = 1 if marital == "Single" else 0
    
    
    # ======================================
    # OverTime
    # ======================================
    
    input_data["OverTime_Yes"] = 1 if overtime == "Yes" else 0
    
    # ======================================
    # Job Role Encoding
    # ======================================
    
    # Initialize all job role features to 0
    input_data["JobRole_Human Resources"] = 0
    input_data["JobRole_Laboratory Technician"] = 0
    input_data["JobRole_Manager"] = 0
    input_data["JobRole_Manufacturing Director"] = 0
    input_data["JobRole_Research Director"] = 0
    input_data["JobRole_Research Scientist"] = 0
    input_data["JobRole_Sales Executive"] = 0
    input_data["JobRole_Sales Representative"] = 0
    
    # Set the selected job role to 1
    
    if job_role == "Human Resources":
        input_data["JobRole_Human Resources"] = 1
    
    elif job_role == "Laboratory Technician":
        input_data["JobRole_Laboratory Technician"] = 1
    
    elif job_role == "Manager":
        input_data["JobRole_Manager"] = 1
    
    elif job_role == "Manufacturing Director":
        input_data["JobRole_Manufacturing Director"] = 1
    
    elif job_role == "Research Director":
        input_data["JobRole_Research Director"] = 1
    
    elif job_role == "Research Scientist":
        input_data["JobRole_Research Scientist"] = 1
    
    elif job_role == "Sales Executive":
        input_data["JobRole_Sales Executive"] = 1
    
    elif job_role == "Sales Representative":
        input_data["JobRole_Sales Representative"] = 1

    # ======================================
    # Convert Dictionary to DataFrame
    # ======================================
    
    input_df = pd.DataFrame([input_data])
    
    # ======================================
    # Arrange Columns in Training Order
    # ======================================
    
    feature_order = [
    
        "Age",
        "DistanceFromHome",
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "PercentSalaryHike",
        "TotalWorkingYears",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
    
        "Department_Research & Development",
        "Department_Sales",
    
        "Gender_Male",
    
        "JobRole_Human Resources",
        "JobRole_Laboratory Technician",
        "JobRole_Manager",
        "JobRole_Manufacturing Director",
        "JobRole_Research Director",
        "JobRole_Research Scientist",
        "JobRole_Sales Executive",
        "JobRole_Sales Representative",
    
        "MaritalStatus_Married",
        "MaritalStatus_Single",
    
        "OverTime_Yes"
    
    ]
    
    input_df = input_df[feature_order]
    
    # ======================================
    # Scale the Input Data
    # ======================================
    
    scaled_data = scaler.transform(input_df)
        
    # ======================================
    # Make Prediction
    # ======================================
    
    prediction = model.predict(scaled_data)
    
    prediction_probability = model.predict_proba(scaled_data)
    
    prediction = prediction[0]
    
    probability = prediction_probability[0][1]
    
    probability_percent = round(probability * 100, 2)
    
    if prediction:
        status = "High Risk of Attrition"
    else:
        status = "Likely to Stay"
    
    # ======================================
    # Risk Level
    # ======================================
    
    if probability_percent >= 70:
        risk = "🔴 High Risk"
    
    elif probability_percent >= 40:
        risk = "🟡 Medium Risk"
    
    else:
        risk = "🟢 Low Risk"
    
    with result_placeholder.container():

        # ======================================
        # Employee Status
        # ======================================
        
        st.subheader("👨‍💼 Employee Status")
        
        if probability_percent >= 70:
            st.error("🚨 High Risk of Attrition")
        
        elif probability_percent >= 40:
            st.warning("⚠ Medium Risk of Attrition")
        
        else:
            st.success("✅ Employee Likely to Stay")
        
        
        st.subheader("📈 Probability")
        st.metric(
            label="Attrition Probability",
            value=f"{probability_percent}%"
        )
        st.progress(probability_percent / 100)
        
        if probability_percent >= 70:
            st.error(risk)
        
        elif probability_percent >= 40:
            st.warning(risk)
        
        else:
            st.success(risk)

            
        st.subheader("💡 HR Recommendations")

        if probability_percent >= 70:
        
            st.info("""
        Recommended Actions
        
        • Schedule a discussion with the employee.
        
        • Review workload and overtime.
        
        • Consider salary revision.
        
        • Improve work-life balance.
        
        • Offer career development opportunities.
        """)
        
        elif probability_percent >= 40:
        
            st.info("""
        Recommended Actions
        
        • Monitor employee engagement.
        
        • Conduct regular one-on-one meetings.
        
        • Encourage training and skill development.
        
        • Review job satisfaction.
        """)
        
        else:
        
            st.info("""
        Recommended Actions
        
        • Employee appears stable.
        
        • Continue regular engagement.
        
        • Recognize good performance.
        
        • Maintain current development plan.
        """)
        
#-----------------------------------------
        # Footer
#-----------------------------------------
st.divider()
st.markdown(
    """
    <center>
    <h5>
    Developed by <b>Mezan Akhtar</b><br>
    AI & Machine Learning Engineer
    </h5>
    </center>
    """,
    unsafe_allow_html=True
)