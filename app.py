
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.joblib")

model = load_model()

st.title("🏦 Loan Approval Predictor")
st.write(
    "Enter applicant details to get a prediction from the trained "
    "machine learning model."
)

st.info(
    "Educational demo only. This prediction is not a real lending "
    "decision and should not be used to determine actual credit eligibility."
)

with st.form("loan_form"):
    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox(
            "Number of Dependents", ["0", "1", "2", "3+"]
        )
        education = st.selectbox(
            "Education", ["Graduate", "Not Graduate"]
        )
        self_employed = st.selectbox(
            "Self Employed", ["No", "Yes"]
        )
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"]
        )

    with col2:
        applicant_income = st.number_input(
            "Applicant Monthly Income", min_value=0, value=5000, step=500
        )
        coapplicant_income = st.number_input(
            "Co-applicant Monthly Income", min_value=0, value=0, step=500
        )
        loan_amount = st.number_input(
            "Loan Amount (in thousands)", min_value=1, value=120, step=10
        )
        loan_term = st.selectbox(
            "Loan Term (months)", [360, 180, 120, 240, 60, 300, 12]
        )
        credit_history = st.selectbox(
            "Credit History", ["Meets guidelines", "Does not meet guidelines"]
        )

    submitted = st.form_submit_button(
        "Predict Loan Status",
        type="primary",
        use_container_width=True,
    )

if submitted:
    credit_value = 1.0 if credit_history == "Meets guidelines" else 0.0

    applicant = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": credit_value,
        "Property_Area": property_area,
    }])

    try:
        prediction = int(model.predict(applicant)[0])
        probabilities = model.predict_proba(applicant)[0]
        approved_probability = float(probabilities[1])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Model prediction: Loan Approved")
        else:
            st.error("Model prediction: Loan Not Approved")

        st.metric(
            "Model-estimated approval probability",
            f"{approved_probability:.1%}",
        )
        st.progress(approved_probability)

        st.caption(
            "This probability reflects the trained model's estimate, "
            "not a guarantee of approval or a calibrated lending probability."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
