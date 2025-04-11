import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import logging
import traceback

# Configure logging
logging.basicConfig(
    filename="credit_loan_app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

logging.info("Credit Loan Eligibility Predictor app started.")

# Set the page title and description
st.markdown("<h1 style='text-align: center;'>Credit Loan Eligibility Predictor</h1>", unsafe_allow_html=True)
st.write("""
This app predicts whether a loan applicant is eligible for a loan 
based on various personal and financial characteristics.
""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()

# Load the pre-trained model
try:
    with open("models/RFmodel.pkl", "rb") as rf_pickle:
        rf_model = pickle.load(rf_pickle)
    logging.info("Random Forest model loaded successfully.")
except Exception as e:
    logging.error("Failed to load Random Forest model.")
    logging.error(traceback.format_exc())
    st.error("Model could not be loaded. Please check and try again.")
    st.stop()

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.markdown("<h3 style='text-align: center;'>Loan Applicant Details</h3>", unsafe_allow_html=True)
    
    # Create 3 columns
    cols = st.columns(3)
    
    # Row 1
    # Column 1
    with cols[0]:
        Gender = st.selectbox("Gender", options=["Male", "Female"])
    # Column 2
    with cols[1]:
        Married = st.selectbox("Marital Status", options=["Yes", "No"])
    # Column 3
    with cols[2]:
        Dependents = st.selectbox("Number of Dependents", options=["0", "1", "2", "3+"])

    # Row 2
    cols = st.columns(3)
    # Column 1
    with cols[0]:
        Education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
    # Column 2
    with cols[1]:
        Self_Employed = st.selectbox("Self Employed", options=["Yes", "No"])
    # Column 3
    with cols[2]:
        ApplicantIncome = st.number_input("Applicant Monthly Income", min_value=0, step=1000)

    # Row 3
    cols = st.columns(3)
    # Column 1
    with cols[0]:
        CoapplicantIncome = st.number_input("Coapplicant Monthly Income", min_value=0, step=1000)
    # Column 2
    with cols[1]:
        LoanAmount = st.number_input("Loan Amount", min_value=0, step=1000)
    # Column 3
    with cols[2]:
        Loan_Amount_Term = st.selectbox("Loan Amount Term (Months)", options=["360", "180", "240", "120", "60"])

    # Row 4
    cols = st.columns(3)
    # Column 1
    with cols[0]:
        Credit_History = st.selectbox("Credit History", options=["1", "0"])
    # Column 2
    with cols[1]:
        Property_Area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])
    # Column 3
    with cols[2]:
        submitted = st.form_submit_button("Predict Loan Eligibility")

# Handle the dummy variables to pass to the model
if submitted:
    try:
        # Handle dependents
        Gender_Male = 0 if Gender == "Male" else 1
        Gender_Female = 1 if Gender == "Female" else 0

        Married_Yes = 1 if Married == "Yes" else 0
        Married_No = 1 if Married == "No" else 0

        Dependents_0 = 1 if Dependents == "0" else 0
        Dependents_1 = 1 if Dependents == "1" else 0
        Dependents_2 = 1 if Dependents == "2" else 0
        Dependents_3 = 1 if Dependents == "3+" else 0

        Education_Graduate = 1 if Education == "Graduate" else 0
        Education_Not_Graduate = 1 if Education == "Not Graduate" else 0

        Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
        Self_Employed_No = 1 if Self_Employed == "No" else 0

        Property_Area_Rural = 1 if Property_Area == "Rural" else 0
        Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
        Property_Area_Urban = 1 if Property_Area == "Urban" else 0

        # Convert Loan Amount Term and Credit History to integers
        Loan_Amount_Term = int(Loan_Amount_Term)
        Credit_History = int(Credit_History)

        # Prepare the input for prediction. This has to go in the same order as it was trained
        prediction_input = [[
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Gender_Female, Gender_Male,
            Married_No, Married_Yes, Dependents_0, Dependents_1,
            Dependents_2, Dependents_3, Education_Graduate,
            Education_Not_Graduate, Self_Employed_No, Self_Employed_Yes,
            Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban
        ]]

        # Make prediction
        new_prediction = rf_model.predict(prediction_input)
        logging.info(f"Prediction completed: {new_prediction[0]}")

        # Display result
        st.subheader("Prediction Result:")
        if new_prediction[0] == "Y":
            st.success("You are eligible for the loan!")
        else:
            st.error("Sorry, you are not eligible for the loan.")

        try:
            st.image("feature_importance.png")
            logging.info("Feature importance image displayed.")
        except Exception as e:
            logging.warning("Failed to load feature importance image.")
            logging.warning(traceback.format_exc())
            st.warning("Could not display feature importance image.")

    except Exception as e:
        logging.error("Error during prediction process.")
        logging.error(traceback.format_exc())
        st.error("An error occurred while processing your input. Please try again.")

st.write("""
We used a machine learning (Random Forest) model to predict your eligibility, the features used in this prediction are ranked by relative
importance below.
""")
