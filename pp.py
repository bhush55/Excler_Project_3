import streamlit as st
import joblib
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()

# Set background image for the page
st.markdown(
    """
    <style>
        body {
            background-image: url('https://example.com/courthouse.jpg');  # Replace with your image URL or path
            background-size: cover;
            background-position: center;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stMarkdown {
            font-family: 'Arial', sans-serif;
        }
        .prediction-result {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Title of the app with styling
st.markdown("<h1 style='text-align: center; color: #FFD700;'>Attorney Involvement Prediction</h1>", unsafe_allow_html=True)

# Sidebar Header and Description
st.sidebar.markdown("<h3 style='text-align: center;'>Input Claim Details</h3>", unsafe_allow_html=True)
st.sidebar.markdown("Please enter the following claim details to get a prediction on attorney involvement.")

# Input fields with tooltips for better understanding
casenum = st.sidebar.text_input("**Case Number**", value="0", help="Enter a unique identifier for the claim (e.g., ABC123).")
clmsex = st.sidebar.selectbox("**Claimant Sex**", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Select the gender of the claimant.")
clmins = st.sidebar.selectbox("**Claimant Insured**", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Is the claimant insured?")
seatbelt = st.sidebar.selectbox("**Seatbelt Used?**", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Did the claimant use a seatbelt?")
clmage = st.sidebar.number_input("**Claimant Age**", min_value=0, max_value=120, value=30, help="Enter the age of the claimant.")
loss = st.sidebar.number_input("**Loss Amount ($)**", min_value=0.0, value=5000.0, help="Total loss amount in dollars.")
acc_severity = st.sidebar.selectbox("**Accident Severity**", [0, 1, 2], format_func=lambda x: "Low" if x == 0 else "Medium" if x == 1 else "High", help="Severity of the accident.")
claim_amount = st.sidebar.number_input("**Claim Amount Requested ($)**", min_value=0.0, value=10000.0, help="Amount requested for the claim.")
claim_status = st.sidebar.selectbox("**Claim Approval Status**", [0, 1], format_func=lambda x: "Rejected" if x == 0 else "Approved", help="Status of the claim approval.")
settlement = st.sidebar.number_input("**Settlement Amount ($)**", min_value=0.0, value=8000.0, help="Amount of the settlement.")
policy_type = st.sidebar.selectbox("**Policy Type**", [0, 1, 2], format_func=lambda x: "Basic" if x == 0 else "Standard" if x == 1 else "Premium", help="Type of insurance policy.")
driving_record = st.sidebar.selectbox("**Driving Record**", [0, 1, 2], format_func=lambda x: "Clean" if x == 0 else "Minor Violation" if x == 1 else "Major Violation", help="Driving record of the claimant.")

# Create a dataframe with matching feature names
input_data = pd.DataFrame([[casenum, clmsex, clmins, seatbelt, clmage, loss, acc_severity, 
                            claim_amount, claim_status, settlement, policy_type, driving_record]],
                          columns=['CASENUM', 'CLMSEX', 'CLMINSUR', 'SEATBELT', 'CLMAGE', 'LOSS', 
                                   'Accident_Severity', 'Claim_Amount_Requested', 
                                   'Claim_Approval_Status', 'Settlement_Amount', 
                                   'Policy_Type', 'Driving_Record'])

# Ensure column order matches training data
expected_features = model.feature_names_in_  # Extract correct feature names from model
input_data = input_data.reindex(columns=expected_features, fill_value=0)  # Match feature order

# Prediction Button with custom styling
if st.sidebar.button("Predict", use_container_width=True):
    prediction = model.predict(input_data)[0]
    
    # Prediction Display with style
    if prediction == 1:
        st.markdown("<div class='prediction-result' style='color: #FF6347;'>Prediction: Attorney is likely to be involved.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-result' style='color: #32CD32;'>Prediction: Attorney is unlikely to be involved.</div>", unsafe_allow_html=True)

# Footer with app description and info
st.markdown("---")
st.markdown("""
    <p style='text-align: center;'>This app predicts attorney involvement in insurance claims using a trained Random Forest model. 
    The model is based on historical claim data and helps insurance companies optimize their processes.</p>
    <p style='text-align: center;'>Developed with Streamlit and Python.</p>
""", unsafe_allow_html=True)
