import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders safely
try:
    model = joblib.load("loan_model")
    label_encoders = joblib.load("label_encoders")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and Sidebar
st.title("🏦 Loan Eligibility Prediction System")
st.sidebar.header("User Input Features")

# Feature order
feature_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

# User Input Function
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Married", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=500)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=2000, step=500)
    loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, value=100, step=10)
    loan_amount_term = st.sidebar.selectbox("Loan Amount Term (months)", [360, 180, 120, 60])
    credit_history = st.sidebar.selectbox("Credit History", [1, 0])
    property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    # Encoding categorical inputs
    data = {}
    for col, value in zip(['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'],
                           [gender, married, education, self_employed, property_area]):
        data[col] = label_encoders[col].transform([value])[0] if col in label_encoders else 0
    
    # Convert '3+' to 3 for Dependents
    data['Dependents'] = 3 if dependents == "3+" else int(dependents)
    
    # Numeric values
    data.update({
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
    })
    
    return pd.DataFrame([data], columns=feature_order)

input_df = user_input_features()

# Prediction Button
if st.sidebar.button("🔍 Predict Eligibility"):
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] * 100  # Approval probability
        
        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.success(f"✅ Loan Approved with {proba:.2f}% probability!")
        else:
            st.error(f"❌ Loan Not Approved (Approval Chance: {proba:.2f}%)")
        
        # Visualization
        st.subheader("📈 Loan Distribution Insights")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=['Applicant Income', 'Coapplicant Income', 'Loan Amount'],
                    y=[input_df['ApplicantIncome'][0], input_df['CoapplicantIncome'][0], input_df['LoanAmount'][0]], ax=ax)
        ax.set_ylabel("Amount (in thousands)")
        st.pyplot(fig)
        
        # Display user input summary
        st.subheader("📜 User Input Summary")
        st.write(input_df)
        
        # Additional insights
        st.subheader("📊 Additional Insights")
        if input_df['Credit_History'][0] == 1:
            st.write("✔ Good credit history increases approval chances!")
        else:
            st.write("❌ Poor credit history might lower approval chances.")
        
        if input_df['ApplicantIncome'][0] > 8000:
            st.write("💰 High applicant income is a positive factor!")
        
        if input_df['LoanAmount'][0] > 300:
            st.write("⚠️ Large loan amounts may require additional scrutiny.")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
