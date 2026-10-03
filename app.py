import streamlit as st
import joblib
import numpy as np

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background-color:#E6F7FF;
}

/* Title */
h1{
    color:#003366;
    text-align:center;
    font-weight:bold;
}

/* Labels */
label{
    color:#003366 !important;
    font-weight:600;
    font-size:16px;
}

/* Select Box */
div[data-baseweb="select"] > div{
    background:#CFEFFF !important;
    border:2px solid #5DADE2 !important;
    border-radius:10px !important;
}

/* Number Input */
div[data-baseweb="input"]{
    background:#CFEFFF !important;
    border:2px solid #5DADE2 !important;
    border-radius:10px !important;
}

input{
    background:#CFEFFF !important;
    color:#003366 !important;
    font-weight:bold;
}

/* Button */
.stButton>button{
    width:100%;
    height:50px;
    background:#0096D6;
    color:white;
    font-size:18px;
    font-weight:bold;
    border:none;
    border-radius:10px;
}

.stButton>button:hover{
    background:#0077B6;
    color:white;
}

/* Metric */
[data-testid="stMetricValue"]{
    color:#003366;
    font-weight:bold;
}

/* Success */
div[data-testid="stSuccess"]{
    border-radius:10px;
}

/* Error */
div[data-testid="stError"]{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = joblib.load("Titanic_Survival_Prediction_Model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🚢 Titanic Survival Prediction")

st.write("""
#### Predict whether a passenger survived the Titanic disaster.
Enter the passenger details below and click **Predict Survival**.
""")

st.divider()

# --------------------------------------------------
# Inputs
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1,2,3]
    )

    sex = st.selectbox(
        "Sex",
        ["Male","Female"]
    )

    sex = 1 if sex=="Male" else 0

    age = st.number_input(
        "Age",
        0,
        100,
        25
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=32.0
    )

with col2:

    sibsp = st.number_input(
        "Siblings / Spouse",
        min_value=0,
        value=0
    )

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        value=0
    )

    deck = st.number_input(
        "Deck (Encoded)",
        min_value=0,
        value=0
    )

    embarked = st.selectbox(
        "Embarked",
        ["C","Q","S"]
    )

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
family_size = sibsp + parch + 1

is_alone = 1 if family_size == 1 else 0

fare_per_person = fare / family_size

embarked_Q = 1 if embarked=="Q" else 0
embarked_S = 1 if embarked=="S" else 0

st.divider()

# --------------------------------------------------
# Metrics
# --------------------------------------------------
m1,m2,m3 = st.columns(3)

m1.metric("Family Size",family_size)

m2.metric(
    "Is Alone",
    "Yes" if is_alone else "No"
)

m3.metric(
    "Fare / Person",
    round(fare_per_person,2)
)

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Survival"):

    data = np.array([[

        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        deck,
        family_size,
        is_alone,
        fare_per_person,
        embarked_Q,
        embarked_S

    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0]==1:

        st.success(
            "🎉 Passenger Survived"
        )

    else:

        st.error(
            "❌ Passenger Did Not Survive"
        )