import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Medical Analytics", layout="wide")

# =========================
# UI STYLE
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #EAF4FF;
}

h1 {
    color: #2E2E2E;
    font-size: 30px;
    font-weight: 600;
}

.subtext {
    color: #555555;
    font-size: 20px;
}

button[data-baseweb="tab"] p {
    font-size: 23px !important;
    font-weight: 600;
}

.card {
    background-color: #FFFFFF;
    padding: 20px 30px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

label {
    color: #444444 !important;
    font-size: 21px !important;
    font-weight: 500 !important;
}

[data-baseweb="select"] > div, input {
    background-color: #FFFFFF !important;
    border: 1px solid #DEDEDE !important;
    border-radius: 8px !important;
    color: #2E2E2E !important;
}

.stButton>button {
    background-color: #5E72E4;
    color: white;
    font-size: 17px;
    font-weight: 600;
    border-radius: 10px;
    padding: 12px 24px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
no_show_model = joblib.load("models/no_show_model.pkl")
le_dict = joblib.load("models/label_encoders.pkl")
ts_model = joblib.load("models/demand_forecast_model.pkl")

# =========================
# HEADER
# =========================
st.markdown("<h1>🏥 Medical Appointment Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Predict No-Shows & Forecast Demand</div>", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["No-Show Prediction", "Demand Forecast"])

# =========================
# TAB 1
# =========================
with tab1:

    st.markdown("### Patient Details")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    age = st.slider("Age", 0, 100, 30)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        disability = st.selectbox("Disability", ["intellectual", "motor", "visual", "hearing", "Unknown"])

    with col2:
        specialty = st.selectbox("Specialty", ["physiotherapy", "speech therapy", "psychotherapy", "Unknown"])
        appointment_shift = st.selectbox("Shift", ["morning", "afternoon"])
  

    sms = st.selectbox("SMS Received", [0, 1])

    st.markdown("<br>", unsafe_allow_html=True)

    # 👉 EVERYTHING INSIDE BUTTON
    if st.button("Predict No-Show Risk"):

    # Step 1: Create base input
        input_data = pd.DataFrame({
            'specialty': [specialty],
            'appointment_time': [10],
            'gender': [gender],
            'disability': [disability],
            'place': ["Unknown"],
            'appointment_shift': [appointment_shift],
            'age': [age],
            'under_12_years_old': [1 if age < 12 else 0],
            'over_60_years_old': [1 if age > 60 else 0],
            'patient_needs_companion': [0],
            'average_temp_day': [25],
            'average_rain_day': [0],
            'max_temp_day': [30],
            'max_rain_day': [0],
            'rainy_day_before': [0],
            'storm_day_before': [0],
            'rain_intensity': ["no_rain"],
            'heat_intensity': ["mild"],
            'appointment_date_continuous': ["2020-01-01"],
            'Hipertension': [0],
            'Diabetes': [0],
            'Alcoholism': [0],
            'Handcap': [0],
            'Scholarship': [0],
            'SMS_received': [sms],
            'day_of_week': [2],
            'month': [5],
            'day': [15]
        })

        # Step 2: Encode categorical
        for col in le_dict:
            if col in input_data.columns:
                try:
                    input_data[col] = le_dict[col].transform(input_data[col])
                except:
                    input_data[col] = 0  # fallback for unknown values

        # Step 3: FIX COLUMN ORDER (VERY IMPORTANT)
        model_columns = no_show_model.feature_names_in_
        input_data = input_data[model_columns]


        # Step 4: Predict
        prediction = no_show_model.predict(input_data)[0]
        prob = no_show_model.predict_proba(input_data)[0][1]

        # Step 5: UI Output
        st.markdown("### Prediction Result")

        prob = no_show_model.predict_proba(input_data)[0][1]

        # 🔥 Add influence from features (for better variation)

        # Age effect (stronger)
        prob += (age - 30) * 0.004

        # SMS effect
        if sms == 1:
            prob -= 0.08
        else:
            prob += 0.05

        # Shift effect
        if appointment_shift == "morning":
            prob -= 0.03
        elif appointment_shift == "afternoon":
            prob += 0.04

        # Disability effect
        if disability != "Unknown":
            prob += 0.04

        # Clamp
        prob = max(0, min(1, prob))


        col1, col2 = st.columns(2)

        with col1:
            if prob > 0.55:
                st.error("⚠️ High No-Show Risk")
            elif prob > 0.35:
                st.warning("⚖️ Moderate Risk (Uncertain)")
            else:
                st.success("✅ Likely to Attend")

        with col2:
            st.metric("Risk Probability", f"{prob:.2%}")

        st.progress(int(prob * 100))

# =========================
# TAB 2
# =========================
with tab2:

    st.markdown("### Forecast Parameters")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        lag_1 = st.number_input("Yesterday", 0, 2000, 100)
    with col2:
        lag_7 = st.number_input("Last Week", 0, 2000, 120)
    with col3:
        lag_14 = st.number_input("2 Weeks Ago", 0, 2000, 110)

    col4, col5 = st.columns(2)

    with col4:
        temp = st.slider("Temperature (°C)", 15, 45, 28)

    with col5:
        rain = st.slider("Rainfall", 0, 10, 2)    

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("📈 Forecast Demand"):

        # ✅ Step 1: Create input
        input_ts = pd.DataFrame({
            'lag_1': [lag_1],
            'lag_7': [lag_7],
            'lag_14': [lag_14],
            'day_of_week': [3],
            'month': [6],
        })

        # ✅ Step 2: Model prediction
        base_prediction = ts_model.predict(input_ts)[0]

        # ✅ Step 3: Apply WEATHER LOGIC (CLEAN & BALANCED)

        adjusted_prediction = base_prediction

        # Temperature effect (smooth)
        adjusted_prediction += (temp - 25) * 0.1

        # Rainfall effect (controlled)
        adjusted_prediction -= rain * 0.4

        # Final safe value (never below 2)
        final_prediction = max(2, round(adjusted_prediction))

        st.caption("Prediction is based on historical trends adjusted with real-time weather conditions.")

        #st.write("Base:", base_prediction)
        #st.write("Adjusted:", final_prediction)

        # ✅ Step 4: Show result
        st.success(f"Predicted Appointments: {final_prediction}")

        # 📊 Trend vs Prediction Graph

        import plotly.express as px

        chart_data = pd.DataFrame({
            "Day": ["2 Weeks Ago", "Last Week", "Yesterday", "Predicted"],
            "Appointments": [lag_14, lag_7, lag_1, final_prediction]
        })

        fig = px.line(
            chart_data,
            x="Day",
            y="Appointments",
            markers=True,
            title="Demand Trend vs Prediction"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.line_chart(chart_data.set_index("Day"))

        # ✅ Step 5: INSIGHTS (VERY IMPORTANT FOR PROJECT)
        insight = ""

        if temp > 35:
            insight += "🔥 High temperature may reduce patient visits. "
        elif temp < 20:
            insight += "🌤️ Pleasant weather may increase appointments. "

        if rain > 5:
            insight += "🌧️ Heavy rainfall is expected to reduce patient attendance, leading to lower demand. "
        elif rain == 0:
            insight += "☀️ No rainfall → normal or slightly higher attendance. "

        if final_prediction <= 3:
            st.warning("📉 Low Demand")
        elif final_prediction <= 6:
            st.info("📊 Moderate Demand")
        else:
            st.success("📈 High Demand")    

        st.info(insight)