import os
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Industrial ML Monitor", page_icon="🏭", layout="centered")
st.title("Industrial Equipment Failure-Risk Demo")
st.caption("Educational demo trained on synthetic data. Do not use for real safety decisions.")

api_url = st.sidebar.text_input("API base URL", DEFAULT_API_URL).strip().rstrip("/")
st.sidebar.write("API docs: " + api_url + "/docs")

with st.form("prediction_form"):
    equipment_id = st.text_input("Equipment ID", value="PUMP-001", max_chars=64)
    temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=200.0, value=82.0)
    vibration = st.number_input("Vibration (mm/s)", min_value=0.0, max_value=100.0, value=4.0)
    speed = st.number_input("Speed (RPM)", min_value=0.0, max_value=10000.0, value=1500.0)
    hours = st.number_input("Operating hours", min_value=0.0, max_value=250000.0, value=12000.0)
    submitted = st.form_submit_button("Estimate failure risk")

if submitted:
    if not equipment_id.strip():
        st.error("Equipment ID is required.")
        st.stop()

    payload = {
        "equipment_id": equipment_id.strip(),
        "temperature_c": temperature,
        "vibration_mm_s": vibration,
        "speed_rpm": speed,
        "operating_hours": hours,
    }
    try:
        response = requests.post(
            f"{api_url}/predict", json=payload, timeout=10
        )
        if response.status_code == 422:
            st.error("Input validation failed. Check units and allowed ranges.")
        elif response.status_code == 503:
            st.error("The model is unavailable. Train the model and restart the API.")
        else:
            response.raise_for_status()
            result = response.json()
            st.metric("Estimated failure probability", f"{result['failure_probability']:.1%}")
            st.write("Decision at configured threshold:", result["predicted_failure"])
            st.write("Demo risk band:", result["risk_band"].title())
            st.write("Model version:", result["model_version"])
            st.info(result["disclaimer"])
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Start the API in a separate terminal.")
    except requests.exceptions.Timeout:
        st.error("The API request timed out.")
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
