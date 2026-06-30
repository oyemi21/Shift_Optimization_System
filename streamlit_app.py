import streamlit as st
import requests
import pandas as pd
import os
import datetime
import numpy as np

from src.data.data_ingestion import load_data

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="NorDex Shift Dashboard", layout="wide")
st.title("NorDex Manufacturing Shift Intelligence Dashboard")

#LOAD CATEGORICAL VALUES ONLY
@st.cache_data
def get_categorical_values():
    df = load_data()
    return{
        "shift_name": sorted(df["shift_name"].dropna().unique()),
        "skill_category": sorted(df["skill_category"].dropna().unique()),
        "machine_status": sorted(df["machine_status"].dropna().unique()),
    }


categories = get_categorical_values()

#INPUT SECTION
st.header("Shift Input Parameter")

col1, col2 = st.columns(2)

with col1:
    experience_level = st.slider("Experience Level", 1, 15, 6)
    runtime_hours = st.number_input("Runtime Hours", 0.0, 12.0, 6.0)
    downtime_minutes = st.number_input("Downtime Minutes", 0.0, 180.0, 30.0)
    maintenance_flag = st.selectbox("Maintenance Flag", [0, 1])
    maintenance_downtime = st.number_input("Maintenance Downtime", 0.0, 180.0, 0.0)

with col2:
    units_produced = st.number_input("Units Produced", 0, 2000, 800)
    defect_count = st.number_input("Defect Count", 0, 200, 20)
    temperature = st.number_input("Temperature", 15.0, 40.0, 22.0)
    humidity = st.number_input("Humidity", 10.0, 100.0, 50.0)

cycle_time_avg = (runtime_hours * 60 / units_produced) if units_produced > 0 else 35.65
shift_duration = runtime_hours
total_operation_hours = runtime_hours + (downtime_minutes / 60)
day_of_week = datetime.datetime.today().weekday()
hour_of_day = datetime.datetime.now().hour
day_of_week_sin  = np.sin(2 * np.pi * day_of_week / 7)
day_of_week_cos  = np.cos(2 * np.pi * day_of_week / 7)
hour_of_day_sin  = np.sin(2 * np.pi * hour_of_day / 24)
hour_of_day_cos  = np.cos(2 * np.pi * hour_of_day / 24)

st.subheader("Shift Context")

shift_name = st.selectbox("Shift Name", categories["shift_name"])
skill_category = st.selectbox("Skill Category", categories["skill_category"])
machine_status = st.selectbox("Machine Status", categories["machine_status"])


#PREDICTION
st.markdown("---")

if st.button("Predict Shift Efficiency"):

    payload = {
        "experience_level":  experience_level ,
        "downtime_minutes":  downtime_minutes ,
        "defect_count":  defect_count ,
        "units_produced":  units_produced ,
        "maintenance_flag": maintenance_flag ,
        "maintenance_downtime": maintenance_downtime,
        "cycle_time_avg": cycle_time_avg ,
        "temperature": temperature ,
        "humidity": humidity ,
        "shift_name": shift_name,
        "skill_category": skill_category,
        "machine_status": machine_status,
        "shift_duration": shift_duration,
        "runtime_hours": runtime_hours,
        "total_operation_hours": total_operation_hours,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "day_of_week_sin": day_of_week_sin,
        "day_of_week_cos": day_of_week_cos,
        "hour_of_day_sin": hour_of_day_sin,
        "hour_of_day_cos": hour_of_day_cos

    }

    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.ok:
        result = response.json()["predicted_shift_efficiency_score"]
        st.metric("Shift Efficiency Score", f"{result:.2f}")
    else:
        st.error(response.text)

#OPTIMIZATION (OPTUNA)
st.markdown("---")
st.header("Optimization Explorer")

exp_min, exp_max = st.slider("Experience Range", 1, 15, (5, 10))
dt_min, dt_max = st.slider("Downtime Range", 0, 120, (10, 60))
def_min, def_max = st.slider("Defect Range", 0, 100, (5, 30))

n_trials = st.slider("Optimization Trials", 10, 200, 50)

if st.button("Run Optimization"):

    payload = {
        "shift_name": shift_name,
        "skill_category": skill_category,
        "machine_status": machine_status,
        "exp_range": (exp_min, exp_max),
        "downtime_range": (dt_min, dt_max),
        "defect_range": (def_min, def_max),
        "n_trials": n_trials
    }

    response = requests.post(f"{API_URL}/optimize", json=payload)

    if response.ok:
        result = response.json()

        st.success("Optimization completed")

        st.subheader("Best Resuslt")
        st.json(result["best_parameters"])
        st.metric("Best Score", f"{result['best_score']:.4f}")

        st.subheader("Top Trials")
        st.dataframe(pd.DataFrame(result["top_trials"]))

    else:
        st.error(response.text)


#RETRAIN
st.markdown("---")

if st.button("Retrain Model"):
    response = requests.post(f"{API_URL}/retrain")

    if response.ok:
        st.info("Training started in the background")
    else:
        st.error(response.text)