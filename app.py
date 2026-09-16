import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="California Housing Price Predictor", page_icon="🏡")

MODEL_PATH = "my_california_housing_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🏡 California Housing Price Predictor")
st.write(
    "Estimate the median house value for a California district based on 1990 "
    "census-style features. Built as the deployment step of an end-to-end ML "
    "pipeline (see the accompanying notebook for the full workflow)."
)

model = load_model()

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23, format="%.4f")
    latitude = st.number_input("Latitude", value=37.88, format="%.4f")
    housing_median_age = st.number_input("Housing median age", value=41, min_value=0, max_value=100)
    total_rooms = st.number_input("Total rooms", value=880, min_value=1)
    total_bedrooms = st.number_input("Total bedrooms", value=129, min_value=0)

with col2:
    population = st.number_input("Population", value=322, min_value=1)
    households = st.number_input("Households", value=126, min_value=1)
    median_income = st.number_input("Median income (tens of thousands $)", value=8.3252, format="%.4f")
    ocean_proximity = st.selectbox(
        "Ocean proximity",
        ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
    )

if st.button("Predict median house value"):
    input_df = pd.DataFrame(
        [{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        }]
    )
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted median house value: **${prediction:,.0f}**")
