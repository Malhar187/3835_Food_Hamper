import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data_for_modeling.csv")
daily_data = pd.read_csv("daily_data.csv")

# Load the trained SARIMA model
model = joblib.load("new_sarima_model_SARIMAX.pkl")

# Define exog_cols globally
exog_cols = ['scheduled_pickups', 'family_size',
             'special_occasion_Dhu al-Qadah', 'special_occasion_Eid al-Adha', 'special_occasion_Eid al-Fitr',
             'special_occasion_Muharram', 'special_occasion_Rabi al-Awwal', 'special_occasion_Rajab',
             'special_occasion_Ramadan', 'special_occasion_Shaban', 'season_Summer', 'season_Winter', 'season_Spring']

# Page 1: Dashboard
def dashboard():
    st.title("Food Hamper Pickup Prediction Dashboard")
    st.write("This dashboard helps predict the number of food hamper pickups based on historical data.")

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    st.write("Exploring the dataset used for modeling.")
    st.write(data.head())

    # Example Visualization
    fig = px.histogram(data, x="actual_pickups", title="Distribution of Actual Pickups")
    st.plotly_chart(fig)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Food Hamper Pickup Prediction")

    # Get user input
    input_features = get_user_input()
    st.subheader("User Input Parameters")
    st.write(input_features)

    # Initialize session_state.input_features if it does not exist
    if 'input_features' not in st.session_state:
        st.session_state.input_features = input_features

    if st.button("Predict"):
        prediction = predict_hamper_pickups(input_features)
        st.subheader("Prediction")
        st.write(prediction)

    # Add button to generate the XAI report
    if st.button("Generate XAI Report"):
        xai_report()  # Now no need to pass input_features to this function

# Page 4: XAI Report
def xai_report():
    st.title("XAI (Explainable AI) Report")
    st.write("This report provides an explanation for the prediction made by the model.")

    # Get user input for prediction
    input_features = st.session_state.input_features
    input_features_shap = input_features.reindex(columns=exog_cols, fill_value=0).astype(float)

    # Initialize the SHAP explainer with the model
    explainer = shap.Explainer(model, input_features_shap)
    shap_values = explainer(input_features_shap)

    # SHAP Summary Plot
    st.subheader("SHAP Summary Plot (Feature Importance)")
    shap.summary_plot(shap_values, input_features_shap)
    st.pyplot(bbox_inches='tight')

    # Display SHAP values (Feature Contributions to Prediction)
    st.subheader("SHAP Force Plot for the Prediction")
    shap.initjs()  # Initialize JavaScript for force plot visualization
    st.pyplot(shap.force_plot(shap_values.base_values[0], shap_values.values[0], input_features_shap.iloc[0]))

    # Additional details
    st.write("The SHAP summary plot above displays the importance of each feature in making the prediction.")
    st.write("The SHAP force plot illustrates how each feature has contributed to the model's prediction for the selected input parameters.")


# Function to get user input
def get_user_input():
    st.sidebar.header("User Input Parameters")

    pickup_date = st.sidebar.date_input("Pickup Date")
    actual_pickup_boxcox = st.sidebar.number_input("Actual Pickup Boxcox")
    scheduled_pickups = st.sidebar.number_input("Scheduled Pickups", min_value=0)
    family_size = st.sidebar.number_input("Family Size", min_value=0)

    special_occasions = {
        "Dhu al-Qadah": st.sidebar.checkbox("Special Occasion: Dhu al-Qadah"),
        "Eid al-Adha": st.sidebar.checkbox("Special Occasion: Eid al-Adha"),
        "Eid al-Fitr": st.sidebar.checkbox("Special Occasion: Eid al-Fitr"),
        "Muharram": st.sidebar.checkbox("Special Occasion: Muharram"),
        "Rabi al-Awwal": st.sidebar.checkbox("Special Occasion: Rabi al-Awwal"),
        "Rajab": st.sidebar.checkbox("Special Occasion: Rajab"),
        "Ramadan": st.sidebar.checkbox("Special Occasion: Ramadan"),
        "Shaban": st.sidebar.checkbox("Special Occasion: Shaban")
    }

    season = st.sidebar.radio("Season", ("Summer", "Winter", "Spring"))

    user_data = {
        "actual_pickup_boxcox": [actual_pickup_boxcox],
        "scheduled_pickups": [scheduled_pickups],
        "family_size": [family_size],
        **{key: [value] for key, value in special_occasions.items()},
        "season_Summer": [1 if season == "Summer" else 0],
        "season_Winter": [1 if season == "Winter" else 0],
        "season_Spring": [1 if season == "Spring" else 0]
    }

    return pd.DataFrame(user_data)

# Function to make predictions
def predict_hamper_pickups(input_features):
    # Ensure input features match expected columns
    input_features = input_features.reindex(columns=exog_cols, fill_value=0)
    input_features = input_features.astype(float)

    start = len(data)
    end = start
    prediction = model.predict(start=start, end=end, exog=input_features)

    return prediction

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "XAI Report"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "XAI Report":
        xai_report() 

if __name__ == "__main__":
    main()
