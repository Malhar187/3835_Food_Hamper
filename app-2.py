import streamlit as st
import pandas as pd
import joblib

# Load the trained fitted SARIMAX model (fitted results, not the base model)
model = joblib.load("sarima_model_SARIMAX.pkl")  # The model should be a fitted SARIMAX model
data = pd.read_csv("data_for_modeling.csv")

st.write('''# Food Hamper Pickup Prediction App''')

st.sidebar.header('User Input Parameters')

def user_input_features():
    # Input features (replace with your actual feature names and ranges)
    pickup_date = st.sidebar.date_input("Pickup Date")
    actual_pickups = st.sidebar.number_input("Actual Pickups", min_value=0)
    actual_pickup_boxcox = st.sidebar.number_input("Actual Pickup Boxcox") 
    scheduled_pickups = st.sidebar.number_input("Scheduled Pickups", min_value=0)
    actual_pickup_lag7 = st.sidebar.number_input("Actual Pickup Lag 7")
    actual_pickup_lag14 = st.sidebar.number_input("Actual Pickup Lag 14")
    scheduled_pickup_lag7 = st.sidebar.number_input("Scheduled Pickup Lag 7")
    scheduled_pickup_lag14 = st.sidebar.number_input("Scheduled Pickup Lag 14")
    family_size = st.sidebar.number_input("Family Size", min_value=0)
    # Special Occasion Features (using checkboxes)
    special_occasion_Dhu_al_Qadah = st.sidebar.checkbox("Special Occasion: Dhu al-Qadah")
    special_occasion_Eid_al_Adha = st.sidebar.checkbox("Special Occasion: Eid al-Adha")
    special_occasion_Eid_al_Fitr = st.sidebar.checkbox("Special Occasion: Eid al-Fitr")
    special_occasion_Muharram = st.sidebar.checkbox("Special Occasion: Muharram")
    special_occasion_Rabi_al_Awwal = st.sidebar.checkbox("Special Occasion: Rabi al-Awwal")
    special_occasion_Rajab = st.sidebar.checkbox("Special Occasion: Rajab")
    special_occasion_Ramadan = st.sidebar.checkbox("Special Occasion: Ramadan")
    special_occasion_Shaban = st.sidebar.checkbox("Special Occasion: Shaban")
    
    # Season Features (using radio buttons)
    season = st.sidebar.radio("Season", ("Summer", "Winter", "Spring"))  

    data = {
        'pickup_date': [pickup_date],
        'actual_pickups': [actual_pickups],
        'actual_pickup_boxcox': [actual_pickup_boxcox],
        'scheduled_pickups': [scheduled_pickups],
        'actual_pickup_lag7': [actual_pickup_lag7],
        'actual_pickup_lag14': [actual_pickup_lag14],
        'scheduled_pickup_lag7': [scheduled_pickup_lag7],
        'scheduled_pickup_lag14': [scheduled_pickup_lag14],
        'family_size': [family_size],
        'special_occasion_Dhu_al_Qadah': [special_occasion_Dhu_al_Qadah],
        'special_occasion_Eid_al_Adha': [special_occasion_Eid_al_Adha],
        'special_occasion_Eid_al_Fitr': [special_occasion_Eid_al_Fitr],
        'special_occasion_Muharram': [special_occasion_Muharram],
        'special_occasion_Rabi_al_Awwal': [special_occasion_Rabi_al_Awwal],
        'special_occasion_Rajab': [special_occasion_Rajab],
        'special_occasion_Ramadan': [special_occasion_Ramadan],
        'special_occasion_Shaban': [special_occasion_Shaban],
        'season_Summer': [1 if season == "Summer" else 0],  # One-hot encoding for season
        'season_Winter': [1 if season == "Winter" else 0],
        'season_Spring': [1 if season == "Spring" else 0]
    }  

    # Create a DataFrame from the dictionary
    features = pd.DataFrame(data)

    # Ensure all columns exist (for debugging purposes)
    expected_columns = ['special_occasion_Dhu_al_Qadah', 'special_occasion_Eid_al_Adha', 
                        'special_occasion_Eid_al_Fitr', 'special_occasion_Muharram', 
                        'special_occasion_Rabi_al_Awwal', 'special_occasion_Rajab', 
                        'special_occasion_Ramadan', 'special_occasion_Shaban', 
                        'season_Summer', 'season_Winter', 'season_Spring']
    
    for col in expected_columns:
        if col not in features.columns:
            features[col] = 0  # Set default value to 0 if the column is missing

    return features

# Get the user input features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# Ensure exog features are used in prediction
exog_cols = ['scheduled_pickups', 'actual_pickup_lag7', 'actual_pickup_lag14',
             'scheduled_pickup_lag7', 'scheduled_pickup_lag14', 'family_size',
             'special_occasion_Dhu_al_Qadah', 'special_occasion_Eid_al_Adha', 'special_occasion_Eid_al_Fitr',
             'special_occasion_Muharram', 'special_occasion_Rabi_al_Awwal', 'special_occasion_Rajab',
             'special_occasion_Ramadan', 'special_occasion_Shaban', 'season_Summer', 'season_Winter', 'season_Spring']

# Ensure exog is in the correct format
df_exog = df[exog_cols].astype(float)

# Forecasting using the fitted SARIMAX model
start = len(data)  # Forecast starts from the next time step
end = start  # Single-step forecast

# Ensure model is already fitted (using `model.get_fittedparams()`)
# Use the fitted SARIMAX model for prediction, without needing additional params
prediction = model.predict(start=start, end=end, exog=df_exog)

st.subheader('Prediction')
st.write(prediction)
