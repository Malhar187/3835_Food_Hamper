import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


st.set_page_config(page_title="Food Drive Prediction", layout="wide")

# Load Data
data = pd.read_csv("data_for_modeling.csv")
daily_data = pd.read_csv("daily_data.csv")

# Load SARIMA Model
model = joblib.load("new_sarima_model_SARIMAX.pkl")

# Exogenous Variables
exog_cols = ['scheduled_pickups', 'family_size', 'special_occasion_Dhu al-Qadah',
             'special_occasion_Eid al-Adha', 'special_occasion_Eid al-Fitr', 'special_occasion_Muharram',
             'special_occasion_Rabi al-Awwal', 'special_occasion_Rajab', 'special_occasion_Ramadan',
             'special_occasion_Shaban', 'season_Summer', 'season_Winter', 'season_Spring']

# 🎯 **Dashboard Page**
def dashboard():
    st.title("📊 Food Hamper Pickup Prediction Dashboard")
    st.write("🔍 Track food hamper demand & plan accordingly.")

    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Total Days", len(daily_data))
    col2.metric("📦 Total Pickups", int(daily_data['actual_pickups'].sum()))
    col3.metric("📈 Avg Pickups Per Day", round(daily_data['actual_pickups'].mean(), 2))

    # 📉 **Time Series Chart**
    fig = px.line(daily_data, x="pickup_date", y="actual_pickups",
                  title="📊 Daily Hamper Pickups Over Time",
                  labels={"pickup_date": "Date", "actual_pickups": "Pickups"},
                  template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# 📊 **EDA Page**
def exploratory_data_analysis():
    st.title("📊 Exploratory Data Analysis")
    
    st.subheader("🔍 Sample Data")
    st.dataframe(data.head())

    # 📉 Histogram of Pickups
    st.subheader("📦 Distribution of Actual Pickups")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(daily_data['actual_pickups'], bins=30, kde=True, ax=ax, color="royalblue")
    ax.set_title("Distribution of Hamper Pickups")
    st.pyplot(fig)

    # 📈 **Time Series Plot**
    st.subheader("📆 Hamper Pickups Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_data['pickup_date'], daily_data['actual_pickups'], color="green", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Pickups")
    ax.set_title("📊 Daily Hamper Pickups Trend")
    st.pyplot(fig)

# 🤖 **Machine Learning Modeling Page**
def machine_learning_modeling():
    st.title("🤖 Food Hamper Pickup Prediction")

    input_features = get_user_input()
    st.subheader("📝 User Input Parameters")
    st.dataframe(input_features)

    if "input_features" not in st.session_state:
        st.session_state.input_features = input_features

    if st.button("🚀 Predict"):
        prediction = predict_hamper_pickups(input_features)
        st.success(f"Predicted Pickups: {prediction.iloc[0]:.2f}")

    if st.button("📊 Generate XAI Report"):
        xai_report()

# 🔍 **XAI (Explainable AI) Report**
def xai_report():
    st.title("🔍 XAI (Explainable AI) Report")
    st.write("🧐 Understanding how the model makes predictions.")

    if "input_features" not in st.session_state:
        st.error("⚠️ No input found! Go to ML Modeling and enter values first.")
        return

    input_features = st.session_state.input_features
    input_features = input_features.reindex(columns=exog_cols, fill_value=0).astype(float)
    prediction = predict_hamper_pickups(input_features)

    st.subheader("📈 Prediction Result")
    st.success(f"Predicted Pickup: {prediction.iloc[0]:.2f}")

    model_coefficients = model.params
    coefficients_df = pd.DataFrame(model_coefficients, columns=["Coefficient"])
    
    # Find Matching Features
    matching_features = [col for col in coefficients_df.index if any(feature in col for feature in exog_cols)]
    if not matching_features:
        st.error("⚠️ No matching features found. Check your model.")
        return

    exog_coeffs = coefficients_df.loc[matching_features]

    # 🌈 **Feature Importance Bar Chart**
    st.subheader("📊 Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y=exog_coeffs.index, x=exog_coeffs["Coefficient"], palette="coolwarm", ax=ax)
    ax.set_title("Feature Importance Based on SARIMAX Coefficients")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Features")
    st.pyplot(fig)

    st.info("📌 Features with **higher absolute values** are more impactful in predictions.")

# 📥 **User Input Form**
def get_user_input():
    st.sidebar.title("📝 Input Features")
    scheduled_pickups = st.sidebar.number_input("📦 Scheduled Pickups", min_value=0)
    family_size = st.sidebar.number_input("👨‍👩‍👧 Family Size", min_value=0)

    # 🎉 Special Occasions
    special_occasions = {f"special_occasion_{name}": st.sidebar.checkbox(f"🎊 {name}")
                         for name in ["Dhu al-Qadah", "Eid al-Adha", "Eid al-Fitr", "Muharram", 
                                      "Rabi al-Awwal", "Rajab", "Ramadan", "Shaban"]}

    season = st.sidebar.radio("🌦 Season", ["Summer", "Winter", "Spring"])

    user_data = {
        "scheduled_pickups": [scheduled_pickups],
        "family_size": [family_size],
        **{key: [int(value)] for key, value in special_occasions.items()},
        "season_Summer": [int(season == "Summer")],
        "season_Winter": [int(season == "Winter")],
        "season_Spring": [int(season == "Spring")]
    }

    return pd.DataFrame(user_data)

# 🚀 **Prediction Function**
def predict_hamper_pickups(input_features):
    input_features = input_features.reindex(columns=exog_cols, fill_value=0).astype(float)
    start = len(data)
    end = start
    return model.predict(start=start, end=end, exog=input_features)

# 🤖 **RAG Chatbot Page**
def rag_chatbot_page():
    st.title("🤖 RAG Chatbot")
    st.write("Ask me anything about food hamper pickups!")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text2text-generation", model="google/flan-t5-large")

    # Documents
    data = {
        "Client_ID": [101, 102, 103],
        "Name": ["Alice", "Bob", "Charlie"],
        "Scheduled_Pickups": [5, 3, 4],  # Scheduled pickups count (or number of pickups requested)
        "Family_Size": [4, 3, 5],  # Family size of the clients
        "Special_Occasion_Dhu_al-Qadah": [0, 1, 0],  # Whether Dhu al-Qadah was a special occasion (1 or 0)
        "Special_Occasion_Eid_al-Adha": [1, 0, 0],  # Whether Eid al-Adha was a special occasion
        "Special_Occasion_Eid_al-Fitr": [0, 1, 0],  # Whether Eid al-Fitr was a special occasion
        "Special_Occasion_Muharram": [0, 0, 1],  # Whether Muharram was a special occasion
        "Special_Occasion_Rabi_al-Awwal": [0, 0, 0],  # Whether Rabi al-Awwal was a special occasion
        "Special_Occasion_Rajab": [0, 0, 0],  # Whether Rajab was a special occasion
        "Special_Occasion_Ramadan": [1, 0, 0],  # Whether Ramadan was a special occasion
        "Special_Occasion_Shaban": [0, 1, 0],  # Whether Shaban was a special occasion
        "Season_Summer": [1, 0, 0],  # Whether it's Summer season (1 or 0)
        "Season_Winter": [0, 1, 0],  # Whether it's Winter season (1 or 0)
        "Season_Spring": [0, 0, 1]  # Whether it's Spring season (1 or 0)
    }

    documents = pd.DataFrame(data)

    # Chatbot
    user_input = st.text_input("Ask a question:")
    if user_input:
        query_embedding = embedder.encode([user_input], convert_to_tensor=True)
        corpus_embeddings = embedder.encode(documents["Name"].tolist(), convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        best_match_idx = np.argmax(similarity_scores)
        best_match = documents.iloc[best_match_idx]

        # Use the information to make a SARIMAX prediction
        input_features = pd.DataFrame({
            "scheduled_pickups": [best_match["Scheduled_Pickups"]],
            "family_size": [best_match["Family_Size"]],
            **{f"special_occasion_{col}": [best_match[col]] for col in special_occasions.keys()},
            "season_Summer": [best_match["Season_Summer"]],
            "season_Winter": [best_match["Season_Winter"]],
            "season_Spring": [best_match["Season_Spring"]]
        })

        prediction = predict_hamper_pickups(input_features)
        
        # Generate chatbot response
        response = generator(f"What are the expected hamper pickups for {user_input}?\nAnswer based on: {prediction.iloc[0]:.2f}")
        st.write(response[0]['generated_text'])

# 🤖 **Main Page Navigation**
def main():
    menu = ["Dashboard", "EDA", "ML Modeling", "XAI", "RAG Chatbot"]
    choice = st.sidebar.selectbox("Select Option", menu)

    if choice == "Dashboard":
        dashboard()
    elif choice == "EDA":
        exploratory_data_analysis()
    elif choice == "ML Modeling":
        machine_learning_modeling()
    elif choice == "XAI":
        xai_report()
    elif choice == "RAG Chatbot":
        rag_chatbot_page()

if __name__ == '__main__':
    main()
