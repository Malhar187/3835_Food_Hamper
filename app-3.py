import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Initial Setup
st.set_page_config(page_title="Food Hamper Prediction", layout="wide")
st.markdown("""
    <style>
        .css-18e3th9 { padding-top: 2rem; }
        .main { background-color: #f9f9f9; }
        .stButton > button { width: 100%; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# Load data & model
data = pd.read_csv("data_for_modeling.csv")
daily_data = pd.read_csv("daily_data.csv")
model = joblib.load("new_sarima_model_SARIMAX.pkl")

exog_cols = ['scheduled_pickups', 'family_size', 'special_occasion_Dhu al-Qadah',
             'special_occasion_Eid al-Adha', 'special_occasion_Eid al-Fitr', 'special_occasion_Muharram',
             'special_occasion_Rabi al-Awwal', 'special_occasion_Rajab', 'special_occasion_Ramadan',
             'special_occasion_Shaban', 'season_Summer', 'season_Winter', 'season_Spring']

# 🎯 Dashboard
def dashboard():
    st.title("📊 Food Hamper Pickup Dashboard")
    st.markdown("Easily track demand patterns and optimize your resources.")

    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("📅 Total Days", len(daily_data))
        col2.metric("📦 Total Pickups", int(daily_data['actual_pickups'].sum()))
        col3.metric("📈 Avg Daily Pickups", round(daily_data['actual_pickups'].mean(), 2))

    fig = px.line(daily_data, x="pickup_date", y="actual_pickups",
                  title="📈 Daily Hamper Pickups Over Time",
                  labels={"pickup_date": "Date", "actual_pickups": "Pickups"},
                  template="plotly_white", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# 📊 EDA
def exploratory_data_analysis():
    st.title("🔍 Exploratory Data Analysis")

    st.markdown("### 📌 Sample Data")
    st.dataframe(data.head())

    st.markdown("### 📦 Pickups Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(daily_data['actual_pickups'], bins=30, kde=True, color="dodgerblue", ax=ax)
    ax.set_title("Distribution of Pickups")
    st.pyplot(fig)

    st.markdown("### 📆 Time Series Overview")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_data['pickup_date'], daily_data['actual_pickups'], color="green", linewidth=2)
    ax.set_title("Hamper Pickups Over Time")
    st.pyplot(fig)

# 🧠 Modeling Page
def machine_learning_modeling():
    st.title("🤖 Predict Future Hamper Pickups")

    input_features = get_user_input()
    st.markdown("### 📝 Your Inputs")
    st.dataframe(input_features)

    if "input_features" not in st.session_state:
        st.session_state.input_features = input_features

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 Predict Next 7 Days"):
            prediction_df = predict_hamper_pickups(input_features)
            st.success("Prediction Completed")
            st.dataframe(prediction_df)

            fig = px.line(prediction_df, x='Date', y='Predicted Pickups', title="🔮 7-Day Forecast", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if st.button("📊 Explain Prediction"):
            xai_report()

# 📥 Sidebar Form
def get_user_input():
    with st.sidebar.form(key="input_form"):
        st.markdown("## 📝 Input Parameters")

        scheduled_pickups = st.number_input("📦 Scheduled Pickups", min_value=0)
        family_size = st.number_input("👨‍👩‍👧 Family Size", min_value=0)

        st.markdown("### 🎉 Special Occasions")
        special_occasions = {
            f"special_occasion_{name}": st.checkbox(f"🎊 {name}")
            for name in ["Dhu al-Qadah", "Eid al-Adha", "Eid al-Fitr", "Muharram",
                         "Rabi al-Awwal", "Rajab", "Ramadan", "Shaban"]
        }

        season = st.radio("🌦 Season", ["Summer", "Winter", "Spring"])
        submitted = st.form_submit_button("Submit")

    return pd.DataFrame({
        "scheduled_pickups": [scheduled_pickups],
        "family_size": [family_size],
        **{key: [int(val)] for key, val in special_occasions.items()},
        "season_Summer": [int(season == "Summer")],
        "season_Winter": [int(season == "Winter")],
        "season_Spring": [int(season == "Spring")]
    })

# 📈 Prediction
def predict_hamper_pickups(input_features):
    data = pd.read_csv("data_for_modeling.csv", parse_dates=["pickup_date"])
    data.set_index("pickup_date", inplace=True)

    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

    exog_columns = exog_cols.copy()
    exog_data = []

    for date in forecast_dates:
      row = dict.fromkeys(exog_columns, 0)
      row["scheduled_pickups"] = input_features["scheduled_pickups"][0]
      row["family_size"] = input_features["family_size"][0]

      # Set the season flags based on the forecasted date
      month = date.month
      if month in [3, 4, 5]:
        row["season_Spring"] = 1
      elif month in [6, 7, 8]:
        row["season_Summer"] = 1
      elif month in [12, 1, 2]:
        row["season_Winter"] = 1

      # Copy over any special occasion flags from user input
      for col in exog_columns:
         if "special_occasion" in col and col in input_features.columns:
            row[col] = input_features[col][0]

      exog_data.append(row)

    exog_df = pd.DataFrame(exog_data)
    start = len(data)
    end = start + 6

    forecast = model.predict(start=start, end=end, exog=exog_df)
    return pd.DataFrame({"Date": forecast_dates, "Predicted Pickups": forecast.values})

# 📊 XAI Page
def xai_report():
    st.title("🧠 Explainable AI (XAI) Report")
    st.markdown("Understanding model behavior using coefficients")

    if "input_features" not in st.session_state:
        st.warning("Please predict first.")
        return

    coeffs = model.params
    df = pd.DataFrame(coeffs, columns=["Coefficient"])
    exog_coeffs = df[df.index.isin(exog_cols)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y=exog_coeffs.index, x=exog_coeffs["Coefficient"], palette="coolwarm", ax=ax)
    ax.set_title("SARIMA Feature Coefficients")
    st.pyplot(fig)

# 🤖 Chatbot
def rag_chatbot_page():
    st.title("💬 RAG Chatbot Assistant")
    st.write("Ask anything related to food hamper planning!")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

    # Contexts
    transaction_data = pd.DataFrame({
        "Client_ID": [101, 102, 103],
        "Name": ["Alice", "Bob", "Charlie"],
        "Scheduled_Pickups": [5, 3, 4],
        "Family_Size": [4, 3, 5],
        "Special_Occasion_Ramadan": [1, 0, 0],
        "Season_Summer": [1, 0, 0],
        "Season_Winter": [0, 1, 0],
        "Season_Spring": [0, 0, 1],
        "Location": ["Downtown", "Uptown", "Midtown"]
    })

    transaction_text = "\n".join(
        f"{row.Name} requested {row.Scheduled_Pickups} pickups (Family size: {row.Family_Size}) "
        f"during {['Summer', 'Winter', 'Spring'][[row.Season_Summer, row.Season_Winter, row.Season_Spring].index(1)]}."
        for _, row in transaction_data.iterrows()
    )

    charity_info = (
        "We optimize hamper delivery based on scheduled pickups, family size, and special events like Ramadan or Eid."
    )

    documents = {
        "charity_info": charity_info,
        "transactions": transaction_text
    }

    doc_embeddings = {
        doc_id: embedder.encode(text, convert_to_tensor=True)
        for doc_id, text in documents.items()
    }

    def retrieve_context(query):
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        top_doc = max(doc_embeddings.items(), key=lambda x: util.pytorch_cos_sim(query_embedding, x[1]).item())
        return documents[top_doc[0]]

    def rag_chatbot(query):
        context = retrieve_context(query)
        prompt = f"Context:\n{context}\n\nUser Question: {query}\n\nAnswer:"
        return generator(prompt, max_new_tokens=150)[0]["generated_text"].split("Answer:")[-1].strip()

    query = st.text_input("Ask your question")
    if st.button("Ask"):
        if query:
            answer = rag_chatbot(query)
            st.success("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

# 🚀 App Controller
def main():
    st.sidebar.title("📍 Navigation")
    choice = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "XAI Report", "Chatbot"])

    if choice == "Dashboard":
        dashboard()
    elif choice == "EDA":
        exploratory_data_analysis()
    elif choice == "ML Modeling":
        machine_learning_modeling()
    elif choice == "XAI Report":
        xai_report()
    elif choice == "Chatbot":
        rag_chatbot_page()

if __name__ == "__main__":
    main()
