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
import torch

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
    generator = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

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
        "Season_Spring": [0, 0, 1],  # Whether it's Spring season (1 or 0)
        "Location": ["Downtown", "Uptown", "Midtown"],  # Client's location
    }

    # Convert the data dictionary into a DataFrame
    transaction_data = pd.DataFrame(data)

    # Generating a transaction narrative based on your updated data
    transaction_narrative = "Here are the latest food hamper transactions:\n"
    for idx, row in transaction_data.iterrows():
        special_occasions = [
            occ for occ in ['Dhu al-Qadah', 'Eid al-Adha', 'Eid al-Fitr', 'Muharram',
                            'Rabi al-Awwal', 'Rajab', 'Ramadan', 'Shaban']
            if row[f'Special_Occasion_{occ.replace(" ", "_")}'] == 1
        ]
        special_occasions_str = ', '.join(special_occasions)

        transaction_narrative += (
            f"Client {row['Client_ID']} ({row['Name']}, Family Size {row['Family_Size']}) requested "
            f"{row['Scheduled_Pickups']} scheduled hampers. The request was made during the "
            f"season of {'Summer' if row['Season_Summer'] == 1 else 'Winter' if row['Season_Winter'] == 1 else 'Spring'}.\n"
            f"The special occasion(s) influencing the demand were: "
            f"{special_occasions_str}.\n"  # Using the formatted string here
            f"Client's location: {row['Location']}.\n"
        )

    # Example charity info for the chatbot
    charity_info = (
    "Our organization works to provide food hampers based on seasonal demand, family size, and special occasions. "
    "We track various seasonal and cultural events, such as Ramadan, Eid, and others, which significantly influence the need for hampers. "
    "We also consider factors like family size and scheduled pickups when predicting the required number of hampers.\n\n"
    "We focus on understanding demand patterns across different seasons (Summer, Winter, Spring) and adjust our distribution accordingly. "
    "For example, larger families or families during special occasions like Ramadan or Eid might require more hampers. "
    "This helps us efficiently plan and ensure timely deliveries to those who need support."
    )

    documents = {
        "doc1": charity_info,
        "doc2": transaction_narrative
    }

    doc_embeddings = {
        doc_id: embedder.encode(text, convert_to_tensor=True)
        for doc_id, text in documents.items()
    }

    def retrieve_context(query, top_k=1):
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        scores = {}
        for doc_id, emb in doc_embeddings.items():
            score = util.pytorch_cos_sim(query_embedding, emb).item()
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_doc_ids = [doc_id for doc_id, score in sorted_docs[:top_k]]
        context = "\n\n".join(documents[doc_id] for doc_id in top_doc_ids)
        return context

    def query_llm(query, context):
        prompt = (
            "You have some background info plus transaction data below. "
            "Analyze the context and answer the user’s query clearly and succinctly.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            "Answer:"
        )

        outputs = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
        raw_output = outputs[0]['generated_text']

        if raw_output.startswith(prompt):
            raw_output = raw_output[len(prompt):].strip()

        return raw_output.strip()

    def rag_chatbot(query):
        context = retrieve_context(query, top_k=2)
        answer = query_llm(query, context)
        return answer

    user_query = st.text_input("Enter your question:")
    if st.button("Ask"):
        if user_query:
            answer = rag_chatbot(user_query)
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

# 🏠 **Main App Navigation**
def main():
    st.sidebar.title("📌 Navigation")
    app_page = st.sidebar.radio("📂 Select Page", ["Dashboard", "EDA", "ML Modeling", "XAI Report", "Chatbot"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "XAI Report":
        xai_report()
    elif app_page == "Chatbot":
        rag_chatbot_page()

if __name__ == "__main__":
    main()
