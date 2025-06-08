import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("cleaned_apartment_data.csv")

# Streamlit config
st.set_page_config(page_title="Apartment Rent Predictor", layout="wide")

st.markdown("<h1 style='text-align: center;'>🏡 Apartment Rent Predictor</h1>", unsafe_allow_html=True)

tabs = st.tabs(["📊 Graphs", "📈 Predictor"])

# ======== Graphs TAB ==========
with tabs[0]:
    st.markdown("## 🧮 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔢 Bedrooms Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="bedroom", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("#### 💰 Rent Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["rent"], kde=True, ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 📏 Area vs Rent")
        fig, ax = plt.subplots()
        sns.scatterplot(x="area", y="rent", data=df, ax=ax)
        st.pyplot(fig)

    with col4:
        st.markdown("#### 🚿 Bathrooms vs Rent")
        fig, ax = plt.subplots()
        sns.boxplot(x="bathrooms", y="rent", data=df, ax=ax)
        st.pyplot(fig)

    st.markdown("#### 🌇 Balcony vs Rent")
    fig, ax = plt.subplots()
    sns.boxplot(x="balcony", y="rent", data=df, ax=ax)
    st.pyplot(fig)

# ======== Predictor TAB ==========
with tabs[1]:
    st.markdown("## 🛠️ Enter apartment details:")

    area = st.number_input("Total Area (sq. ft.):", min_value=100.0, max_value=10000.0, value=1000.0, step=10.0)
    bedroom = st.number_input("Number of Bedrooms:", min_value=1.0, max_value=5.0, value=2.0, step=1.0)
    bathrooms = st.number_input("Number of Bathrooms:", min_value=1.0, max_value=5.0, value=2.0, step=1.0)
    balcony = st.number_input("Number of Balconies:", min_value=0.0, max_value=3.0, value=1.0, step=1.0)

    if st.button("🔴 Predict Rent"):
        input_df = pd.DataFrame([[area, bathrooms, balcony, bedroom]],
                                columns=["area", "bathrooms", "balcony", "bedroom"])
        predicted_rent = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div style="background-color: #1f513f; padding: 15px; border-radius: 10px; color: white; font-size: 18px;">
                💰 <b>Estimated Rent:</b> ₹{predicted_rent:.2f} per month
            </div>
            """,
            unsafe_allow_html=True
        )
