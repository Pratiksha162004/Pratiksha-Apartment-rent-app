import streamlit as st
import pandas as pd
import plotly.express as px

# App config
st.set_page_config(page_title="Apartment Rent Explorer", layout="wide")
st.title("🏠 Apartment Rent Listings Explorer")

# ⬇️ Google Drive Dataset Link
st.markdown("📥 [Download CSV Dataset (from Google Drive)](https://drive.google.com/uc?id=1jK92nJKHyTxG9JXYZABC123456)")

# File uploader
uploaded_file = st.file_uploader("Upload the CSV file here", type=["csv"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    try:
        # Read CSV (semicolon-delimited, no header)
        raw_df = pd.read_csv(uploaded_file, sep=';', header=None, encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
    else:
        # Use first row as header
        header = raw_df.iloc[0]
        df = raw_df[1:].copy()
        df.columns = header
        df = df[df.columns[df.columns.notna()]]  # Remove empty unnamed columns
        df.reset_index(drop=True, inplace=True)

        # Clean rows where ID is invalid
        if 'id' in df.columns:
            df = df[df['id'].astype(str).str.isnumeric()]

        # Convert numeric columns
        if 'fee' in df.columns:
            df['fee'] = pd.to_numeric(df['fee'], errors='coerce')
        if 'bedrooms' in df.columns:
            df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')

        # Show in tabs
        tab1, tab2 = st.tabs(["📄 Data Preview", "📊 Graphs"])

        # Tab 1: Table
        with tab1:
            st.subheader("🧾 Dataset Overview (Clean Table)")
            st.dataframe(df, use_container_width=True)

        # Tab 2: Graphs
        with tab2:
            st.subheader("📊 Graphs Based on Dataset")

            if 'category' in df.columns:
                fig1 = px.histogram(df, x='category', title='Category Distribution')
                st.plotly_chart(fig1, use_container_width=True)

            if 'fee' in df.columns:
                fig2 = px.histogram(df, x='fee', nbins=40, title='Fee Distribution')
                st.plotly_chart(fig2, use_container_width=True)

            if 'bedrooms' in df.columns:
                fig3 = px.pie(df, names='bedrooms', title='Bedroom Counts')
                st.plotly_chart(fig3, use_container_width=True)

            if 'category' not in df.columns and 'fee' not in df.columns and 'bedrooms' not in df.columns:
                st.warning("⚠️ No suitable columns found for graphs.")
