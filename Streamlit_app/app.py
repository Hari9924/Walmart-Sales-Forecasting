import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# Load Model
model = joblib.load("random_forest_model.joblib")

# Title
st.title("📈 Walmart Weekly Sales Forecasting")
st.markdown(
    """
    Predict weekly sales using a trained **Random Forest Regression Model**.
    Enter the store details and click **Predict Sales**.
    """
)

# Sidebar Inputs
st.sidebar.header("Input Features")

week = st.sidebar.slider("Week Number", 1, 52, 10)

store = st.sidebar.number_input(
    "Store ID",
    min_value=1,
    max_value=45,
    value=1
)

dept = st.sidebar.number_input(
    "Department",
    min_value=1,
    value=1
)

cpi = st.sidebar.number_input(
    "CPI",
    value=210.0,
    format="%.2f"
)

unemployment = st.sidebar.number_input(
    "Unemployment",
    value=7.5,
    format="%.2f"
)

size = st.sidebar.number_input(
    "Store Size",
    value=151315
)

store_type = st.sidebar.selectbox(
    "Store Type",
    ["A", "B", "C"]
)

# Type Encoding
type_mapping = {
    "A": 0,
    "B": 1,
    "C": 2
}

type_value = type_mapping[store_type]

# Input DataFrame
input_data = pd.DataFrame({
    "Week": [week],
    "CPI": [cpi],
    "Unemployment": [unemployment],
    "Size": [size],
    "Type": [type_value],
    "Dept": [dept],
    "Store": [store]
})

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    if st.button("🚀 Predict Sales"):

        prediction = model.predict(input_data)[0]

        st.metric(
            label="Predicted Weekly Sales",
            value=f"${prediction:,.2f}"
        )

        chart_data = pd.DataFrame({
            "Metric": ["Predicted Sales"],
            "Value": [prediction]
        })

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Value",
            title="Predicted Weekly Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

with col2:

    st.subheader("Model Information")

    st.info(
        """
        **Algorithm:** Random Forest Regressor

        **Features Used:**
        - Week
        - CPI
        - Unemployment
        - Size
        - Type
        - Department
        - Store
        """
    )

# Batch Prediction Section
st.markdown("---")

st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    predictions = model.predict(batch_df)

    batch_df["Predicted_Weekly_Sales"] = predictions

    st.dataframe(
        batch_df.head(),
        use_container_width=True
    )

    csv = batch_df.to_csv(index=False)

    st.download_button(
        "⬇ Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )

# About Section
with st.expander("About This Project"):

    st.write(
        """
        This application predicts Walmart weekly sales
        using Machine Learning.

        Model: Random Forest Regressor

        Built using:
        - Python
        - Streamlit
        - Scikit-Learn
        - Pandas
        - Plotly
        """
    )