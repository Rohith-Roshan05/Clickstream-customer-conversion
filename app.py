import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the trained model
def load_pickle_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the scaler
def load_pickle_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Preprocessing function for data
def preprocess_data(df):
    # Convert 'page2_clothing_model' from string (e.g., 'C20') to numeric
    if 'page2_clothing_model' in df.columns:
        df['page2_clothing_model'] = df['page2_clothing_model'].str.extract('(\d+)', expand=False).astype(int)
    return df

# Function to make predictions
def make_prediction(df, model_path, scaler_path, features):
    model = load_pickle_model(model_path)
    scaler = load_pickle_scaler(scaler_path)
    
    df_scaled = scaler.transform(df[features])
    predictions = model.predict(df_scaled)
    return predictions

# Main function for Streamlit app
def main():
    st.title("🛒 Clickstream Prediction App")
    st.sidebar.header("Upload Data or Enter Manually")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
    model_type = st.sidebar.radio("Select Model Type", ("Regression", "Classification"))
    
    # Define features and models for regression/classification
    if model_type == "Regression":
        expected_features = ['page1_main_category', 'page2_clothing_model', 'colour']
        model_path = "Z:\\Clickstream\\xgb_regressor.pkl"
        scaler_path = "Z:\\Clickstream\\scaler_reg.pkl"
        expected_target = "price"
    else:
        expected_features = ['month', 'page1_main_category', 'order', 'price', 'page2_clothing_model', 'colour', 'location', 'model_photography']
        model_path = "Z:\\Clickstream\\logistic_model_cls.pkl"
        scaler_path = "Z:\\Clickstream\\scaler_cls.pkl"
        expected_target = "price_2"
    
    # Handling file upload
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)
        st.subheader("Uploaded Data Preview")
        st.write(df)
        
        # Ensure required features exist in uploaded data
        missing_features = [feature for feature in expected_features if feature not in df.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
            return
        
        # Allow user to select features and target
        features = st.sidebar.multiselect("Select Features", expected_features, default=expected_features)
        target = st.sidebar.selectbox("Select Target", [expected_target])
        
        if features and target:
            df = df[features + [target]]
            row_index = st.sidebar.selectbox("Select a Row for Prediction", df.index)
            selected_row = df.loc[[row_index], features]
        else:
            st.warning("Please select features and target to proceed.")
    
    else:
        st.sidebar.subheader("Enter Data Manually")

        if model_type == "Regression":
            # Input fields for regression model
            page1_main_category = st.sidebar.number_input("Page1 Main Category", min_value=1, max_value=5, value=3)
            page2_clothing_model = st.sidebar.text_input("Page2 Clothing Model (e.g. C20)", "C20")
            colour = st.sidebar.number_input("Colour", min_value=1, max_value=20, value=5)

            df = pd.DataFrame({
                'page1_main_category': [page1_main_category],
                'page2_clothing_model': [page2_clothing_model],
                'colour': [colour]
            })

        else:  # Classification Model
            # Input fields for classification model
            page1_main_category = st.sidebar.number_input("Page1 Main Category", min_value=1, max_value=10, value=3)
            page2_clothing_model = st.sidebar.text_input("Page2 Clothing Model (e.g. C20)", "C20")
            colour = st.sidebar.number_input("Colour", min_value=1, max_value=20, value=5)
            order = st.sidebar.number_input("Order", min_value=1, max_value=50, value=10)
            price = st.sidebar.number_input("Price", min_value=1, max_value=1000, value=50)
            location = st.sidebar.number_input("Location", min_value=1, max_value=10, value=3)
            model_photography = st.sidebar.number_input("Model Photography", min_value=1, max_value=5, value=2)
            month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)

            df = pd.DataFrame({
                'page1_main_category': [page1_main_category],
                'page2_clothing_model': [page2_clothing_model],
                'colour': [colour],
                'order': [order],
                'price': [price],
                'location': [location],
                'model_photography': [model_photography],
                'month': [month]
            })

        selected_row = preprocess_data(df)

    # Prediction button
    if st.button("Predict"):        
        predictions = make_prediction(selected_row, model_path, scaler_path, expected_features)
        
        # **Updated Classification Output**
        if model_type == "Regression":
            st.success(f"Predicted Price: ${round(predictions[0])}")
        else:
            if int(predictions[0]) == 1:
                st.success("🛍️ The user is likely to **buy** the product!")
            else:
                st.warning("🚫 The user is **not** likely to buy the product.")

        # **Visualization**
        st.subheader("📊 Visualization of Predictions")
        fig, ax = plt.subplots()
        ax.bar(expected_features, selected_row[expected_features].values[0], color='skyblue')
        ax.set_ylabel("Feature Values")
        ax.set_title("Feature Importance in Prediction")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
