# Clickstream-customer-conversion

### 📌 Overview

This project analyzes clickstream data to predict customer conversion. Using machine learning models, it identifies key patterns in user interactions to determine the likelihood of a purchase.

### 📈 Data

- train_data.csv: Used for training the models.

- test_data.csv: Used for evaluation and testing.

### 🚀 Usage

- Data Cleaning & Preprocessing: Run Clickstream_Datacleaning.ipynb to clean and preprocess the data. 

- Model Training & Prediction:The trained models (logistic_model_cls.pkl and xgb_regressor.pkl) are used for classification and regression tasks.

- Scalers (scaler_cls.pkl and scaler_reg.pkl) normalize the input data.

- Run the Streamlit App:Upload a CSV file with clickstream data to get conversion predictions.

### 📊 Machine Learning Models

- Classification: Logistic Regression Model (logistic_model_cls.pkl) predicts customer conversion.

- Regression: XGBoost Regressor (xgb_regressor.pkl) estimates probability scores for conversion likelihood

 

### 🔥 Features Used

- Clickstream behavior features: Number of pages visited, session duration, click rate.

- User attributes: Country, category of products viewed, and color preferences.

### 💡 Future Enhancements

- Improve feature engineering for better accuracy.

- Implement additional models for comparison.

- Optimize the Streamlit app UI/UX.


---
### 📬 Contact
**Author**: Rohith V  
**Email**: Rohithroshan047@gmail.com  

⭐ **If this project helps you, don't forget to star it on GitHub!** ⭐
