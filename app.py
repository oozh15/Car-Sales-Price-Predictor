import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

@st.cache_data
def load_data():
    f = r"C:\Users\yp104\Desktop\ds\car_purchasing.csv"
    df = pd.read_csv(f, encoding='ISO-8859-1')
    df = df.drop(columns=["customer name", "customer e-mail", "country"])
    df = df.dropna()
    return df

df = load_data()

st.sidebar.title("🚗 Car Sales Price Predictor")

if st.sidebar.checkbox("Show Dataset Sample"):
    st.write(df.head())

X = df.drop("car purchase amount", axis=1)
y = df["car purchase amount"]

categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(), categorical_cols)
    ])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5)
mean_cv_score = np.mean(cv_scores)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader(" Model Evaluation Metrics")
st.write(f"R² Score: `{r2:.2f}`")
st.write(f" Mean Absolute Error: `${mae:,.2f}`")
st.write(f" Mean Cross-Validation Score: `{mean_cv_score:.2f}`")

if st.checkbox("Show Actual vs. Predicted Plot"):
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Car Purchase Amount")
    st.pyplot(fig)

joblib.dump(model_pipeline, "car_model_pipeline.pkl")

st.title("Predict Car Purchase Amount")

gender = st.selectbox("gender", ["male", "female"])
age = st.slider("age", 18, 80, 30)
salary = st.number_input("annual Salary ($)", value=50000)
credit_card_debt = st.number_input("credit card debt ($)", value=1000)
net_worth = st.number_input("net worth ($)", value=100000)

gender_num = 1 if gender == "male" else 0

if st.button("Predict Purchase Amount"):
    input_data = pd.DataFrame([[gender_num, age, salary, credit_card_debt, net_worth]],
                              columns=["gender", "age", "annual Salary", "credit card debt", "net worth"])
    
    prediction = model_pipeline.predict(input_data)
    
    st.success(f" Predicted Car Purchase Amount: ${prediction[0]:,.2f}")
