GrowthLink
# Car Sales Price Predictor

This project predicts the car purchase amount based on user input features such as age, salary, credit card debt, and net worth using a Random Forest Regressor model.

## Objective
The objective of this project is to predict the car purchase amount for users based on various personal financial data.

## Features
- User input interface built with Streamlit for predicting car purchase amount.
- Random Forest Regressor model used for prediction.
- Model evaluation with R² Score and Mean Absolute Error.
- Cross-validation for model reliability.
- Actual vs predicted plot for model performance.

## Steps to Run the Project

1. Clone this repository to your local machine:
   git clone https://github.com/yourusername/car-sales-predictor.git


   1. Install Dependencies
Make sure all the necessary dependencies are installed. Based on your code, you need Streamlit, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and Joblib.

To install these, use the following command:
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib

2. Set Up Your Python Script
Make sure your Python script (the one containing your code for the Car Sales Price Predictor) is ready. The script should include the Streamlit components like this:
python app.py  # Replace with your Python file name (e.g., car_sales_predictor.py)

4. Running Your Streamlit Application Locally
In your project folder where the script is located, you can run the Streamlit app using the following command:
streamlit run app.py  # Replace 'app.py' with the name of your Python script
This will start a local web server, and Streamlit will open the app in your default web browser.

4. Interacting with the Application
Once the application opens, you'll be able to:
Input values into the fields provided (like gender, age, salary, etc.).
Click the "Predict Purchase Amount" button.
View the predicted car purchase amount based on your input.

5. Deploying the Application (Optional)
If you want others to access your application, you can deploy it to a cloud service such as Streamlit Cloud, Heroku, or any other platform that supports Python.
Here’s how you can deploy on Streamlit Cloud:
Go to Streamlit Cloud.
Sign in with your GitHub account.
Click on New App and connect it to your GitHub repository where the code is hosted.
Select the branch and script file (e.g., app.py) to deploy.
Click Deploy, and it will be live.

6. Updating the Application
If you make any changes to your code after deploying, just push the changes to GitHub (if it's linked with Streamlit Cloud) and it will automatically update the deployed app.
