
import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Set the page title and layout
st.set_page_config(page_title="Purchase Prediction App", layout="wide")

# Load and display a background image
image = Image.open(r"Z:\Jupyter\srkNotes\Resume Project\1680187651527.jpeg")  # Replace with your image path
st.image(image, use_column_width=True)

# App title
st.title('Purchased Prediction App')
st.subheader('Predict whether a user will purchase their dream Car based on age and salary.')

# Input fields for user data
age = st.number_input('Enter Your Age', min_value=0, max_value=120)
salary = st.number_input('Enter Your Salary', min_value=0, step=100)

# Load the saved scaler
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Load the saved logistic regression model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Create a DataFrame from the user input
df = pd.DataFrame({'Age': [age], 'Salary': [salary]})

# Button to make prediction
if st.button('Predict Purchase'):
    # Scale the input data using the loaded scaler
    scaled_data = loaded_scaler.transform(df)

    # Predict the outcome using the loaded model
    output = loaded_model.predict(scaled_data)

    # Display the prediction result
    if output[0] == 1:
        st.success('✅ The user is likely to purchase the Car.')
    else:
        st.error('❌ The user is not likely to purchase the Car.')
