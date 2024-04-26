import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('medical_cost.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
def predict_cost(age, sex, bmi, smoker):
    features = np.array([age, sex, bmi, smoker])
    features = features.reshape(1,-1)
    cost = model.predict(features)
    return cost[0]

# Streamlit UI
st.title('MEDICAL COST PREDICTION')
st.write("""
## Input Features
ENTER THE VALUES FOR THE INPUT FEATURES TO PREDICT MEDICAL COST.
""")

# Input fields for user

age = st.number_input('age')
sex = st.number_input('sex')
bmi = st.number_input('bmi')
smoker = st.number_input('smoker')

# Prediction button
if st.button('Predict'):
    # Predict cost
    cost_prediction = predict_cost(age, sex, bmi, smoker)
    st.write(f"PREDICTED MEDICAL COST: {cost_prediction}")