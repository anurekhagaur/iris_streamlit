import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load('iris_model.pkl')

# Load the Iris dataset for target names
iris = load_iris()

# Define the user interface
st.title("Iris Species Prediction")

st.write("""
This app predicts the **Iris species** based on the input features.
""")
import pandas as pd
# Sample data for the DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Streamlit app layout
st.title('DataFrame Display Example')

# Display the DataFrame in the Streamlit app
st.write("Here is the DataFrame:", df)


# Input fields for the features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0)

# Predict button
if st.button('Predict'):
    # Create an array from the input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make a prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f'The predicted Iris species is: {iris.target_names[prediction][0]}')
