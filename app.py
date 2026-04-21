
# PharmaGuard AI - Fraud Detection Prototype
# import necessary libraries
import streamlit as st 
import pandas as pd
import openai as ai


st.title("PharmaGuard AI")
st.write("Fraud detection prototype")


# Upload file 
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Read the file (CSV file)
    df = pd.read_csv(uploaded_file)
    
    st.write("File uploaded successfully!")
    st.write(df.head())  # Display the first few rows of the dataframe


