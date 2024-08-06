import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
clf = pickle.load(open("case_study_emp.pkl","rb"))

def predict(data):
    clf = pickle.load(open("case_study_emp.pkl","rb"))
    return clf.predict(data)


st.title("Employee Attrition Rate Project using Machine Learning")
st.markdown("This Model Identify weather performance of employee are good no not")

st.header("Employee Details")
col1,col2 = st.columns(2)

with col1:
	st.text("Education Level")
	el=st.slider("Education Level", 1,5,2)
	st.text("Time of Service")
	tos = st.slider("Time of Service", 1.0, 100.0, 0.5)
	st.text("Time of Promotion")
	top = st.slider("Time of Promotion", 0.0, 50.0, 0.5)
	
with col2:
	st.text("Growth Rate")
	gr = st.slider("Growth Rate", 1,5,2)
	st.text("Post Level")
	gr1 = st.slider("Post Level", 1,5,2)

st.text('')
if st.button("Predict Performance Rate"):
    result = clf.predict(
        np.array([[el,tos,top,gr,gr1,1,1,1,1,1]]))
    st.text(result[0])
st.markdown("Developed by Nishant Doma Sawaimoon in NIELT")
