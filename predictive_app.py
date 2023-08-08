# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 23:32:58 2023

@author: Bruse
"""



import numpy as np
import pickle
import streamlit as st
import pandas as pd
from catboost import Pool

#loading the saved model
loaded_model=pickle.load(open("C:/deployement capstone project/predictive_app.py",'rb'))
#loaded_model=pickle.load(open('C:\deployement capstone project/train_model_capstone','rb'))
column=['manufacturer', 'model', 'description', 'transmission','transmission_type', 'engine_size_cm3', 'fuel', 'powertrain','power_ps']
column=np.array(column)
column=column.astype('object')
df1=pd.DataFrame(columns=column)
cat_column = [df1.columns.get_loc(col) for col in ['manufacturer', 'model', 'description', 'transmission', 'transmission_type', 'fuel', 'powertrain']]
#creating a function for prediction
def implement(input_data):
    
    # Convert the input_data to a DataFrame (assuming it's a single data point)
    input_data_df = pd.DataFrame([input_data], columns=df1.columns)

    # Create the Pool object
    pool_data = Pool(input_data_df, cat_features=cat_column)

    # Make prediction using the trained CatBoost model
    prediction = loaded_model.predict(pool_data)[0]
    print("Predicted CO2 emissions: {:.2f} g/km".format(prediction))
    return "Predicted CO2 emissions: {:.2f} g/km".format(prediction)

def main():
    #creatin a title
    st.title(" CO2 emissions predict web app")
    manufacturer = st.text_input("enter the manufaturer: ")
    model =st.text_input("enter the model: ")
    description = st.text_input("enter the description: ")
    transmission = st.text_input("enter the transmisssion: ")
    transmission_type = st.text_input("enter the transmisssion_type: ")
    engine_size=st.number_input("enter the engine_size: ")
    fuel = st.text_input("enter the fuel type: ")
    powertrain = st.text_input("enter the powertrain: ")
    power_ps=st.number_input("enter the power in ps: ")
    
    input_data = [manufacturer, model, description, transmission, transmission_type,engine_size, fuel, powertrain,power_ps]        
    #code for prediction
    dignosis= ''
    
    #creating a button
    
    if st.button("prediction test"):
        
        
        # Get the predicted CO2 emissions
        dignosis = implement(input_data)
        
    st.success(dignosis)

if __name__=='__main__':
    main()  