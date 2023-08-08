# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:46:48 2023

@author: Bruse
"""
import pandas as pd
import numpy as np
from catboost import Pool, CatBoostRegressor
import pickle



column=['manufacturer', 'model', 'description', 'transmission','transmission_type', 'engine_size_cm3', 'fuel', 'powertrain','power_ps']
column=np.array(column)
column=column.astype('object')
df1=pd.DataFrame(columns=column)
cat_column = [df1.columns.get_loc(col) for col in ['manufacturer', 'model', 'description', 'transmission', 'transmission_type', 'fuel', 'powertrain']]
#loading the saved model
loaded_model=pickle.load(open('C:\deployement capstone project/train_model_capstone','rb'))

input_data = ("ALFA ROMEO","308","Limited 1.6 120hp E6d MT FWD","5MT","Manual",1338,"Diesel","Internal Combustion Engine (ICE)",145)

    # Convert the input_data to a DataFrame (assuming it's a single data point)
input_data_df = pd.DataFrame([input_data], columns=df1.columns)

    # Create the Pool object
pool_data = Pool(input_data_df, cat_features=cat_column)

# Make prediction using the trained CatBoost model
prediction = loaded_model.predict(pool_data)[0]
print("Predicted CO2 emissions: {:.2f} g/km".format(prediction))