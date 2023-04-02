import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import h5py
import xgboost as xgb
import os
import pickle as pkl
import pandas as pd


Year = st.text_input("How Much Old is Your Car")

Kilometers = st.text_input("Kilometers Driven")

Mileage = st.text_input("Mileage")

Engine = st.text_input("Engine (in CC)")

Power = st.text_input("Power")

No_Of_Seats = st.text_input("No. Of Seats")

options = ['CNG', 'Diesel', 'LPG', 'Petrol']
selected_fuel = st.selectbox('Fuel:', options)

options = ['Manual', 'Automatic']
selected_operation = st.selectbox('Transmission:', options)

options = ['1st_Owner', '2nd_Owner', '3rd_Owner', '4th_Owner']
selected_hand = st.selectbox('Owner Type:', options)

options = ['1st_Owner', '2st_Owner', '3st_Owner', '4st_Owner']
selected_hand = st.selectbox('Owner Type:', options)

if selected_fuel == 'CNG':
    selected_fuel_Diesel = 0
    selected_fuel_LPG = 0
    selected_fuel_Petrol = 0
if selected_fuel == 'Diesel':
    selected_fuel_Diesel = 1
    selected_fuel_LPG = 0
    selected_fuel_Petrol = 0
if selected_fuel == 'LPG':
    selected_fuel_Diesel = 0
    selected_fuel_LPG = 1
    selected_fuel_Petrol = 0
if selected_fuel == 'Petrol':
    selected_fuel_Diesel = 0
    selected_fuel_LPG = 0
    selected_fuel_Petrol = 1

if selected_operation == 'Manual':
    selected_operation_Manual = 1
if selected_operation == 'Automatic':
    selected_operation_Manual = 0

if selected_hand == '1st_Owner':
    selected_hand_2nd_Owner = 0
    selected_hand_3rd_Owner = 0
    selected_hand_4th_Owner = 0
if selected_hand == '2nd_Owner':
    selected_hand_2nd_Owner = 1
    selected_hand_3rd_Owner = 0
    selected_hand_4th_Owner = 0
if selected_hand == '3rd_Owner':
    selected_hand_2nd_Owner = 0
    selected_hand_3rd_Owner = 1
    selected_hand_4th_Owner = 0
if selected_hand == '4th_Owner':
    selected_hand_2nd_Owner = 0
    selected_hand_3rd_Owner = 0
    selected_hand_4th_Owner = 1

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file != None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image = np.array(image)
    model = keras.models.load_model(
        "C:/Users/Ayush/Downloads/parrot/parrot/pages/model.h5")
#     # model2 = keras.models.load_model('D:\Data1\pages\xgboost_model.h5')
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = np.reshape(image, (1, 256, 256, 3))
    preds = model.predict(image)
    rounded_predictions = np.argmax(preds, axis=1)

    if rounded_predictions == 0:
        st.write("Damaged")
    else:
        st.write("Car is not Damaged")


# Load the XGBoost model from the .h5 file
with open('C:/Users/Ayush/Downloads/parrot/parrot/pages/boosting.pkl', 'rb') as f:
    xgb_model = pkl.load(f)


print(f'year is {str(Year)}')
# input_data = []
submit = st.button("Submit")
if submit:
    input_data = [int(Year), int(Kilometers), float(Mileage), float(Engine), float(Power), float(No_Of_Seats), int(selected_fuel_Diesel), int(selected_fuel_LPG),
                      int(selected_fuel_Petrol), int(selected_operation_Manual), int(selected_hand_4th_Owner), int(selected_hand_2nd_Owner), int(selected_hand_3rd_Owner)]

    input_data = pd.DataFrame(np.array(input_data))
    print(input_data)
    # Use the loaded model to predict the output based on the input parameters
    output = xgb_model.predict(int(Year), int(Kilometers), float(Mileage), float(Engine), float(
        Power), float(No_Of_Seats), int(selected_fuel_Diesel), int(selected_fuel_LPG),
    int(selected_fuel_Petrol), int(selected_operation_Manual), int(selected_hand_4th_Owner),
    int(selected_hand_2nd_Owner), int(selected_hand_3rd_Owner))

    st.write('Output:', output[0])
