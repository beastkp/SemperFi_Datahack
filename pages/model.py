import streamlit as st
from PIL import Image
from tensorflow import keras
import numpy as np
import cv2

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image = np.array(image)
    model = keras.models.load_model('D:\Data1\pages\model.h5')
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = np.reshape(image, (1, 256, 256, 3))
    preds = model.predict(image)
    rounded_predictions = np.argmax(preds, axis=1)
    if (rounded_predictions == 0):
        st.write("Damaged")
    else:
        st.write("Car Sahi hai")
