import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyxlsb import open_workbook as open_xlsb
import tensorflow as tf
import pickle
import os
from io import BytesIO
from PIL import Image
import base64
import time

from functions.preprocessing import *
from functions.functions import *


  ## GENERAL SETTINGS AND HEADER   ###

im = Image.open("logo_noBG.png")

st.set_page_config(
    page_title="GAIA - Home",
    page_icon=im,
    layout="wide"
)

im2 = Image.open("logo_noBG.png")

col1, col2 = st.columns([1.5, 1])
with col1:
    st.title("GAIA")
    st.header("Geo Artificial Intelligence thermobArometry")
    st.write(
        "A deep learning model to estimate temperatures and pressures of volcanoes starting from geochemical analysis.")
    st.write(
        "The model is based on [cit.] and use artificial neural networks to estimate the temperature and the pressure of \
        the magma chambers by starting from the geochemical analysis of rocks. The project was born from the collaboration \
        between the department of Physics and Astronomy and the Department of Earth Sciences of University of Florence. \
        Please see the info page to more information. ")
with col2:
    st.image(im2, width=350)

    ## INSTRUCTION PART ##

st.header("Instructions")
st.markdown("The structure of the file to be used as an input must be like the following:")           
input_example =  pd.read_excel('files/input_example.xlsx')
st.dataframe(input_example)

st.markdown("The columns ***Index***, ***sample***, ***notes*** and ***notes*** can be used to identify the samples.\
            The columns, ***SiO2***, ***TiO2***, ***Al2O3***, ***Cr2O3***,***FeO tot***,***MnO***,***NiO***, ***MgO***, ***CaO***, ***Na2O***, ***K2O*** and ***tot*** \
            must be filled with the oxides analyses. If the oxide has not been analysed or is below detection limit the corresponding cell can be leave blank or set to zero. \
            The same can be done if the total has not been calculated.")
            
st.markdown("An empty file with the right structure can be downloaded by using the button below.")
            
            
df_input_sheet = pd.read_excel('files/input_sheet.xlsx')
df_input_sheet_xlsx = to_excel(df_input_sheet)
st.download_button(label='Download the input file form', data=df_input_sheet_xlsx , file_name= 'input_sheet.xlsx')


    ## PROCESSING PART ##
    
st.header("Processing")

st.markdown("Upload a file with the structure as specified above:")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    filename = uploaded_file.name
    nametuple = os.path.splitext(filename)

    if nametuple[1] == '.csv':
        # read csv
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
    elif nametuple[1] == '.xls' or nametuple[1] == '.xlsx':
        # read xls or xlsx
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)
    else:
        st.warning("File type wrong (you need to upload a csv, xls or xlsx file)")

if st.button('Preprocess data'):
    data = preprocessing(df)
    st.markdown(
        f'<p style="font-size:20px;border-radius:2%;">{"Preprocessed components:."}</p>',
        unsafe_allow_html=True)
    comp = data['components'].copy()
    comp['Sum'] = data['sum_of_components']
    st.dataframe(comp)
    st.markdown(
    f'<p style="font-size:20px;border-radius:2%;">{"predictions in progress..."}</p>',
    unsafe_allow_html=True) 
    
    # predict and show results
    
    df_output = predict(data['components'])   
    colcomp = df_output.columns[4:]
    df_output.loc[data['checks']['cpx_selection']==False, colcomp] ='n.c.'  # not computable samples (check not passed)
    st.write('Predicted values:')
    st.dataframe(df_output)
    
    
    # Add a placeholder
    
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i + 1}')
        bar.progress(i + 1)
        time.sleep(0.1)
      
        
    csv = convert_df(df_output)
    st.download_button(
        label="Download data as csv!",
        data=csv,
        file_name='Prediction_' + nametuple[0] + '.csv',
        mime='text/csv',
    )
    df_xlsx_pred = to_excel(df_output)
    st.download_button(label='Download data as xlsx!',
                       data=df_xlsx_pred,
                       file_name='Prediction_' + nametuple[0] + '.xlsx')

    
    
    col1, col2 = st.columns(2)
    with col1:
        plothist(df_output.loc[data['checks']['cpx_selection']])


