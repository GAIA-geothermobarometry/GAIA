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

#set_png_as_page_bg('./imgs/Background.png')

im2 = Image.open("logo_noBG.png")
im3 = Image.open("imgs/GraphicalAbstract.jpg")

col1, col2 = st.columns([1.2, 1])
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
    #st.image(im3, width=450)
with col2:
    st.image(im2, width=350)
    

    ## INSTRUCTION PART ##

st.header("Instructions")
st.markdown("The structure of the file to be used as an input must be like the following:")           
input_example =  pd.read_excel('files/input_example.xlsx')
st.dataframe(input_example)

st.markdown("The columns ***Index***, ***sample***, ***notes*** and ***notes.1*** can be used to identify the samples.\
            The columns, ***SiO2***, ***TiO2***,  ***Al2O3***, ***Cr2O3***, ***FeO tot***, ***MnO***, ***NiO***, ***MgO***, ***CaO***, ***Na2O***, ***K2O*** and ***tot*** \
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

if st.button('Preprocess data and make predictions'):
    data = preprocessing(df)
    st.markdown(
        f'<p style="font-size:20px;border-radius:2%;">{"Preprocessed components:"}</p>',
        unsafe_allow_html=True)
    comp = data['components'].copy()
    comp['Sum'] = data['sum_of_components']
    st.dataframe(comp)
    st.markdown(
    f'<p style="font-size:20px;border-radius:2%;">{"Predictions in progress..."}</p>',
    unsafe_allow_html=True) 
    
    # predict and show results
    
    df_output = predict(data['input_NN'])   
    colcomp = df_output.columns[4:]
    df_output.loc[data['checks']['cpx_selection']==False, colcomp] ='n.c.'  # not computable samples (check not passed)
    st.write('Predicted values:')
    st.dataframe(df_output)
    

    # create multi sheet file
    #dictionary_output = {'Predictions':df_output, 'Cations': data['cations'], 'T site': data['site_T'],\
    #                     'M1 and M2 site': data['site_M1&2'],'Classifications':data['classifications'] ,\
    #                     'Components': comp , 'checks': data['checks']}
    #df_summary_output_xlsx = to_excel_multi_sheet(dictionary_output)
    #st.download_button(label='Download the output file', data=df_summary_output_xlsx , file_name= 'Prediction_' + nametuple[0] + '.xlsx')
  
  
    empty_col = pd.DataFrame(columns=['-'])
    out = pd.concat([df_output[df_output.columns[:4]],
                     empty_col,
                     df_output[df_output.columns[4:]],
                     empty_col,
                     data['major'][data['major'].columns[4:]],
                     empty_col,
                     data['site_T'],
                     empty_col,data['site_M1&2'],
                     empty_col,
                     data['classifications'],
                     empty_col,
                     comp[comp.columns[4:]],
                     empty_col,
                     data['checks']],
                     axis = 1 )
       
    #csv = convert_df(df_output)
    #st.download_button(
    #    label="Download prediction as csv!",
    #    data=csv,
    #    file_name='Prediction_' + nametuple[0] + '.csv',
    #    mime='text/csv',
    #)                    
    df_xlsx_pred = to_excel(out)
    st.download_button(label='Download the output file',
                       data=df_xlsx_pred,
                       file_name='Prediction_' + nametuple[0] + '.xlsx')

    
    
    col1, col2 = st.columns(2)
    with col1:
        plothist(df_output.loc[data['checks']['cpx_selection']])


