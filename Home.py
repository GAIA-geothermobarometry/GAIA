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
im3 = Image.open("imgs/GraphicalAbstract.jpg")

col1, col2 = st.columns([1.2, 1])
with col1:
    st.title("GAIA")
    st.header("Geo Artificial Intelligence thermobArometry")
    st.write(
        "Deep learning artificial neural network for P-T estimates of volcano plumbing systems using clinopyroxene composition.")
    st.write(
        "The project was born from the collaboration between the Department of Physics and Astronomy and the Department of Earth Sciences of the University of Firenze, \
         Italy. See the info page for details on people who developed the app. ")
with col2:
    st.image(im2, width=350)
    

    ## INSTRUCTION PART ##

st.header("Instructions")
st.markdown("The structure of the file to be used as input must be like the following:")           
input_example =  pd.read_excel('files/input_example.xlsx')
st.dataframe(input_example)

st.markdown( "The columns ***Index***, ***sample***, ***notes*** and ***notes.1*** can be used to identify clinopyroxenes. The major element\
            composition of clinopyroxene (wt%) must be in the indicated order \
            (***SiO2***, ***TiO2***,  ***Al2O3***, ***Cr2O3***, ***FeO tot***, ***MnO***, ***NiO***, ***MgO***, ***CaO***, ***Na2O***, ***K2O***). \
            Whether an oxide has not been analysed or is below detection limit input either “0” or leave it as a blank cell (not “-“ or other characters).\
            The same applies to the last column (***total***).")
            
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
  
    # Generate the output dataframe with multiindex columns
    
    dfs = [df_output[df_output.columns[:4]],
           df_output[df_output.columns[4:]],        
           data['major'][data['major'].columns[4:]],
           data['site_T'],
           data['site_M1&2'],
           data['classifications'],
           comp[comp.columns[4:]],
           data['checks']]
    global_labels = ['Samples', 'Predictions','Major Elements', 'Site T', 'Site M1&M2', ' Classifications', ' Components', 'Checks']
    
    empty_col = pd.DataFrame(columns=['-'])
    
    concat_df = pd.concat([dfs[0],empty_col,dfs[1],empty_col,dfs[2],empty_col,dfs[3],empty_col,dfs[4],empty_col,dfs[5],empty_col,dfs[6],empty_col,  dfs[7]], axis = 1 )
    st.dataframe(concat_df)
    bound = [[0,4],[5,9],[10,23], [24,28], [29,40], [41,46], [47,59], [60,71]]
    col_tuple = []
    for i in range(8):
        col_tuple = col_tuple + [(global_labels[i], c) for c in concat_df.columns[bound[i][0]:bound[i][1]]] + [('-','-')]
    col_tuple = col_tuple[:-1]

    out = pd.DataFrame(concat_df.values, columns= pd.MultiIndex.from_tuples(col_tuple), index = df.index)
    
    #csv = convert_df(df_output)
    #st.download_button(
    #    label="Download prediction as csv!",
    #    data=csv,
    #    file_name='Prediction_' + nametuple[0] + '.csv',
    #    mime='text/csv',
    #)                    
    df_xlsx_pred = to_excel(out, index=True)
    st.download_button(label='Download the output file',
                       data=df_xlsx_pred,
                       file_name='Prediction_' + nametuple[0] + '.xlsx')

    
    
    col1, col2 = st.columns(2)
    with col1:
        plothist(df_output.loc[data['checks']['cpx_selection']])


