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

from preprocessing import *



def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def predict(data):
    for tg in [0, 1]:
        if tg == 0:
            directory = 'Pressure_models'
        else:
            directory = 'Temperature_models'

        targets = ['P (kbar)', 'T (C)']
        target = targets[tg]
        names_targets = ['pressure', 'temperature']
        names_target = names_targets[tg]

        sect = 'only_cpx'

        with open(directory + '/mod_' + names_target + '_' + sect + '/Global_variable.pickle', 'rb') as handle:
            g = pickle.load(handle)
        N = g['N']
        array_max = g['array_max']

        col = data.columns
        index_col = [col[i] for i in range(0, 4)]
        df_noindex = data.drop(columns=index_col)

        if tg == 0:
            df_output = pd.DataFrame(
                columns=index_col[:] + ['mean - ' + targets[0], 'std - ' + targets[0], 'mean - ' + targets[1],
                                        'std - ' + targets[1]])

        results = np.zeros((len(df_noindex), N))
        for e in range(N):
            print(e)
            model = tf.keras.models.load_model(
                directory + "/mod_" + names_target + '_' + sect + "/Bootstrap_model_" + str(e) + '.h5')
            results[:, e] = model(df_noindex.values.astype('float32')).numpy().reshape((len(df_noindex),))

        results = results * array_max[0]

        df_output[index_col] = df[index_col]
        df_output['mean - ' + target] = results.mean(axis=1)
        df_output['std - ' + target] = results.std(axis=1)
    return df_output


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


def plothist(df_output):
    targets = ['P (kbar)', 'T (C)']
    col = ['tab:green', 'tab:red']
    titles = ['P distribution', 'T distribution']
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    for tg in [0, 1]:
        x = df_output['mean - ' + targets[tg]].values.reshape(-1, 1)
        ax[tg].hist(df_output['mean - ' + targets[tg]].values, density=True, edgecolor='k', color=col[tg], label='hist')
        ax[tg].set_title(titles[tg], fontsize=13)
        ax[tg].set_xlabel(targets[tg], fontsize=13)
    fig.tight_layout(pad=2.0)
    st.pyplot(fig)


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

st.header("Instructions:")
st.markdown("The structure of the file to be used as an input must be like the following:")           
#input_example =  pd.read_excel('files/input_example.xlsx')
#st.table('input_example')

st.markdown("The columns ***Index***, ***sample***, ***notes*** and ***notes*** can be used to identify the samples.")
st.markdown("The columns, ***SiO2***, ***TiO2***, ***Al2O3***, ***Cr2O3***,***FeO tot***,***MnO***,***NiO***, ***MgO***, ***CaO***, ***Na2O*** and ***K2O*** \
             must be filled with the analyses.")
            
            
df_input_sheet = pd.read_excel('files/input_sheet.xlsx')
df_input_sheet_xlsx = to_excel(df_input_sheet)
st.download_button(label='Download the input file form', data=df_input_sheet_xlsx , file_name= 'input_sheet.xlsx')


# set_png_as_page_bg('./imgs/Background.png')

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
    
    
    df_output = predict(data['components'])
    
    colcomp = df_output.columns[4:]
    df_output.loc[data['checks']['cpx_selection']==False, colcomp] ='n.c.'  # not computable samples (check not passed)
    

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

    
    st.write('Predicted values:')
    st.dataframe(df_output)
    
    col1, col2 = st.columns(2)
    with col1:
        plothist(df_output.loc[data['checks']['cpx_selection']])


