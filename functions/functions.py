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


def libversions():
    return ['tensorflow', tf.__version__,
            'numpy', np.__version__,
            'pyxlsb', pyxlsb.__version__,
            'pickle', pickle.__version__,
            'matplotlib', matplotlib.__version__,
            'xlsxwriter', xlsxwriter.__version__,
            'inotify', inotify.__version__,
            'pandas', pandas.__version__]


def to_excel(df, index=False, startrow = 0):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=index, startrow=startrow, sheet_name='Sheet1')           
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def to_excel_multi_sheet(dic_cal):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    for key in dic_cal:   
        dic_cal[key].to_excel(writer, index=False, sheet_name=key)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def predict(data):    
    for tg in [0, 1]:
        if tg == 0:
            directory = 'Pressure_models'
            N = 100
            array_max = [10.0]
        else:
            directory = 'Temperature_models'
            N = 20
            array_max = [1400.0]

        targets = ['P (kbar)', 'T (C)']
        target = targets[tg]
        names_targets = ['pressure', 'temperature']
        names_target = names_targets[tg]
        sect = 'only_cpx'
        
        # Add a placeholder
        latest_iteration = st.empty()
        st.write('Predicting ' + names_target +' ...')
        bar = st.progress(0)
        
        #load global variable          
        #with open(directory + '/mod_' + names_target + '_' + sect + '/Global_variable.pickle', 'rb') as handle:
            #g = pickle.load(handle)
            #g = pd.read_pickle(handle)
        #N = g['N']
        #array_max = g['array_max']

        col = data.columns
        index_col = [col[i] for i in range(0, 4)]
        df_noindex = data.drop(columns=index_col)

        if tg == 0:
            df_output = pd.DataFrame(
                columns=index_col[:] + ['mean - ' + targets[0], 'std - ' + targets[0], 'mean - ' + targets[1],
                                        'std - ' + targets[1]])

        results = np.zeros((len(df_noindex), N))
        for e in range(N):
            
            #update bar
            latest_iteration.text(f'Applying model n°{e + 1}')
            bar.progress(int((e + 1)/N*100))
            time.sleep(0.1)
            
            #load modell
            model = tf.keras.models.load_model(
                directory + "/mod_" + names_target + '_' + sect + "/Bootstrap_model_" + str(e) + '.h5')
            results[:, e] = model(df_noindex.values.astype('float32')).numpy().reshape((len(df_noindex),))

        results = results * array_max[0]

        df_output[index_col] = data[index_col]
        mean =  results.mean(axis=1).round(np.mod(tg+1,2))
        std = results.std(axis=1).round(np.mod(tg+1,2))
        if tg == 1:
            mean = mean.astype('int')
            std = std.astype('int')
        df_output['mean - ' + target] = mean
        df_output['std - ' + target] = std
    return df_output


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file, opacity=1):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-opacity:opacity;
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
