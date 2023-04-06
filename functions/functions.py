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
