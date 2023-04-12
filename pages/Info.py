import streamlit as st
import pandas as pd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import openpyxl
import streamlit as st
from PIL import Image
import webbrowser


def to_excel_multi_sheet(dic_cal):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    for key in dic_cal:   
        dic_cal[key].to_excel(writer, index=False, sheet_name=key)
    #workbook = writer.book
    #worksheet = writer.sheets['Sheet1']
    #format1 = workbook.add_format({'num_format': '0.00'}) 
    #worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data


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

im = Image.open("logo_noBG.png")
st.set_page_config(
    page_title="GAIA - Info",
    page_icon=im,
    layout="wide"
)

st.title("GAIA")

    
st.write("The anatomy of the plumbing system of active volcanoes is fundamental to understanding how magma is stored and channeled to the surface. Reliable geothermobarometric estimates are, therefore, critical to assess the depths and temperatures of the complex system of magmatic reservoirs that form a volcano apparatus. Here, we developed a novel Machine Learning approach based upon Feedforward Neural Networks (GAIA) to estimate P-T conditions of magma storage and migration within the crust. Our Feedforward Neural Network method applied to clinopyroxene compositions yields better uncertainties (Root-Mean-Square Error and R2 score) than previous Machine Learning methods and set the basis for a novel generation of reliable geothermobarometers, which extends beyond the paradigm associated to crystal-liquid equilibrium. Also, the bootstrap procedure, inherent to the Feedforward Neural Network architecture, permits to perform a rigorous assessment of the P-T uncertainty associated to each clinopyroxene composition, as opposed to the Root-Mean-Square Error representing the P-T uncertainty of whole set of clinopyroxene compositions.")

st.header("The people")
col1, col2,col3, col4 = st.columns([1,1.45,1.5,1])

with col1:
    im = Image.open("imgs/lorenzo.jpeg")
    st.image(im,use_column_width=True, caption='Lorenzo Chicchi, Università degli Studi di Firenze, Dipartimento di Fisica e Astrofisica, INFN')
with col2:
    #st.write("#")
    im = Image.open("imgs/luca.png")
    st.image(im,use_column_width=True, caption='Prof. Luca Bindi, Università degli Studi di Firenze, Dipartimento di Scienze della Terra')
with col3:
    im = Image.open("imgs/duccio.png")
    st.image(im,use_column_width=True, caption='Prof. Duccio Fanelli, Università degli Studi di Firenze, Dipartimento di Fisica e Astrofisica, INFN')
with col4:
    im = Image.open("imgs/simone.png")
    st.image(im,use_column_width=True, caption='Prof. Simone Tommasini, Università degli Studi di Firenze, Dipartimento di Scienze della Terra')



