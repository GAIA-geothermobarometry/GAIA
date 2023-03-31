import streamlit as st
import pandas as pd
import pandas as pd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import streamlit as st
from PIL import Image
import webbrowser

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
    page_title="Deep Thermobarometer",
    page_icon=im,
    layout="wide"
)

st.title("Info")

st.header("Istructions")






    
st.write("Before using the deep learning model is necessary to make a calculation of clinopyroxene components and check on analysis quality as described in [].")

url = 'https://github.com/GAIA-geothermobarometry/GAIA/raw/main/pages/Calculation.xlsx/'
if st.button('Download the calculation file'):
    webbrowser.open_new_tab(url)

#df_calc = pd.read_excel('pages/Calculation.xlsx') 
#df_calc_xlsx = to_excel(df_calc)
#st.download_button(label='Download the calculation file', data=df_calc_xlsx ,file_name= 'Calculation.xlsx')

st.write("To carry out the processing it is necessary to download the Calculation.xlsx file with the botton above and follow the subsequent steps, each relating to a sheet of the file.")
st.markdown("- **data input**: Clinopyroxene analyses. Input the analyses (paste special values) as indicated in the example (blank cell if the oxide has not been analysed or is below detection limit).")
st.markdown("- **calculation-1**:	Calculation of clinopyroxene formula based on 4 cations and Fe3+ on charge balance. Do use the fill down command from column A to column CR to perform the calculation on all the analyses.")
st.markdown("- **calculation-2**:	Calculation of clinopyroxene components. Do use the fill down command from column A to FF to perform the calculation on all the analyses.")
st.markdown("- **data output**: -	Do use the fill down command from column A to BV to perform the check on all the analyses.")
st.markdown("- **data sorting** -	Sorting of the analyses to be used in the Feedforward Neural Network clinopyroxene-only geoT-P. Copy and paste values from data output, sort the data according to column CA (true/false), and then paste the index, sample, and cpx components in the empty form file of the app (button below). Drag and drop the empty form in the Home page, run the app and then save and copy the results in columns F-I of the data sorting sheet.")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)

#df_empy = pd.read_excel('pages/Form_input.xlsx') 
#df_xlsx = to_excel(df_empy)
st.download_button(label='Download an empty form here!',
                                data='pages/Form_input.xlsx',#df_xlsx ,
                                file_name= 'Empty_form.xlsx')


st.write("A the end of the elaboration the input dataset must be a .xlsx file with the following structure:")
st.write("**Only clinopyroxene dataset:**")
df = pd.read_excel('pages/Example_input.xlsx')
st.table(df)


st.header("References")
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




