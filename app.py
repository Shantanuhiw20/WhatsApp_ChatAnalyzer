import streamlit as st
import preprocessing

st.sidebar.title("Whatsapp Chat Analyser")

uploaded_file = st.sidebar.file_uploader("Choose a File")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessing.preprocess(data)

    styler = df.style.format({
    'year': '{:d}',       # no commas on ints
    'day':  '{:d}',
    'hour': '{:d}',
    # leave month alone, it’s already a string
    })
    st.dataframe(styler)

    
    

