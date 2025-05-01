import streamlit as st
import preprocessing
import helper

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

    #fetch unique users
    user_list = df['Sender'].unique().tolist()
    user_list.sort()
    user_list.insert(0,"Overall")
    selected_user = st.sidebar.selectbox("Show Analysis with respect to", user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages = helper.fetch_stats(selected_user,df)
        
        col1, col2, col3, col4 = st.beta_columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
 

