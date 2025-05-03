import streamlit as st
import preprocessing
import helper
import pandas as pd
import plotly.express as px
from datetime import timedelta

# --------------------------------------------------
# Streamlit App: Chats Analyser
# --------------------------------------------------

# Page config
st.set_page_config(layout="wide", page_title="Chats Analyser")

# Custom WhatsApp-themed CSS
st.markdown(
    """
    <style>
    /* Sidebar */
    .css-1d391kg {background-color: #075E54;}
    .css-1v3fvcr {color: #ffffff;}
    /* Main header */
    .app-header {background-color: #128C7E; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .app-header h1 {color: #ffffff; margin: 0;}
    /* Metrics styling */
    .stMetric > div:nth-child(1) {color: #25D366;}  /* titles */
    .stMetric > div:nth-child(2) {color: #075E54;}  /* values */
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Chats Analyser")
uploaded_file = st.sidebar.file_uploader("Upload WhatsApp chat (.txt)")

# Only show controls once file uploaded
if uploaded_file:
    # Preprocess raw data once
    raw = uploaded_file.getvalue().decode('utf-8')
    df = preprocessing.preprocess(raw)

    # User filter dropdown
    users = ['Overall'] + sorted(df['Sender'].unique().tolist())
    selected = st.sidebar.selectbox("Select User", users, key='user_select')

    # Show Analysis button
    show_btn = st.sidebar.button("Show Analysis")

    if show_btn:
        # Filter for selected user
        df_sel = df if selected == 'Overall' else df[df['Sender'] == selected]

        # Display header
        st.markdown(
            "<div class='app-header'><h1>Chats Analyser</h1></div>",
            unsafe_allow_html=True
        )

        # Fetch stats
        msgs, words, media_count, emoji_count = helper.fetch_stats(selected, df)

        # KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", msgs)
        col2.metric("Total Words", words)
        col3.metric("Total Media Shared", media_count)
        col4.metric("Total Emojis Shared", emoji_count)

        # 1. Messages per user
        st.subheader("Messages per User")
        mpu = helper.messages_per_user(df_sel)
        fig1 = px.bar(
            mpu, x='Sender', y='count',
            color='count', color_continuous_scale='Greens',
            title=None
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Activity heatmap
        st.subheader("Activity Heatmap (Hour vs Weekday)")
        heat = helper.activity_heatmap(df_sel)
        fig2 = px.imshow(
            heat, aspect='auto', origin='lower',
            color_continuous_scale='Tealgrn',
            labels={'x':'Weekday','y':'Hour','color':'Messages'},
            title=None
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Time series
        st.subheader("Time Series")
        daily = helper.daily_volume(df_sel)
        fig3 = px.line(
            daily, x='date', y='count',
            color_discrete_sequence=['#25D366'],
            title='Daily Volume'
        )
        st.plotly_chart(fig3)
        monthly = helper.monthly_volume(df_sel)
        fig4 = px.line(
            monthly, x='date', y='count',
            color_discrete_sequence=['#128C7E'],
            title='Monthly Volume'
        )
        st.plotly_chart(fig4)

        # 4. Response times
        st.subheader("Response Time")
        dist, avg = helper.compute_response_times(df_sel)
        fig5 = px.histogram(
            dist, nbins=50,
            title='Reply Distribution (hrs)',
            color_discrete_sequence=['#25D366']
        )
        st.plotly_chart(fig5)
        fig6 = px.bar(
            avg, x='Sender', y='avg_hours',
            title='Avg Reply Time (hrs)',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig6)

        # 5. Word analysis
        st.subheader("Top Words & Wordcloud")
        topw = helper.top_n_words(df_sel)
        fig7 = px.bar(
            topw, x='word', y='count',
            title='Top 20 Words',
            color_continuous_scale='Greys'
        )
        st.plotly_chart(fig7)
        wc = helper.generate_wordcloud(df_sel)
        st.image(wc.to_array(), use_column_width=True)

        # 6. Message types
        st.subheader("Message Types")
        txt, med, links = helper.message_type_counts(df_sel)
        fig8 = px.pie(
            names=['Text','Media','Links'],
            values=[txt, med, links],
            color_discrete_sequence=['#128C7E','#075E54','#25D366'],
            title=None
        )
        st.plotly_chart(fig8)

        # 7. Sentiment over time
        st.subheader("Sentiment Over Time")
        sent = helper.sentiment_time_series(df_sel)
        fig9 = px.line(
            sent, x='date', y='sentiment',
            color_discrete_sequence=['#34B7F1'],
            title='7-Day Rolling Sentiment'
        )
        st.plotly_chart(fig9)

        # 8. Emoji leaderboard
        st.subheader("Top Emojis")
        topem = helper.top_emojis(df_sel)
        fig10 = px.bar(
            topem, x='emoji', y='count',
            title='Top Emojis',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig10)
else:
    st.markdown(
        "<h2>Upload a file to begin.</h2>",
        unsafe_allow_html=True
    )
