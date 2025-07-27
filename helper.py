import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from textblob import TextBlob
import emoji
from urlextract import URLExtract

# ------------------------------------------------------------------
# Helper module for WhatsApp Chat Analyzer
# ------------------------------------------------------------------

def filter_media(df):
    """
    Remove media-only, empty, or single punctuation messages (".", "?") from analysis.
    """
    mask = (~df['Message'].str.contains('<Media omitted>', na=False)) & \
           df['Message'].notna() & (df['Message'] != '') & \
           (~df['Message'].isin(['.', '?']))
    return df[mask].copy()

# 1. Overall stats including media and emoji counts

def fetch_stats(selected_user, df):
    """
    Return: total_messages, total_words, total_media, total_emojis
    for Overall or specific user.
    """
    if selected_user == 'Overall':
        subset = df
    else:
        subset = df[df['Sender'] == selected_user]

    total_messages = subset.shape[0]
    # media count
    total_media = subset['Message'].str.contains('<Media omitted>', na=False).sum()
    # text-only messages
    text_df = filter_media(subset)
    # word count
    total_words = text_df['Message'].str.split().apply(len).sum()
    # emoji count
    def extract_emojis(s): return [c for c in s if c in emoji.EMOJI_DATA]
    emojis = text_df['Message'].astype(str).apply(extract_emojis)
    total_emojis = emojis.apply(len).sum()
    # Links count
    extractor = URLExtract()
    y = []
    for m in df['Message']:
        y.extend(extractor.find_urls(m))

    return total_messages, total_words, total_media, total_emojis, len(y)

# 2. Messages per user (text only)

def messages_per_user(df):
    """
    Return a DataFrame with count of top 10 non-media messages per user, sorted descending.
    """
    text_df = filter_media(df)
    result = (text_df.groupby('Sender').size().reset_index(name='count').sort_values('count', ascending=False).head(10))
    return result

def avg_msg_per_user(df):
    total_messages = len(df)
    counts = df['Sender'].value_counts()
    percentages = counts / total_messages * 100

    # build the DataFrame
    df_avg = percentages.reset_index()
    df_avg.columns = ['Sender', 'Message Percentage']

    # format with two decimals + percent sign
    df_avg['Message Percentage'] = df_avg['Message Percentage'].map(lambda x: f"{x:.2f}%")
    df_avg.reset_index(drop=True, inplace=True)

    return df_avg

# 3. Activity heatmap data (text only)

def activity_heatmap(df):
    """
    Return pivot table: rows=hour, cols=weekday, values=count.
    """
    text_df = filter_media(df)
    text_df['weekday'] = text_df['date'].dt.day_name()
    text_df['hour'] = text_df['date'].dt.hour
    heat = text_df.groupby(['weekday','hour']).size().reset_index(name='count')
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heat['weekday'] = pd.Categorical(heat['weekday'], categories=days, ordered=True)
    pivot = heat.pivot(index='hour', columns='weekday', values='count').fillna(0)
    return pivot

# 4. Time series: daily & monthly (text only)

def daily_volume(df):
    """
    Return daily message counts for text-only messages.
    """
    text_df = filter_media(df).set_index('date')
    return text_df.resample('D').size().reset_index(name='count')


def monthly_volume(df):
    """
    Return monthly message counts for text-only messages.
    """
    text_df = filter_media(df).set_index('date')
    return text_df.resample('M').size().reset_index(name='count')

# 5. Word frequency & wordcloud (text only)

def top_n_words(df, n=20):
    """
    Return DataFrame of top n words (text-only), excluding stopwords.
    """
    text_df = filter_media(df)
    vec = CountVectorizer(max_features=n, stop_words='english')
    bag = vec.fit_transform(text_df['Message'].astype(str))
    words = vec.get_feature_names_out()
    counts = bag.sum(axis=0).A1
    return pd.DataFrame({'word': words, 'count': counts})


def generate_wordcloud(df):
    """
    Return a WordCloud for all text-only messages.
    """
    text = ' '.join(filter_media(df)['Message'].astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wc

# 6. Message type counts for pie chart

def message_type_counts(df):
    """
    Return tuple: text_count, media_count, link_count.
    """
    media = df['Message'].str.contains('<Media omitted>', na=False).sum()
    links = df['Message'].str.contains('http', na=False).sum()
    text_msgs = df.shape[0] - media
    return text_msgs, media, links

# 7. Sentiment over time (text only)

def sentiment_time_series(df, window='7D'):
    """
    Return rolling mean sentiment polarity over time for text-only msgs.
    """
    text_df = filter_media(df).copy()
    text_df['sentiment'] = text_df['Message'].apply(lambda m: TextBlob(m).sentiment.polarity)
    ts = text_df.set_index('date').resample(window)['sentiment'].mean().reset_index()
    return ts

# 8. Emoji usage stats (text only)

def top_emojis(df, n=10):
    """
    Return DataFrame of top n emojis by frequency in text-only messages.
    """
    text_df = filter_media(df)
    def extract_emojis(s): return [c for c in s if c in emoji.EMOJI_DATA]
    ems = text_df['Message'].astype(str).apply(extract_emojis)
    all_ems = [e for sub in ems for e in sub]
    counts = pd.Series(all_ems).value_counts().head(n)
    return counts.reset_index().rename(columns={'index':'emoji', 0:'count'})


