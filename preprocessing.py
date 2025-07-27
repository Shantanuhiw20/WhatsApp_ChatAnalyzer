import re
import pandas as pd

def preprocess(data):
    data = data.replace('\u202f', ' ')

    pattern = re.compile(
    r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm))'  # group 1 = date + time
    r'\s-\s'
    r'([^:]+?):'                                           # group 2 = sender
    r'\s'                                                  # space after the colon
    r'(.+?)(?=(?:\n\d{1,2}/\d{1,2}/\d{2,4},)|\Z)',          # group 3 = message until next date or end
    flags=re.IGNORECASE | re.DOTALL)

    matches = pattern.findall(data)

    dates    = [m[0] for m in matches]
    senders  = [m[1] for m in matches]
    msg = [m[2].strip() for m in matches]  # strip trailing newlines/spaces

    # Now if you really want “sender: message” in one string:
    messages = [f"{s}: {t}" for s, t in zip(senders, msg)]

    df = pd.DataFrame({'user_message':messages,'message_date':dates}) # Putting in a dataframe
    df['message_date'] = pd.to_datetime(df['message_date'], format= '%d/%m/%y, %I:%M %p') # Converting the data type
    df.rename(columns={'message_date':'date'}, inplace = True)

    senders = []
    texts   = []

    for msg in df['user_message']:
        if ": " in msg:
            # split into [sender, rest_of_message]
            sender, text = msg.split(": ", 1)
        else:
            # no “: ” means it’s a system/group notification
            sender, text = "group_notification", msg

        senders.append(sender)
        texts.append(text)

    # assign back to the DataFrame
    df['Sender']  = senders
    df['Message'] = texts

    # drop the old combined column if you like
    df.drop(columns=['user_message'], inplace=True)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    return df
