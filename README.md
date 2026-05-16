# WhatsApp Chats Analyser

> Lightweight Streamlit app to parse and analyse exported WhatsApp chat `.txt` files.

## Overview

This project provides tools to preprocess WhatsApp chat exports and produce interactive visual analyses (message counts, timelines, heatmaps, word frequency, emoji stats and simple sentiment trends) using a Streamlit UI.

## Features

- Upload a WhatsApp chat export `.txt` file via the sidebar
- Per-user or overall analysis
- KPIs: total messages, words, media, emojis, links
- Activity heatmap (hour vs weekday)
- Daily and monthly time series
- Top words and wordcloud
- Message type breakdown (text / media / links)
- Emoji leaderboard
- 7-day rolling sentiment polarity

## Project Layout

- `app.py` — Streamlit application that provides the UI and charts.
- `preprocessing.py` — Parser for raw WhatsApp .txt export; converts to a DataFrame with `date`, `Sender`, and `Message` columns and adds time breakdown columns.
- `helper.py` — Analysis helpers (statistics, heatmap, time series, wordcloud, sentiment, emoji extraction).
- `whatsapp_chat_analysis.ipynb` — Notebook with exploratory analysis and examples (optional).

## Expected Input

The parser in `preprocessing.py` expects WhatsApp exported chats formatted like:

`DD/MM/YY, HH:MM AM/PM - Sender: message`

Examples:

- `12/05/21, 10:05 PM - Alice: Hello!`
- `01/01/2022, 9:30 AM - Bob: <Media omitted>`

Notes:
- The current regex in `preprocessing.py` uses a 12-hour time format with an `am`/`pm` suffix and the `'%d/%m/%y, %I:%M %p'` datetime format. If your export uses a different locale/time format, you may need to adjust the parser.

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
.
```

2. Activate the virtual environment and install dependencies:

On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload your WhatsApp `.txt` export and click `Show Analysis`.

## Development notes

- `TextBlob` is used for sentiment; it may require additional corpora for advanced NLP (e.g., via `nltk.download`). For simple polarity scoring TextBlob works out-of-the-box but installing NLTK corpora can improve results.
- `wordcloud` requires `Pillow`/image support.
- The preprocessing regex is intentionally permissive for multi-line messages; if you encounter missed messages, adjust the regex in `preprocessing.py`.

## Dependencies

Dependencies are provided in `requirements.txt`. Install them into a virtual environment before running the app.

## Next steps / Improvements

- Add unit tests for the parser and helper functions.
- Add an option to detect and adapt to 24-hour timestamp formats.
- Cache expensive computations in `helper.py` to speed up repeated analyses.

---

If you want, I can also:

- pin exact package versions in `requirements.txt`
- add a small `.gitignore` and a short CONTRIBUTING guide
- run the app locally and verify it starts successfully
