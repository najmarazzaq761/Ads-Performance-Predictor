import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Text cleaning function

def clean_caption(text):
    stop = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop])
    return text


# Load trained model

with open("ads_predictor.pkl", "rb") as file:
    model = pickle.load(file)


# Streamlit UI
st.set_page_config(page_title="Ad Performance Predictor", layout="centered")

st.title("📊 Ad Performance Predictor")
st.write("Predict how well your social media ad will perform based on its features!")


# User Inputs

caption = st.text_area("Ad Caption", placeholder="Write or paste your ad caption here...")
account_name = st.text_input("Brand / Account Name", placeholder="e.g. Nike")
platform = st.selectbox("Platform", ["Facebook", "Instagram", "Twitter", "LinkedIn"])

comment_count = st.number_input("Comment Count (expected)", min_value=0, step=1)
like_count = st.number_input("Like Count (expected)", min_value=0, step=1)
caption_length = len(caption)
word_count = len(caption.split())
sentiment_score = st.slider("Sentiment Score (-1 to 1)", -1.0, 1.0, 0.0)


# Prediction Button

if st.button("Predict Ad Engagement"):
    # Prepare data for model
    cleaned_caption = clean_caption(caption)

    input_data = pd.DataFrame([{
        "caption": cleaned_caption,
        "account_name": account_name,
        "platform": platform,
        "comment_count": comment_count,
        "like_count": like_count,
        "caption_length": caption_length,
        "word_count": word_count,
        "sentiment_score": sentiment_score
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Engagement Score: {round(prediction, 2)}")

# Footer

st.markdown("---")
st.caption("Developed by Najma Razzaq & Abdullah")
