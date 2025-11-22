import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv
import os

nltk.download('stopwords')

# 1. Text Cleaning Function

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


# 2. Load Trained Model

with open("ads_predictor.pkl", "rb") as f:
    model = pickle.load(f)


# 3. Configure Gemini API
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_gemini_suggestions(caption, engagement_score):
    prompt = f"""
    I have a social media post with:

    Caption: {caption}
    Predicted Engagement Score: {engagement_score}

    Give me:
    - Detailed analysis of why engagement is high/low
    - Suggestions to improve the caption
    - An improved rewritten caption
    - 5–10 high-engagement hashtags
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text


# 4. Streamlit UI

st.set_page_config(page_title="Ad Performance Predictor", layout="centered")

st.title("Ad Performance Predictor")

# Session State to store prediction
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "cleaned_caption" not in st.session_state:
    st.session_state.cleaned_caption = None


tab1, tab2 = st.tabs(["Prediction", "LLM Suggestions"])


# TAB 1: PREDICTION MODULE

with tab1:
    st.subheader("Enter Ad Details")

    caption = st.text_area("Ad Caption", placeholder="Write or paste your caption...")
    account_name = st.text_input("Brand / Account Name")
    platform = st.selectbox("Platform", ["Facebook", "Instagram", "Twitter", "LinkedIn"])

    comment_count = st.number_input("Expected Comment Count", min_value=0, step=1)
    like_count = st.number_input("Expected Like Count", min_value=0, step=1)
    caption_length = len(caption)
    word_count = len(caption.split())
    sentiment_score = st.slider("Sentiment Score (-1 to 1)", -1.0, 1.0, 0.0)

    if st.button("Predict Ad Engagement"):
        cleaned = clean_caption(caption)

        input_df = pd.DataFrame([{
            "caption": cleaned,
            "account_name": account_name,
            "platform": platform,
            "comment_count": comment_count,
            "like_count": like_count,
            "caption_length": caption_length,
            "word_count": word_count,
            "sentiment_score": sentiment_score
        }])

        pred = model.predict(input_df)[0]

        # Store in session state
        st.session_state.prediction = round(pred, 2)
        st.session_state.cleaned_caption = cleaned

        st.success(f"Predicted Engagement Score: {st.session_state.prediction}")

    st.markdown("---")
    st.caption("Developed by Najma Razzaq & Abdullah")


# TAB 2: LLM SUGGESTION MODULE

with tab2:
    st.subheader("Gemini Caption Improvement Suggestions")

    if st.button("Generate Suggestions"):
        if st.session_state.prediction is None:
            st.warning("Please predict engagement first in the Prediction tab.")
        else:
            suggestions = get_gemini_suggestions(
                st.session_state.cleaned_caption,
                st.session_state.prediction
            )
            st.write(suggestions)
