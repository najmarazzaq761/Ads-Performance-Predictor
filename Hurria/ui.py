import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai  # 🔹 Gemini Import
from pathlib import Path  # 🔹 Added for safe pickle load
from dotenv import load_dotenv
import os

# ------------------------------
# 🔹 Configure Gemini API
# ------------------------------
load_dotenv()  # Load local .env

try:
    # Try Streamlit secrets first (Cloud)
    GOOGLE_API_KEY = st.secrets.get("GEMINI_API_KEY")
except st.errors.StreamlitSecretNotFoundError:
    # Fallback to .env for local development
    GOOGLE_API_KEY = os.getenv("API_KEY")

if not GOOGLE_API_KEY:
    st.error("API key not found! Please set GEMINI_API_KEY in Streamlit secrets or your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ------------------------------
# 🔹 Download Stopwords
# ------------------------------
nltk.download('stopwords')

# ------------------------------
# 🔹 Text Cleaning Function
# ------------------------------
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

# ------------------------------
# 🔹 Gemini Suggestion Function
# ------------------------------
def get_gemini_suggestions(caption, engagement_score):
    prompt = f"""
    I have a social media post with the following:

    Caption: {caption}
    Predicted Engagement Score: {engagement_score}

    Provide:
    1. Explanation of the score
    2. Suggestions to improve the post
    3. An improved caption
    4. Best hashtags to use
    """

    model_gemini = genai.GenerativeModel("gemini-pro-latest")
    response = model_gemini.generate_content([prompt])

    return response.text

# ------------------------------
# 🔹 Load Trained Model (Safe for Streamlit Cloud)
# ------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ads_predictor.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ------------------------------
# 🔹 Streamlit UI
# ------------------------------
st.set_page_config(page_title="Ad Performance Predictor", layout="centered")

st.title("📊 Ad Performance Predictor")
st.write("Predict how well your social media ad will perform — now with AI Suggestions powered by Gemini!")

# ------------------------------
# 🔹 User Inputs
# ------------------------------
caption = st.text_area("Ad Caption", placeholder="Write or paste your ad caption here...")
account_name = st.text_input("Brand / Account Name", placeholder="e.g. Nike")
platform = st.selectbox("Platform", ["Facebook", "Instagram", "Twitter", "LinkedIn"])

comment_count = st.number_input("Comment Count (expected)", min_value=0, step=1)
like_count = st.number_input("Like Count (expected)", min_value=0, step=1)

caption_length = len(caption)
word_count = len(caption.split())
sentiment_score = st.slider("Sentiment Score (-1 to 1)", -1.0, 1.0, 0.0)

# ------------------------------
# 🔹 Prediction + Gemini Suggestions
# ------------------------------
if st.button("Predict Ad Engagement"):

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

    # Predict engagement score
    predicted_score = model.predict(input_data)[0]
    st.success(f"Predicted Engagement Score: {round(predicted_score, 2)}")

    # 🔥 Generate AI Suggestions
    suggestions = get_gemini_suggestions(caption, predicted_score)

    st.subheader("🤖 AI Suggestions (Gemini)")
    st.write(suggestions)

# ------------------------------
# 🔹 Footer
# ------------------------------
st.markdown("---")
