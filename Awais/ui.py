import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
import os

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Ad Performance Predictor", layout="centered")

# ------------------------------
# Load API Key
# ------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("❌ API Key not found in .env file")

# ------------------------------
# NLTK Stopwords
# ------------------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords", quiet=True)

# ------------------------------
# Text Cleaning
# ------------------------------
def clean_caption(text):
    try:
        stop = set(stopwords.words("english"))
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return " ".join([w for w in text.split() if w not in stop])
    except:
        return text

# ------------------------------
# List Available Models (Debug)
# ------------------------------
def list_available_models():
    """Helper function to check available models"""
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        return available_models
    except Exception as e:
        return [f"Error listing models: {str(e)}"]

# ------------------------------
# Gemini AI Suggestions (FIXED)
# ------------------------------
def get_gemini_suggestions(caption, engagement_score):
    prompt = f"""Analyze this social media post:

Caption: {caption}
Predicted Engagement Score: {engagement_score}

Please provide:
1. Explanation of the engagement score
2. Suggestions to improve the caption
3. An improved version of the caption
4. Best hashtags to use (5-10 relevant hashtags)

Format your response clearly with numbered sections."""

    # List of model names to try (in order of preference)
    # Using Gemini 2.x models which are available in your account
    model_names = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-2.0-flash",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]
    
    last_error = None
    
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=800,
                )
            )
            return response.text
        except Exception as e:
            last_error = str(e)
            continue
    
    # If all models fail, show available models
    available = list_available_models()
    return f"""❌ Unable to connect to Gemini AI

**Error:** {last_error}

**Available Models:**
{chr(10).join(available)}

**Troubleshooting Steps:**
1. Verify your API key at: https://aistudio.google.com/app/apikey
2. Make sure Gemini API is enabled
3. Check if you're using the correct API key in .env file
4. Ensure billing is enabled (free tier available)

**Your API Key Status:** {"✅ Key Found" if GOOGLE_API_KEY else "❌ No Key"}"""

# ------------------------------
# Load ML Model
# ------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ads_predictor.pkl"

model = None
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.warning("⚠️ Model file 'ads_predictor.pkl' not found. Prediction will be unavailable.")
except Exception as e:
    st.error(f"❌ Error loading ML model: {e}")

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📊 Ad Performance Predictor")
st.write("Predict social media ad performance — powered by **ML + Gemini AI**")

# Sidebar for debug info
with st.sidebar:
    st.header("🔧 Debug Info")
    st.write(f"**API Key Status:** {'✅ Loaded' if GOOGLE_API_KEY else '❌ Missing'}")
    st.write(f"**Model Status:** {'✅ Loaded' if model else '❌ Not Loaded'}")
    
    if st.button("🔍 Check Available Models"):
        with st.spinner("Checking..."):
            models = list_available_models()
            st.write("**Available Gemini Models:**")
            for m in models:
                st.code(m, language=None)

# Main form
st.subheader("📝 Enter Ad Details")

caption = st.text_area(
    "Ad Caption", 
    placeholder="Enter your ad caption here...",
    help="The main text content of your social media ad"
)

col1, col2 = st.columns(2)
with col1:
    account_name = st.text_input("Account Name", placeholder="Nike, Adidas...")
    platform = st.selectbox("Platform", ["Facebook", "Instagram", "Twitter", "LinkedIn"])
    comment_count = st.number_input("Comment Count", min_value=0, value=0)

with col2:
    like_count = st.number_input("Like Count", min_value=0, value=0)
    sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.1)

caption_length = len(caption)
word_count = len(caption.split())

st.write(f"**Caption Stats:** {word_count} words, {caption_length} characters")

# ------------------------------
# Predict + Gemini Suggestions
# ------------------------------
if st.button("🚀 Predict Ad Engagement", type="primary"):
    if not caption.strip():
        st.warning("⚠️ Please enter a caption first!")
    else:
        # Prediction
        if model is not None:
            try:
                cleaned_caption = clean_caption(caption)
                df = pd.DataFrame([{
                    "caption": cleaned_caption,
                    "account_name": account_name,
                    "platform": platform,
                    "comment_count": comment_count,
                    "like_count": like_count,
                    "caption_length": caption_length,
                    "word_count": word_count,
                    "sentiment_score": sentiment_score
                }])

                predicted_score = model.predict(df)[0]
                st.success(f"✅ **Predicted Engagement Score:** {round(predicted_score, 2)}")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                predicted_score = 0.0
        else:
            st.info("ℹ️ ML model not available. Using default score.")
            predicted_score = 5.0
        
        # Gemini Suggestions
        st.subheader("🤖 Gemini AI Suggestions")
        
        if not GOOGLE_API_KEY:
            st.error("❌ Cannot generate suggestions: API key not found")
        else:
            with st.spinner("Generating AI suggestions..."):
                suggestions = get_gemini_suggestions(caption, predicted_score)
                st.markdown(suggestions)

st.markdown("---")
st.caption("Developed by AWAIS HANIF, HOORIA ABBASI, QUTAB SHAH")