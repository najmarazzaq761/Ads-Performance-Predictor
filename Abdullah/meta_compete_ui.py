import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv
import os

nltk.download('stopwords', quiet=True) # Added quiet=True

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

try:
    with open("ads_predictor.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'ads_predictor.pkl' not found.")
    model = None


# 3. Configure Gemini API (Simplified Text Output)
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- UPDATED FUNCTION TO ACCEPT PLATFORM AND ACCOUNT ---
def get_gemini_suggestions(caption, engagement_score, platform, account_name):
    if not GOOGLE_API_KEY:
        return "ERROR: Gemini API key not configured."
        
    # ---  PROMPT FOR SUGGESTIONS ---
    prompt = f"""
    You are an **Expert Social Media Strategist**.
    
    I have a social media ad post with the following details:

    * **Platform:** {platform}
    * **Account Name (Brand):** {account_name if account_name else 'N/A'}
    * **Original Caption:** {caption}
    * **Predicted Engagement Score:** {engagement_score}
    
    (Assume an average score for this platform is around 50.)

    Please give professional, actionable, and platform-specific suggestions to improve this post.
    
    I need four clear sections in your response, formatted using Markdown headings and lists:
    
    ## 1. Engagement Analysis
    * Explain simply why the score ({engagement_score}) is likely high or low based on the content and platform best practices.
    
    ## 2. Actionable Tips (For Max Engagement)
    * Provide 3-5 quick, specific tips to make the caption better (e.g., adding a stronger Call-to-Action, using platform-specific features, improving tone).
    
    ## 3. Improved Caption
    * Provide one new, improved, and catchy caption that is optimized for {platform} and the brand tone.
    
    ## 4. Top 5 Hashtags
    * List 5 high-engagement and relevant hashtags for the post.
    """
    # --------------------------------------------------

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"ERROR communicating with Gemini API: {e}"


# 4. Streamlit UI

st.set_page_config(page_title="Ad Performance Predictor", layout="centered")

st.title("Ad Performance Predictor")

# Session State
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "original_caption" not in st.session_state:
    st.session_state.original_caption = ""
if "platform" not in st.session_state:
    st.session_state.platform = ""
if "account_name" not in st.session_state:
    st.session_state.account_name = ""


tab1, tab2 = st.tabs(["Prediction", "AI Suggestions"])


# TAB 1: PREDICTION MODULE

with tab1:
    st.subheader("Enter Ad Details")

    caption = st.text_area("Ad Caption", placeholder="Write or paste your caption...")
    account_name = st.text_input("Brand / Account Name")
    platform = st.selectbox("Platform", ["Facebook", "Instagram"])

    comment_count = st.number_input("Comments", min_value=0, step=1)
    like_count = st.number_input("Likes", min_value=0, step=1)
    caption_length = len(caption)
    word_count = len(caption.split())
    sentiment_score = st.slider("Sentiment Score (-1 to 1)", -1.0, 1.0, 0.0)

    if st.button("Predict Ad Engagement"):
        if model is None:
            st.warning("Cannot predict: Model failed to load.")
        elif not caption:
             st.warning("Please enter a caption to predict engagement.")
        else:
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

            try:
                pred = model.predict(input_df)[0]
                # Store in session state, including context data
                st.session_state.prediction = round(pred, 2)
                st.session_state.original_caption = caption
                st.session_state.platform = platform # Store platform
                st.session_state.account_name = account_name # Store account name
                st.success(f"Predicted Engagement Score: {st.session_state.prediction}")
            except Exception as e:
                 st.error(f"Prediction Error: {e}")


    st.markdown("---")
    st.caption("Developed by Mr.Abdullah & Miss Najma Razzaq")


# TAB 2: LLM SUGGESTION MODULE

with tab2:
    st.subheader("Gemini Caption Improvement Suggestions")

    # --- UPDATED BUTTON LOGIC TO PASS CONTEXT ---
    if st.button("Get Enhanced Suggestions"):
        if st.session_state.prediction is None:
            st.warning("Please predict engagement first in the Prediction tab.")
        else:
            with st.spinner('Asking Gemini for expert tips...'):
                suggestions_text = get_gemini_suggestions(
                    st.session_state.original_caption,
                    st.session_state.prediction,
                    st.session_state.platform, # Pass Platform
                    st.session_state.account_name # Pass Account Name
                )
            
            st.markdown(suggestions_text)