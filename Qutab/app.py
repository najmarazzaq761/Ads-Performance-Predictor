import os
import pickle
import pandas as pd
import streamlit as st
import google.generativeai as genai


# ================================
# 🎯 Load ML model
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ads_predictor.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
except Exception as e:
    st.error("❌ Could not load 'ads_predictor.pkl'. Make sure it is in the repo root.")
    st.stop()


# ================================
# 🔐 Configure Gemini
# ================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Add it in Streamlit → Settings → Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro-latest")


# ================================
# 🎨 Social-Media Style CSS
# ================================
st.set_page_config(page_title="Social Media Engagement + Gemini", page_icon="📱", layout="wide")

st.markdown(
    """
    <style>
    /* Overall background */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        font-family: "system-ui", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    /* Header card */
    .hero-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 50%, #fad0c4 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        color: #222;
        box-shadow: 0 14px 30px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        opacity: 0.95;
    }
    /* Input card */
    .card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
        border: 1px solid #f2f2f2;
    }
    .card-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.7rem;
    }
    /* Predict button */
    .stButton>button {
        background: linear-gradient(135deg,#ff4b5c,#ff2e63);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.5rem 1.6rem;
        font-weight: 600;
        box-shadow: 0 10px 20px rgba(255,46,99,0.35);
    }
    .stButton>button:hover {
        filter: brightness(1.05);
    }
    /* Gemini bubble */
    .gemini-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.04);
        border: 1px solid #ffe6f0;
    }
    .gemini-title {
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================
# 📱 Hero Section
# ================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📱 Social Media Engagement Predictor + Gemini</div>
        <div class="hero-subtitle">
            Drop in your post details and get an estimated engagement score plus AI-powered
            suggestions to boost likes, comments and overall reach.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================
# 🧩 Input Form (original features)
# ================================
st.markdown('<div class="card"><div class="card-title">📝 Post Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    account_name = st.text_input("Account / Page Name", placeholder="e.g. Google, Nike, Your Brand")
    caption = st.text_area("Caption Text", height=120, placeholder="Type or paste your post caption here…")

    platform = st.selectbox(
        "Platform",
        ["Instagram", "Facebook", "TikTok", "YouTube", "Twitter (X)", "LinkedIn"],
        index=0,
    )

with col2:
    like_count = st.number_input("Like Count", min_value=0, step=1, value=300)
    comment_count = st.number_input("Comment Count", min_value=0, step=1, value=40)
    sentiment_score = st.slider(
        "Sentiment (how positive is the caption?)",
        min_value=-1.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        help="-1 = very negative, 0 = neutral, 1 = very positive",
    )

# Auto features
caption_length = len(caption) if caption else 0
word_count = len(caption.split()) if caption else 0

st.caption(f"📏 Caption length: **{caption_length}** characters · **{word_count}** words")

st.markdown("</div>", unsafe_allow_html=True)  # close card


# ================================
# 🔮 Prediction + Gemini Suggestions
# ================================
pred = None
gemini_text = None

if st.button("Predict Engagement"):
    if not caption.strip():
        st.warning("Please add a caption before predicting.")
    else:
        # Build input row exactly as model expects
        input_df = pd.DataFrame([{
            "account_name": account_name,
            "caption": caption,
            "caption_length": caption_length,
            "word_count": word_count,
            "sentiment_score": sentiment_score,
            "like_count": like_count,
            "comment_count": comment_count,
            "platform": platform,
        }])

        try:
            pred = float(ml_model.predict(input_df)[0])
            st.success(f"✅ Predicted Engagement Score: **{pred:.2f}**")

            # --- Gemini suggestions based on prediction ---
            prompt = f"""
You are a senior social media strategist.

Here is a social media post with its metadata:

- Account name: {account_name or "N/A"}
- Platform: {platform}
- Caption: {caption}
- Likes: {like_count}
- Comments: {comment_count}
- Caption length: {caption_length} characters
- Word count: {word_count}
- Sentiment score (-1 to 1): {sentiment_score}
- Predicted engagement score: {pred:.2f}

1. Briefly explain what this engagement score means in everyday marketing language.
2. Give 3–4 specific suggestions to improve the caption or post structure to increase engagement.
3. Suggest one improved caption version that keeps the same idea but is more engaging and platform-appropriate.
Use clear bullet points and short paragraphs.
"""

            try:
                gemini_response = gemini_model.generate_content(prompt)
                gemini_text = gemini_response.text
            except Exception as e:
                gemini_text = f"⚠️ Gemini could not generate suggestions.\n\nDetails: {e}"

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


if pred is not None and gemini_text:
    st.markdown(
        """
        <div class="gemini-card">
            <div class="gemini-title">💡 Gemini Suggestions for This Post</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(gemini_text)


# ================================
# 💬 Separate Gemini Chat
# ================================
st.markdown("---")
st.markdown('<div class="card"><div class="card-title">💬 Ask Gemini Anything</div>', unsafe_allow_html=True)

user_q = st.text_area("Type your question for Gemini (strategy, ideas, copywriting, etc.)", height=120)

if st.button("Ask Gemini"):
    if not user_q.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Gemini is thinking…"):
            try:
                resp = gemini_model.generate_content(user_q)
                st.markdown(resp.text)
            except Exception as e:
                st.error(f"⚠️ Gemini request failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)  # close card