import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# --- Page Configuration ---
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Dependencies ---
@st.cache_resource
def load_resources():
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    model = joblib.load("stacking_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return stop_words, model, vectorizer

stop_words, model, vectorizer = load_resources()

# --- Helper Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# --- Custom CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%);
        color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #aab2cd;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Card Styling */
    .stTextArea > div {
        border-radius: 15px;
        border: 1px solid #3b3b5c;
        background-color: #2b2b40;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stTextArea textarea {
        color: #ffffff;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6);
    }

    /* Result Cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
        margin-top: 2rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .safe-message {
        background-color: rgba(46, 204, 113, 0.15);
        border: 1px solid #2ecc71;
        color: #2ecc71;
    }
    
    .danger-message {
        background-color: rgba(231, 76, 60, 0.15);
        border: 1px solid #e74c3c;
        color: #e74c3c;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #2d2d44;
    }
    
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Content ---
with st.sidebar:
    st.image("logo.png", width=120)
    st.markdown("### Detector AI")
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Enter a tweet or text snippet.
    2. Click **Analyze Text**.
    3. Our AI model predicts if it contains hate speech.
    
    **Model:** Stacking Classifier (LinearSVC, Logistic Regression)
    """)
    st.markdown("---")
    st.caption("© 2024 Hate Speech Detector")

# --- Main Layout ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<h1 class="main-title">Twitter Hate Speech Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Analyze text for potential hate speech/offensive language using AI.</p>', unsafe_allow_html=True)

    # Input Section
    user_input = st.text_area("Enter your text here...", height=150, placeholder="Type something to analyze...")

    # Action Buttons
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        if st.button("Analyze Text ⚡"):
            if user_input.strip() == "":
                st.warning("⚠️ Please provide some text to analyze.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    cleaned = clean_text(user_input)
                    vectorized = vectorizer.transform([cleaned])
                    prediction = model.predict(vectorized)[0]

                # Result Display
                if prediction == 1:
                    st.markdown("""
                        <div class="result-card danger-message">
                            <h2>🚨 Hate Speech Detected</h2>
                            <p>The text contains offensive language or hate speech patterns.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="result-card safe-message">
                            <h2>✅ No Hate Speech Detected</h2>
                            <p>The text appears to be neutral or safe.</p>
                        </div>
                    """, unsafe_allow_html=True)

    # Examples Section
    st.markdown("### Try Examples")
    cols = st.columns(3)
    examples = [
        "I love this beautiful day!",
        "This is absolutely terrible and disgusting.",
        "You are amazing and kind."
    ]
    
    # Note: Streamlit buttons don't easily update text_area values directly without session state
    # Implementing a simple workaround or just showing them as static text for now if state management is complex for this snippet
    # Using Session State for examples
    
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

    # This part is a bit tricky with just buttons filling the text area above, 
    # usually requires `st.experimental_rerun()` or callback.
    # For simplicity in this aesthetic upgrade, I'll just list them as copy-pasteable or simple text.
    
    # Better approach for examples in standard Streamlit:
    # Use st.form or just let user type. 
    # Adding clickable pills would be nice but requires rerun.
    
