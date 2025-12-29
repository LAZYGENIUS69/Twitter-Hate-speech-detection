
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (first run only)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load trained model and vectorizer
model = joblib.load("stacking_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

st.title("Twitter Hate Speech Detection")
st.write("Enter a tweet to check if it contains hate speech.")

user_input = st.text_area("Tweet")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("Hate Speech Detected ❌")
        else:
            st.success("No Hate Speech Detected ✅")
