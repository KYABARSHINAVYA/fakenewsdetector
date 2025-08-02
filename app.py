import streamlit as st
import pickle
from utils.text_cleaner import clean_text

# Load model and vectorizer
model = pickle.load(open('model/fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

# Streamlit App
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("📰 Fake News Detection")
st.write("Paste a news article and check whether it's Real or Fake.")

user_input = st.text_area("Enter News Content:", height=250)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        st.success(f"✅ Prediction: {prediction}")
