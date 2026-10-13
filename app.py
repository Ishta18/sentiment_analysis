import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")      
vectorizer = joblib.load("tfidf_vectorizer.pkl") 

st.title("📊 Sentiment Analysis App")
st.write("Enter a review or text below to analyze its sentiment.")

user_input = st.text_area("Review Text", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        X_input = vectorizer.transform([user_input])

        prediction = model.predict(X_input)[0]

        if prediction == "Positive":
            st.success("✅ Sentiment: Positive")
        elif prediction == "Negative":
            st.error("⚠️ Sentiment: Negative")
        else:
            st.info("➖ Sentiment: Neutral")
    else:
        st.warning("Please enter some text to analyze.")
