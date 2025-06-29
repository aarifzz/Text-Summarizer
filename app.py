import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from transformers import pipeline
import fitz
import requests
from bs4 import BeautifulSoup

# Load summarizer
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # force CPU
)


st.title("📰 AI-Powered Text Summarizer")

option = st.sidebar.radio(
    "Choose your input type",
    ("Paste Text", "Upload PDF", "From Web URL")
)

if option == "Paste Text":
    text = st.text_area("Paste your text below", height=300)
    if st.button("Summarize"):
        if text:
            summary = summarizer(text[:1000], max_length=130, min_length=30, do_sample=False)
            st.subheader("Summary")
            st.write(summary[0]['summary_text'])
        else:
            st.warning("Please paste some text first.")

elif option == "Upload PDF":
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded:
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
        if st.button("Summarize PDF"):
            summary = summarizer(pdf_text[:1000], max_length=130, min_length=30, do_sample=False)
            st.subheader("Summary")
            st.write(summary[0]['summary_text'])

elif option == "From Web URL":
    url = st.text_input("Enter article URL")
    if st.button("Summarize from URL"):
        if url:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = soup.find_all("p")
                web_text = " ".join([p.get_text() for p in paragraphs])
                if web_text:
                    summary = summarizer(web_text[:1000], max_length=130, min_length=30, do_sample=False)
                    st.subheader("Summary")
                    st.write(summary[0]['summary_text'])
                else:
                    st.error("No text found at that URL.")
            except Exception as e:
                st.error(f"Error fetching URL: {e}")
        else:
            st.warning("Please enter a valid URL.")
