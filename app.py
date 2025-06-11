import nltk
nltk.data.path.append("./nltk_data")
import streamlit as st
from keyword_extractors import (
    extract_tfidf_keywords,
    extract_rake_keywords,
    extract_textrank_keywords
)
import PyPDF2

st.set_page_config(page_title="Keyword Extractor", layout="centered", initial_sidebar_state="auto")

st.title("🔍 Keyword Extraction App")

# File uploader section
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

text = ""

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

# Fallback to manual input if no file uploaded
if not text:
    text = st.text_area("Or paste your text here:", height=200)

# Proceed only if text is available
if text:
    st.subheader("Choose Extraction Method")
    method = st.selectbox("Choose Extraction Method", ["TF-IDF", "RAKE", "TextRank"])

    num_keywords = st.slider("Number of keywords to extract", min_value=5, max_value=30, value=10)

    if st.button("Extract Keywords"):
        if method == "TF-IDF":
            keywords = extract_tfidf_keywords(text,num_keywords )
        elif method == "RAKE":
            keywords = extract_rake_keywords(text,num_keywords)
        else:
            keywords = extract_textrank_keywords(text, num_keywords)


        st.subheader("Extracted Keywords")
        st.write(keywords)
else:
    st.info("Upload a file or enter text above to begin.")
