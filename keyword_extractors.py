from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from summa import keywords as textrank_keywords


# TF-IDF
def extract_tfidf_keywords(text, num_keywords):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in sorted_scores[:num_keywords]]
    return keywords
def extract_rake_keywords(text, num_keywords):
    r = Rake()
    r.extract_keywords_from_text(text)
    ranked_phrases = r.get_ranked_phrases()
    return ranked_phrases[:num_keywords]
def extract_textrank_keywords(text, num_keywords):
    from summa import keywords as textrank_keywords

def extract_textrank_keywords(text, num_keywords):
    if not text or not text.strip():
        return []

    try:
        # Use ratio instead of words to control number of keywords
        raw_keywords = textrank_keywords.keywords(text, ratio=0.5, split=True, scores=False)
        # Return only the top N
        return raw_keywords[:num_keywords]
    except Exception as e:
        print(f"TextRank extraction failed: {e}")
        return []
