import streamlit as st
import pickle
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

import nltk
import os

# Set writable download path
nltk.data.path.append('/tmp')

def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir='/tmp')
        nltk.download('punkt_tab', download_dir='/tmp')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir='/tmp')

download_nltk()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)



tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


with open('style.css', 'r') as css_file:
    st.markdown(f'<style>{css_file.read()}</style>', unsafe_allow_html=True)


st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown('<div class="main-container">', unsafe_allow_html=True)


st.markdown('''
<div class="header-section">
    <div class="header-content">
        <h1 class="main-title">SPAM SMS/EMAIL DETECTOR</h1>
        <p class="subtitle">Advanced Message Classification</p>
    </div>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-title"> Analyze Message</h2>', unsafe_allow_html=True)

message = st.text_area(
    label='',
    placeholder='Paste your email or SMS message...',
    height=100,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1.5, 1])

with col2:
    predict_btn = st.button('🔎 ANALYZE', use_container_width=True, key='predict_btn')

st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if predict_btn:
    if message.strip():
        # Preprocessing
        transform_sms = transform_text(message)
        
        # Vectorization
        vector_input = tfidf.transform([transform_sms])
        
        # Prediction
        result = model.predict(vector_input)[0]
        
        # Display result
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        
        if result == 1:
            st.markdown('''
            <div class="result-card spam">
                <div class="result-icon">⚠️</div>
                <h2 class="result-title">SPAM DETECTED</h2>
                <p class="result-description">This message is likely spam or phishing</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="result-card legitimate">
                <div class="result-icon">✅</div>
                <h2 class="result-title">LEGITIMATE</h2>
                <p class="result-description">This message appears safe</p>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning('⚠️ Please enter a message')

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('''
<div class="footer">
    <p>AI-Powered Spam Detection</p>
</div>
''', unsafe_allow_html=True)