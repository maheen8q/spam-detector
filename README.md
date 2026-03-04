#  SMS Spam Detector

A machine learning model that classifies SMS messages as spam or not spam, deployed as an interactive web app using Streamlit.

---

##  Dataset

- **Source:** [SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,572 labeled SMS messages (ham / spam)

---

##  Pipeline

### 1. Preprocessing
- Lowercasing, punctuation and stopword removal
- Tokenization using `nltk`
- Stemming with `PorterStemmer`

### 2. Feature Extraction
- Text vectorized using **TF-IDF** (`TfidfVectorizer`)

### 3. Model
- **Random Forest Classifier**
  - `n_estimators = 200`
  - `random_state = 2`
  - Tuned splitting criteria for better generalization

---

##  Results

| Metric | Score |
|---|---|
| Accuracy | ~97% |
| Precision | 100% |
| False Positives | 0 |

**Confusion Matrix:**

|  | Predicted: Not Spam | Predicted: Spam |
|---|---|---|
| **Actually Not Spam** | 896 | 0 |
| **Actually Spam** | 29 | 109 |

>  Test set performance. Real-world results may vary due to class imbalance and evolving spam patterns.

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `nltk` | NLP preprocessing |
| `scikit-learn` | Modeling & evaluation |
| `matplotlib`, `seaborn` | EDA & visualization |
| `streamlit` | Web app deployment |
| `pickle` | Model serialization |

---

##  Run Locally

```bash
git clone https://github.com/maheen8q/spam-detector.git
cd spam-detector
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
spam-detector/
│
├── spam-detection.ipynb   # EDA, preprocessing, modeling
├── app.py                 # Streamlit app
├── model.pkl              # Trained Random Forest model
├── vectorizer.pkl         # Fitted TF-IDF vectorizer
└── README.md
```

---

## 🔗 Links
https://spam-detector-yvxs9mmkn6sumxug2chaqg.streamlit.app/

- 🎯 **Live Demo:** [link]
- 📓 **Notebook:** [link]
