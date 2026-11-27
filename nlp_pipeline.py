"""
nlp_pipeline.py
================
Modul inti yang meng-encapsulate SELURUH logika dari notebook
`Skripsi_Analisis_Sentimen_Cashless_IndoBERT.ipynb` (versi terbaru) agar bisa
dipakai di aplikasi Streamlit TANPA mengubah satu pun parameter/algoritma.

Urutan pipeline (identik notebook):
    Cleaning - Case Folding - Normalized (Slangword, kamus alay dari GitHub)
    - Tokenizing - Stopwords Removal (pada token) - Stemming
    - Labeling (IndoBERT)
    - Split (80/20, stratified) 
    - Naive Bayes (GridSearchCV) & SVM (GridSearchCV) - Evaluation

"""

import re
import string

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

ORDER = ['Positif', 'Netral', 'Negatif']
PRETRAINED_MODEL = 'mdhugol/indonesia-bert-sentiment-classification'
LABEL_MAP_BERT = {'LABEL_0': 'Positif', 'LABEL_1': 'Netral', 'LABEL_2': 'Negatif'}

# Kamus alay (slangword) -- identik sel 3.3 notebook. Tidak lagi memakai
# indoNLP.preprocessing.replace_slang; digantikan lookup dict dari kamus ini.
SLANG_DICT_URL = ('https://raw.githubusercontent.com/okkyibrohim/'
                   'id-multi-label-hate-speech-and-abusive-language-detection/'
                   'master/new_kamusalay.csv')

NEGATION_WORDS = {'tidak', 'enggak', 'nggak', 'gak', 'ga', 'tak', 'bukan',
                   'jangan', 'belum', 'kaga', 'kagak'}
EXTRA_STOPWORDS = {
    'yg', 'nya', 'nih', 'dong', 'deh', 'kok', 'sih', 'kak', 'min', 'gan', 'ya',
    'aja', 'saja', 'amp', 'via'
}

# Grid parameter -- identik sel 8.1 & 8.2 pada notebook
NB_PARAM_GRID = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
SVM_PARAM_GRID = {'C': [0.1, 1, 10]}


def ensure_nltk_resources():
    """Download resource NLTK yang dibutuhkan (idempotent, sama seperti notebook)."""
    for res in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


def build_stopword_set():
    """Sama persis dengan sel 3.5 di notebook: stopword NLTK Indonesia
    + daftar tambahan, dikurangi kata negasi (dipertahankan untuk konteks)."""
    default_stopwords = set(stopwords.words('indonesian'))
    all_stopwords = (default_stopwords | EXTRA_STOPWORDS) - NEGATION_WORDS
    return all_stopwords


# 3. Text Preprocessing (identik sel 3.1 - 3.6)
# Urutan: Cleaning - Case Folding - Normalized - Tokenizing
#         - Stopwords Removal (pada token) - Stemming


def cleaning(text: str) -> str:
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'RT[\s]+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def case_folding(text: str) -> str:
    return text.lower()


def load_slang_dict() -> dict:
    """Unduh kamus alay dari GitHub & bangun dict slang->formal.
    Identik sel 3.3 notebook. Panggil lewat st.cache_resource di app.py
    supaya hanya diunduh sekali."""
    kamus_alay = pd.read_csv(SLANG_DICT_URL, encoding='ISO-8859-1', header=None,
                              names=['slang', 'formal'])
    return dict(zip(kamus_alay['slang'].str.lower(), kamus_alay['formal'].str.lower()))


def replace_slang(text: str, slang_dict: dict) -> str:
    """Normalisasi slangword berbasis lookup kamus alay -- identik sel 3.3 notebook."""
    if not isinstance(text, str):
        return text
    return ' '.join(slang_dict.get(w, w) for w in text.split())


def tokenizing(text: str):
    return word_tokenize(text)


def remove_stopwords(tokens, stopword_set):
    """Beroperasi pada LIST token (identik sel 3.5 notebook -- bukan pada string)."""
    return [w for w in tokens if w not in stopword_set]


def stem_tokens(tokens, stemmer, stem_cache: dict):
    return [stem_cache.setdefault(t, stemmer.stem(t)) for t in tokens]


def preprocess_dataframe(df: pd.DataFrame, slang_dict: dict, text_col: str = 'text',
                          progress_callback=None) -> pd.DataFrame:
    """Menjalankan seluruh tahap 3.1-3.6 pada dataframe dengan urutan IDENTIK
    notebook: cleaning, case_folding, normalized (kamus alay), tokenizing,
    stopwords removal (pada token), stemming -> text_final."""
    ensure_nltk_resources()

    df_clean = df.drop_duplicates(subset=text_col).reset_index(drop=True)
    df_clean = df_clean[[text_col]].rename(columns={text_col: 'text'})
    df_clean = df_clean.dropna(subset=['text']).reset_index(drop=True)

    if progress_callback:
        progress_callback(0.1, 'Cleaning...')
    df_clean['text_clean'] = df_clean['text'].astype(str).apply(cleaning)

    if progress_callback:
        progress_callback(0.2, 'Case folding...')
    df_clean['text_casefold'] = df_clean['text_clean'].apply(case_folding)

    if progress_callback:
        progress_callback(0.35, 'Normalisasi slangword (kamus alay)...')
    df_clean['text_normalized'] = df_clean['text_casefold'].apply(
        lambda t: replace_slang(t, slang_dict))

    if progress_callback:
        progress_callback(0.5, 'Tokenizing...')
    df_clean['text_tokens'] = df_clean['text_normalized'].apply(tokenizing)

    if progress_callback:
        progress_callback(0.65, 'Stopwords removal (pada token)...')
    stopword_set = build_stopword_set()
    df_clean['text_stopword_tokens'] = df_clean['text_tokens'].apply(
        lambda toks: remove_stopwords(toks, stopword_set))

    if progress_callback:
        progress_callback(0.8, 'Stemming (Sastrawi)...')
    stemmer = StemmerFactory().create_stemmer()
    stem_cache = {}
    df_clean['text_stemmed_tokens'] = df_clean['text_stopword_tokens'].apply(
        lambda toks: stem_tokens(toks, stemmer, stem_cache))
    df_clean['text_final'] = df_clean['text_stemmed_tokens'].apply(' '.join)

    df_clean = df_clean[df_clean['text_final'].str.strip() != ''].reset_index(drop=True)

    if progress_callback:
        progress_callback(1.0, 'Preprocessing selesai.')

    return df_clean



# 4. Labeling Data dengan IndoBERT (identik sel 4)


def load_sentiment_pipeline():
    """Load model IndoBERT (mdhugol/indonesia-bert-sentiment-classification).
    Fungsi ini idempotent -- panggil lewat st.cache_resource di app.py."""
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline(
        'sentiment-analysis',
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=128,
        device=device,
    )
    return sentiment_pipeline


def label_with_indobert(texts, sentiment_pipeline, batch_size=32, progress_callback=None):
    """Sama seperti fungsi label_with_indobert di notebook, dengan tambahan
    progress callback per-batch supaya UI Streamlit bisa menampilkan progress bar."""
    texts = list(texts)
    labels, scores = [], []
    n = len(texts)
    for start in range(0, n, batch_size):
        chunk = texts[start:start + batch_size]
        results = sentiment_pipeline(chunk, batch_size=batch_size)
        labels.extend(LABEL_MAP_BERT[r['label']] for r in results)
        scores.extend(r['score'] for r in results)
        if progress_callback:
            progress_callback(min(1.0, (start + len(chunk)) / n))
    return labels, scores


def label_sentence_bert(text, sentiment_pipeline, slang_dict):
    """Diagnostik satu kalimat -- identik sel 4.2."""
    text_prepro = replace_slang(case_folding(cleaning(text)), slang_dict)
    result = sentiment_pipeline(text_prepro)[0]
    label = LABEL_MAP_BERT[result['label']]
    score = result['score']
    return {'teks_asli': text, 'teks_prepro': text_prepro, 'label_bert': label,
            'confidence': round(score, 4)}



# 5-7. Split, TF-IDF, Modeling (identik sel 33-39, TANPA resampling)


def split_dataset(df_clean: pd.DataFrame, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df_clean[['text', 'text_final', 'polarity']],
        test_size=test_size, random_state=random_state,
        stratify=df_clean['polarity'],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit TF-IDF langsung pada train_df ASLI (tanpa resampling) -- identik sel 6 notebook."""
    tfidf_vectorizer = TfidfVectorizer(max_features=8000, min_df=2, max_df=0.9,
                                        ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text_final'])
    X_test_tfidf = tfidf_vectorizer.transform(test_df['text_final'])
    return tfidf_vectorizer, X_train_tfidf, X_test_tfidf


def train_naive_bayes(X_train_tfidf, y_train):
    """GridSearchCV atas alpha -- identik sel 8.1 notebook."""
    nb_grid = GridSearchCV(MultinomialNB(), NB_PARAM_GRID,
                            cv=5, scoring='f1_weighted', n_jobs=-1)
    nb_grid.fit(X_train_tfidf, y_train)
    return nb_grid.best_estimator_, nb_grid.best_params_


def train_svm(X_train_tfidf, y_train, random_state=42):
    """GridSearchCV atas C, kernel default (RBF) -- identik sel 8.2 notebook."""
    svm_grid = GridSearchCV(SVC(random_state=random_state), SVM_PARAM_GRID,
                             cv=5, scoring='f1_weighted', n_jobs=-1)
    svm_grid.fit(X_train_tfidf, y_train)
    return svm_grid.best_estimator_, svm_grid.best_params_



# 9. Evaluation (identik sel 46)


def evaluate_model(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report_text = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=ORDER)

    header = (
        f"=== Evaluasi Model: {model_name} ===\n"
        f"Accuracy  : {acc:.4f}\n"
        f"Precision : {prec:.4f}\n"
        f"Recall    : {rec:.4f}\n"
        f"F1-Score  : {f1:.4f}\n"
    )
    full_text = header + "\n" + report_text

    return {
        'Model': model_name, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1-Score': f1, 'report_text': full_text,
        'confusion_matrix': cm,
    }



# 11. Prediksi Kalimat Baru (identik sel 54 -- urutan Tokenizing -> Stopwords -> Stemming)


def predict_sentence(text, tfidf_vectorizer, nb_model, svm_model,
                      sentiment_pipeline, stopword_set, stemmer, stem_cache, slang_dict):
    # 1. Preprocessing Lengkap -- identik sel 10 notebook
    text_clean = cleaning(text)
    text_casefold = case_folding(text_clean)
    text_normalized = replace_slang(text_casefold, slang_dict)

    # Urutan: Tokenizing -> Stopwords Removal -> Stemming
    tokens = tokenizing(text_normalized)
    tokens_no_stopwords = remove_stopwords(tokens, stopword_set)
    tokens_stemmed = stem_tokens(tokens_no_stopwords, stemmer, stem_cache)
    text_final = ' '.join(tokens_stemmed)

    # 2. IndoBERT dipanggil menggunakan text_normalized (konteks bahasa utuh)
    bert_result = sentiment_pipeline(text_normalized)[0]
    label_bert = LABEL_MAP_BERT[bert_result['label']]
    conf_bert = bert_result['score']

    # 3. TF-IDF & Prediksi Machine Learning (NB & SVM) menggunakan text_final
    text_tfidf = tfidf_vectorizer.transform([text_final])
    pred_nb = nb_model.predict(text_tfidf)[0]
    pred_svm = svm_model.predict(text_tfidf)[0]

    return {
        'teks_asli': text,
        'teks_normalized': text_normalized,
        'teks_final': text_final,
        'label_indobert': label_bert,
        'confidence_bert': round(conf_bert, 4),
        'prediksi_naive_bayes': pred_nb,
        'prediksi_svm': pred_svm,
    }
