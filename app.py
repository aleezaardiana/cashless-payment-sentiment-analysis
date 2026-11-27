"""
app.py
======
Aplikasi Streamlit untuk Analisis Sentimen Opini Masyarakat terhadap
Sistem Pembayaran Cashless, menggunakan pipeline & model yang 100% identik
dengan notebook penelitian (CRISP-DM, pelabelan IndoBERT, TF-IDF, Naive Bayes & SVM).

Jalankan dengan:
    streamlit run app.py
"""

import io
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import streamlit as st

import nlp_pipeline as pl


# Konfigurasi Halaman


st.set_page_config(
    page_title="Analisis Sentimen Cashless | IndoBERT + NB + SVM",
    layout="wide",
)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

COLOR_MAP = {'Positif': '#4CAF50', 'Netral': '#9E9E9E', 'Negatif': '#F44336'}
EMOJI_MAP = {'Positif': '😊', 'Netral': '😐', 'Negatif': '😞'}

CONTOH_KALIMAT = {
    "Positif": [
        "Bayar pake cash enak, gaharus antri lama",
        "Cashless bikin transaksi jadi lebih cepat",
        "Saya lebih suka bayar pakai QRIS karena praktis",
        "Pembayaran non tunai sangat membantu saat tidak membawa dompet.",
        "Cashless mempermudah pembayaran sehari-hari",
        "Bayar pakai e-wallet lebih nyaman",
        "Cashless aman dan efisien",
        "Sistem cashless ok aja",
    ],
    "Negatif": [
        "Cashless ribet",
        "Saya tidak suka sistem cashless",
        "Cashless bikin repot orang tua",
        "Aplikasi pembayaran sering error",
        "QRIS susah dipakai",
        "Dompet digital sering gagal saat transaksi",
        "Internet jelek jadi cashless tidak bisa dipakai",
        "gua gasetuju kalo ada toko ga nerima uang cash",
        "fnb harus kena undang-undang kalau ga nerima uang cash",
        "uang tunai masih jadi pembayaran yang sah, masa gaditerima sih",
    ],
    "Netral": [
        "Saya membayar menggunakan uang tunai",
        "Hari ini saya memakai QRIS",
        "Di toko ini tersedia pembayaran cash dan cashless",
        "Saya membawa uang tunai dan kartu debit",
        "Metode pembayaran bisa menggunakan cash maupun QRIS",
    ],
}

# Dataset contoh bawaan
SAMPLE_DF = pd.DataFrame({
    'text': (CONTOH_KALIMAT["Positif"] + CONTOH_KALIMAT["Negatif"] + CONTOH_KALIMAT["Netral"])
})



# Resource Caching


@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    return pl.load_sentiment_pipeline()


@st.cache_resource(show_spinner=False)
def get_slang_dict():
    return pl.load_slang_dict()


@st.cache_resource(show_spinner=False)
def get_stopword_set():
    pl.ensure_nltk_resources()
    return pl.build_stopword_set()


@st.cache_resource(show_spinner=False)
def get_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


def init_state():
    defaults = {
        'df_clean': None,
        'df_labeled': None,
        'train_df': None,
        'test_df': None,
        'tfidf_vectorizer': None,
        'nb_model': None,
        'svm_model': None,
        'nb_best_params': None,
        'svm_best_params': None,
        'y_pred_nb': None,
        'y_pred_svm': None,
        'results_nb': None,
        'results_svm': None,
        'stem_cache': {},
        'dataset_name': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# Sidebar Navigasi


st.sidebar.title("Sentimen Cashless")
st.sidebar.caption("IndoBERT Labeling + Naive Bayes + SVM")
page = st.sidebar.radio(
    "Navigasi",
    ["About System", "Upload Dataset", "Statistik & Evaluasi Model",
     "Visualisasi (EDA)", "Try It Now"],
)

st.sidebar.markdown("---")
if st.session_state['df_labeled'] is not None:
    st.sidebar.success(f"Dataset aktif: {st.session_state['dataset_name']}\n\n"
                       f"{len(st.session_state['df_labeled'])} baris (sudah dilabeli)")
else:
    st.sidebar.info("Belum ada dataset yang diproses.")

if st.session_state['nb_model'] is not None:
    st.sidebar.success("Model NB & SVM: sudah dilatih")
else:
    st.sidebar.warning("Model NB & SVM: belum dilatih")



# PAGE 1: ABOUT SYSTEM


if page == "About System":
    st.title("Analisis Sentimen Opini Masyarakat terhadap Sistem Pembayaran Cashless")
    st.subheader("Studi Kasus: Media Sosial X (Twitter)")

    st.markdown("""
Aplikasi ini mendemonstrasikan sistem klasifikasi sentimen berbasis machine learning
yang dibangun mengikuti metodologi **CRISP-DM** (*Cross Industry Standard Process for
Data Mining*), dengan pipeline sebagai berikut:
""")

    st.info(
        "**Alur Sistem:**\n\n"
        "Upload Dataset → Preprocessing (Cleaning → Case Folding → Normalisasi Slang → "
        "Tokenizing → Stopwords Removal → Stemming) → Pelabelan Otomatis (IndoBERT) → "
        "Split Data (80% Latih / 20% Uji, stratified) "
        "→ Ekstraksi Fitur TF-IDF → Modeling (Naive Bayes & SVM) → Evaluasi → Prediksi Interaktif"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Tujuan")
        st.markdown(
            "- Mengklasifikasikan opini masyarakat terhadap sistem pembayaran "
            "*cashless* ke dalam 3 kelas: **Positif, Netral, Negatif**.\n"
            "- Membandingkan performa algoritma **Naive Bayes** dan **Support Vector "
            "Machine (SVM)** dalam tugas klasifikasi sentimen."
        )
    with col2:
        st.markdown("#### Pelabelan Data")
        st.markdown(
            "Label dihasilkan otomatis menggunakan model IndoBERT yang sudah "
            "*fine-tuned* untuk sentimen Bahasa Indonesia:\n\n"
            "`mdhugol/indonesia-bert-sentiment-classification`\n\n"
            "Label ini bersifat *pseudo-label* (bukan anotasi manusia), berfungsi "
            "sebagai target latih untuk NB & SVM."
        )
    with col3:
        st.markdown("#### Model")
        st.markdown(
            "- **Naive Bayes** (`MultinomialNB`) — parameter `alpha` dituning "
            "dengan `GridSearchCV`.\n"
            "- **SVM** (`SVC`) — parameter `C` dituning dengan `GridSearchCV` "
            "(kernel default RBF).\n"
            "- Fitur teks direpresentasikan dengan **TF-IDF** "
            "(unigram + bigram, max 8000 fitur, `sublinear_tf=True`)."
        )

    st.markdown("---")
    st.markdown("#### Detail Teknis Preprocessing & Modeling")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("""
**Text Preprocessing:**
1. **Cleaning** — hapus mention, hashtag, RT, URL, tag HTML, angka, tanda baca
2. **Case Folding** — ubah ke huruf kecil
3. **Normalisasi Slang** — kamus alay ([`id-multi-label-hate-speech-and-abusive-
   language-detection`](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection),
   di-*lookup* per kata)
4. **Tokenizing** — `nltk.word_tokenize`
5. **Stopwords Removal** — dilakukan pada TOKEN hasil tokenizing, `NLTK`
   (Bahasa Indonesia) + daftar tambahan, **kata negasi dipertahankan**
   (tidak, bukan, jangan, gak, dll.)
6. **Stemming** — `Sastrawi` (dengan cache)
""")
    with tcol2:
        st.markdown("""
**Modeling:**
- Split data **80% latih / 20% uji**, *stratified*, `random_state=42`
- **Tanpa resampling** — proporsi kelas asli dipertahankan pada data latih
- **TF-IDF**: `max_features=8000, min_df=2, max_df=0.9, ngram_range=(1,2), sublinear_tf=True`
- **Naive Bayes**: `GridSearchCV(MultinomialNB(), {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}, cv=5, scoring='f1_weighted')`
- **SVM**: `GridSearchCV(SVC(random_state=42), {'C': [0.1, 1, 10]}, cv=5, scoring='f1_weighted')`
- Evaluasi: Accuracy, Precision, Recall, F1-Score (*weighted*), Confusion Matrix
""")



# PAGE 2: UPLOAD DATASET & PREPROCESSING + LABELING


elif page == "Upload Dataset":
    st.title("Upload Dataset")
    st.markdown(
        "Unggah dataset hasil crawling (format `.csv`) yang memiliki kolom teks "
        "komentar/tweet. Atau gunakan dataset contoh bawaan untuk mencoba sistem."
    )

    src_option = st.radio("Sumber data:", ["Upload file CSV", "Gunakan dataset contoh bawaan"],
                          horizontal=True)

    raw_df = None
    text_col = 'text'

    if src_option == "Upload file CSV":
        uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.session_state['dataset_name'] = uploaded_file.name
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

        if raw_df is not None:
            st.success(f"Dataset terbaca: {raw_df.shape[0]} baris x {raw_df.shape[1]} kolom")
            if 'text' not in raw_df.columns:
                text_col = st.selectbox(
                    "Kolom 'text' tidak ditemukan. Pilih kolom yang berisi teks komentar:",
                    options=list(raw_df.columns),
                )
            else:
                text_col = 'text'
    else:
        raw_df = SAMPLE_DF.copy()
        st.session_state['dataset_name'] = "Dataset contoh (built-in)"
        st.success(f"Menggunakan dataset contoh bawaan: {raw_df.shape[0]} baris")

    if raw_df is not None:
        with st.expander("Lihat raw data yang diunggah"):
            st.dataframe(raw_df.head(20), use_container_width=True)

        st.markdown("#### Batch size pelabelan IndoBERT")
        batch_size = st.slider(
            "Batch size",
            min_value=8, max_value=64, value=32, step=8,
        )

        if st.button("Jalankan Preprocessing & Pelabelan IndoBERT", type="primary"):
            progress_bar = st.progress(0.0, text="Memulai preprocessing...")

            def update_progress(frac, msg=""):
                progress_bar.progress(frac, text=msg or "Memproses...")

            with st.spinner("Memuat kamus alay (slangword)..."):
                slang_dict = get_slang_dict()

            with st.spinner("Menjalankan tahap cleaning s/d stemming..."):
                df_clean = pl.preprocess_dataframe(raw_df, slang_dict, text_col=text_col,
                                                   progress_callback=update_progress)
            st.session_state['df_clean'] = df_clean

            with st.spinner("Memuat model IndoBERT..."):
                sentiment_pipeline = get_sentiment_pipeline()

            label_progress = st.progress(0.0, text="Melabeli data dengan IndoBERT...")

            def update_label_progress(frac):
                label_progress.progress(frac, text=f"Melabeli data... ({frac*100:.0f}%)")

            labels, scores = pl.label_with_indobert(
                df_clean['text_normalized'], sentiment_pipeline,
                batch_size=batch_size, progress_callback=update_label_progress,
            )
            df_clean['polarity'] = labels
            df_clean['polarity_confidence'] = scores
            st.session_state['df_labeled'] = df_clean

            # Reset state model
            for k in ['train_df', 'test_df', 'tfidf_vectorizer',
                      'nb_model', 'svm_model', 'nb_best_params', 'svm_best_params',
                      'y_pred_nb', 'y_pred_svm', 'results_nb', 'results_svm']:
                st.session_state[k] = None

            st.success("Preprocessing & pelabelan selesai!")
            st.rerun()

    st.markdown("---")

    if st.session_state['df_labeled'] is not None:
        df_labeled = st.session_state['df_labeled']

        st.markdown(f"### Preview Dataset ({len(df_labeled)} baris setelah preprocessing)")

        preview_cols = ['text', 'text_normalized', 'text_final', 'polarity', 'polarity_confidence']
        preview_cols = [c for c in preview_cols if c in df_labeled.columns]

        tab_top, tab_bottom, tab_full = st.tabs(
            ["10 Data Teratas", "10 Data Terbawah", "Semua Data"])

        with tab_top:
            st.dataframe(df_labeled[preview_cols].head(10), use_container_width=True)
        with tab_bottom:
            st.dataframe(df_labeled[preview_cols].tail(10), use_container_width=True)
        with tab_full:
            event = st.dataframe(
                df_labeled[preview_cols], use_container_width=True, height=350,
                on_select="rerun", selection_mode="single-row", key="full_data_table",
            )
            selected_rows = event.selection.rows if hasattr(event, "selection") else []
            if selected_rows:
                idx = selected_rows[0]
                row = df_labeled.iloc[idx]
                st.markdown("##### Detail Baris Terpilih")
                d1, d2 = st.columns(2)
                with d1:
                    st.write("**Teks Asli:**", row['text'])
                    st.write("**Teks Normalisasi (IndoBERT):**", row.get('text_normalized', '-'))
                    st.write("**Teks Final (TF-IDF):**", row.get('text_final', '-'))
                with d2:
                    st.write("**Label (IndoBERT):**", row.get('polarity', '-'))
                    st.write("**Confidence:**", round(row.get('polarity_confidence', 0), 4))

        st.markdown("### Ringkasan Jumlah Data per Label")
        counts = df_labeled['polarity'].value_counts().reindex(pl.ORDER).fillna(0).astype(int)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Data", len(df_labeled))
        m2.metric("Positif", int(counts.get('Positif', 0)))
        m3.metric("Netral", int(counts.get('Netral', 0)))
        m4.metric("Negatif", int(counts.get('Negatif', 0)))

        csv_buffer = io.StringIO()
        df_labeled.to_csv(csv_buffer, index=False)
        st.download_button("Unduh Dataset Berlabel (CSV)", csv_buffer.getvalue(),
                           file_name="dataset_labeled.csv", mime="text/csv")



# PAGE 3: STATISTIK & EVALUASI MODEL


elif page == "Statistik & Evaluasi Model":
    st.title("Statistik Dataset & Evaluasi Model")

    if st.session_state['df_labeled'] is None:
        st.warning("Silakan upload dan proses dataset terlebih dahulu di halaman **Upload Dataset**.")
        st.stop()

    df_labeled = st.session_state['df_labeled']

    st.markdown("### Statistik Umum Dataset")
    counts = df_labeled['polarity'].value_counts().reindex(pl.ORDER).fillna(0).astype(int)
    pct = (counts / counts.sum() * 100).round(2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", len(df_labeled))
    c2.metric("Positif", int(counts['Positif']), f"{pct['Positif']}%")
    c3.metric("Netral", int(counts['Netral']), f"{pct['Netral']}%")
    c4.metric("Negatif", int(counts['Negatif']), f"{pct['Negatif']}%")

    st.markdown("---")
    st.markdown("### Pelatihan Model")

    if st.button("Latih & Evaluasi Model (Naive Bayes & SVM)", type="primary"):
        with st.spinner("Split dataset 80/20 (stratified)..."):
            train_df, test_df = pl.split_dataset(df_labeled)
            st.session_state['train_df'] = train_df
            st.session_state['test_df'] = test_df

        with st.spinner("Ekstraksi fitur TF-IDF (fit langsung pada data latih, tanpa resampling)..."):
            tfidf_vectorizer, X_train_tfidf, X_test_tfidf = pl.build_tfidf(
                train_df, test_df)
            st.session_state['tfidf_vectorizer'] = tfidf_vectorizer

        y_train = train_df['polarity']
        y_test = test_df['polarity']

        with st.spinner("Melatih Naive Bayes (GridSearchCV atas alpha, cv=5)..."):
            nb_model, nb_best_params = pl.train_naive_bayes(X_train_tfidf, y_train)
            y_pred_nb = nb_model.predict(X_test_tfidf)
            st.session_state['nb_model'] = nb_model
            st.session_state['nb_best_params'] = nb_best_params
            st.session_state['y_pred_nb'] = y_pred_nb

        with st.spinner("Melatih SVM (GridSearchCV atas C, cv=5)..."):
            svm_model, svm_best_params = pl.train_svm(X_train_tfidf, y_train)
            y_pred_svm = svm_model.predict(X_test_tfidf)
            st.session_state['svm_model'] = svm_model
            st.session_state['svm_best_params'] = svm_best_params
            st.session_state['y_pred_svm'] = y_pred_svm

        st.session_state['results_nb'] = pl.evaluate_model('Naive Bayes', y_test, y_pred_nb)
        st.session_state['results_svm'] = pl.evaluate_model('SVM', y_test, y_pred_svm)

        st.success("Model berhasil dilatih & dievaluasi!")
        st.rerun()

    if st.session_state['results_nb'] is not None:
        train_df = st.session_state['train_df']
        test_df = st.session_state['test_df']

        st.markdown("---")
        st.markdown("### Ringkasan Split Data")
        s1, s2 = st.columns(2)
        s1.metric("Data Latih (80%)", len(train_df))
        s2.metric("Data Uji (20%)", len(test_df))

        st.markdown("### Hasil Evaluasi Model (Data Uji)")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("#### Naive Bayes")
            st.caption(f"Parameter terbaik (GridSearchCV): {st.session_state['nb_best_params']}")
            st.code(st.session_state['results_nb']['report_text'], language=None)
        with e2:
            st.markdown("#### SVM")
            st.caption(f"Parameter terbaik (GridSearchCV): {st.session_state['svm_best_params']}")
            st.code(st.session_state['results_svm']['report_text'], language=None)

        st.markdown("### Perbandingan Performa Model")
        comparison_df = pd.DataFrame([
            {k: v for k, v in st.session_state['results_nb'].items()
             if k in ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']},
            {k: v for k, v in st.session_state['results_svm'].items()
             if k in ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']},
        ]).set_index('Model')
        st.dataframe(comparison_df.round(4), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        comparison_df.plot(kind='bar', colormap='viridis', ax=ax)
        ax.set_title('Perbandingan Performa Naive Bayes vs SVM (Data Uji)')
        ax.set_ylabel('Skor')
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(comparison_df.index, rotation=0)
        ax.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Confusion Matrix")
        cm1, cm2 = st.columns(2)
        for col, res in zip([cm1, cm2], [st.session_state['results_nb'], st.session_state['results_svm']]):
            with col:
                st.markdown(f"**{res['Model']}**")
                fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
                sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                            xticklabels=pl.ORDER, yticklabels=pl.ORDER, ax=ax_cm)
                ax_cm.set_xlabel('Prediksi')
                ax_cm.set_ylabel('Aktual')
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)



# PAGE 4: VISUALISASI (EDA)


elif page == "Visualisasi (EDA)":
    st.title("Visualisasi & Exploratory Data Analysis")

    if st.session_state['df_labeled'] is None:
        st.warning("Silakan upload dan proses dataset terlebih dahulu di halaman **Upload Dataset**.")
        st.stop()

    df_labeled = st.session_state['df_labeled']

    st.markdown("### Proporsi Sentimen Dataset")
    counts = df_labeled['polarity'].value_counts().reindex(pl.ORDER).fillna(0)
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=[COLOR_MAP[c] for c in counts.index], startangle=90)
    ax1.set_title('Proporsi Sentimen Opini Masyarakat terhadap Cashless Payment\n'
                  'Label: IndoBERT')
    plt.tight_layout()
    
    col_pie, col_bar = st.columns(2)
    with col_pie:
        st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bars = sns.countplot(data=df_labeled, x='polarity', order=pl.ORDER,
                          palette=COLOR_MAP, ax=ax2)
    ax2.set_title('Distribusi Sentimen Sebelum Split (Label: IndoBERT)')
    ax2.set_xlabel('Sentimen')
    ax2.set_ylabel('Jumlah Data')
    for p in bars.patches:
        ax2.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha='center', va='bottom')
    plt.tight_layout()
    with col_bar:
        st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("---")
    st.markdown("### WordCloud per Kategori Sentimen (Data Latih)")

    if st.session_state.get('train_df') is not None:
        train_df = st.session_state['train_df']
        wc_cols = st.columns(3)
        cmap_by_label = {'Positif': 'Greens', 'Netral': 'Greys', 'Negatif': 'Reds'}
        for col, (label, cmap) in zip(wc_cols, cmap_by_label.items()):
            with col:
                st.markdown(f"**{label}**")
                text_data = ' '.join(
                    train_df.loc[train_df['polarity'] == label, 'text_final'].dropna()
                )
                if len(text_data.strip()) == 0:
                    st.caption(f"Tidak ada data untuk kategori {label}")
                    continue
                wc = WordCloud(width=500, height=350, background_color='white',
                                colormap=cmap, max_words=100).generate(text_data)
                fig_wc, ax_wc = plt.subplots(figsize=(5, 3.5))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                plt.tight_layout()
                st.pyplot(fig_wc)
                plt.close(fig_wc)
    else:
        st.info("Latih model terlebih dahulu di halaman **Statistik & Evaluasi Model** untuk melihat WordCloud.")

    st.markdown("---")
    st.markdown("### Rincian Hasil Prediksi pada Data Uji")
    if st.session_state.get('y_pred_svm') is not None:
        test_df = st.session_state['test_df'].reset_index(drop=True)
        y_pred_nb = st.session_state['y_pred_nb']
        y_pred_svm = st.session_state['y_pred_svm']

        # PERBAIKAN PENTING: reset_index dipastikan sejajar untuk semua kolom
        prediction_table = pd.DataFrame({
            'teks_asli': test_df['text'],
            'teks_prepro': test_df['text_final'],
            'label_bert': test_df['polarity'],
            'prediksi_naive_bayes': y_pred_nb,
            'prediksi_svm': y_pred_svm,
        })

        st.dataframe(prediction_table, use_container_width=True, height=300)

        # Download CSV hasil prediksi data uji
        csv_pred = io.StringIO()
        prediction_table.to_csv(csv_pred, index=False)
        st.download_button("Unduh Hasil Prediksi Data Uji (CSV)", csv_pred.getvalue(),
                           file_name="hasil_prediksi_data_uji.csv", mime="text/csv")

        st.markdown("#### Rincian Hasil Prediksi Berdasarkan Kelas (Model SVM)")
        prediksi_positif = prediction_table[prediction_table['prediksi_svm'] == 'Positif'].reset_index(drop=True)
        prediksi_negatif = prediction_table[prediction_table['prediksi_svm'] == 'Negatif'].reset_index(drop=True)
        prediksi_netral = prediction_table[prediction_table['prediksi_svm'] == 'Netral'].reset_index(drop=True)

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Diprediksi Positif", f"{len(prediksi_positif)} / {len(prediction_table)}")
        rc2.metric("Diprediksi Netral", f"{len(prediksi_netral)} / {len(prediction_table)}")
        rc3.metric("Diprediksi Negatif", f"{len(prediksi_negatif)} / {len(prediction_table)}")

        dl1, dl2, dl3 = st.columns(3)
        for col, label_df, fname in zip(
            [dl1, dl2, dl3],
            [prediksi_positif, prediksi_netral, prediksi_negatif],
            ["prediksi_positif.csv", "prediksi_netral.csv", "prediksi_negatif.csv"],
        ):
            buf = io.StringIO()
            label_df.to_csv(buf, index=False)
            col.download_button(f"Unduh {fname}", buf.getvalue(), file_name=fname, mime="text/csv")



# PAGE 5: TRY IT NOW


elif page == "Try It Now":
    st.title("Coba Analisis Sentimen (Prediksi Real-time)")
    st.markdown("Masukkan kalimat/opini untuk dianalisis dan dibandingkan prediksinya secara langsung.")

    if st.session_state.get('tfidf_vectorizer') is None or st.session_state.get('nb_model') is None:
        st.warning("Model belum dilatih. Silakan upload dataset & latih model terlebih dahulu di halaman **Upload Dataset** dan **Statistik & Evaluasi Model**.")
        st.stop()

    if 'try_it_text' not in st.session_state:
        st.session_state['try_it_text'] = ""

    text_input = st.text_area(
        "Masukkan Komentar", value=st.session_state['try_it_text'],
        height=100, placeholder="Contoh: Pembayaran dengan QRIS sangat praktis dan tidak perlu kembalian.",
    )

    st.markdown("**Atau coba kalimat contoh:**")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("🟢 Contoh Positif"):
        st.session_state['try_it_text'] = np.random.choice(CONTOH_KALIMAT["Positif"])
        st.rerun()
    if ex2.button("⚪ Contoh Netral"):
        st.session_state['try_it_text'] = np.random.choice(CONTOH_KALIMAT["Netral"])
        st.rerun()
    if ex3.button("🔴 Contoh Negatif"):
        st.session_state['try_it_text'] = np.random.choice(CONTOH_KALIMAT["Negatif"])
        st.rerun()

    if st.button("Analisis Sentimen →", type="primary", use_container_width=True):
        if not text_input.strip():
            st.error("Masukkan komentar terlebih dahulu.")
        else:
            sentiment_pipeline = get_sentiment_pipeline()
            slang_dict = get_slang_dict()
            stopword_set = get_stopword_set()
            stemmer = get_stemmer()

            with st.spinner("Memproses text & memprediksi..."):
                hasil = pl.predict_sentence(
                    text_input, st.session_state['tfidf_vectorizer'],
                    st.session_state['nb_model'], st.session_state['svm_model'],
                    sentiment_pipeline, stopword_set, stemmer,
                    st.session_state['stem_cache'], slang_dict,
                )

            st.markdown("---")
            st.markdown("### Hasil Prediksi Model ML")

            # Ditampilkan BERDAMPINGAN seperti di Notebook
            col_nb, col_svm = st.columns(2)
            with col_nb:
                pred_nb = hasil['prediksi_naive_bayes']
                st.markdown(
                    f"<div style='text-align:center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>"
                    f"<h4>Naive Bayes</h4>"
                    f"<div style='font-size: 36px;'>{EMOJI_MAP[pred_nb]}</div>"
                    f"<div style='font-size: 22px; font-weight: bold;'>{pred_nb}</div>"
                    f"</div>", unsafe_allow_html=True,
                )

            with col_svm:
                pred_svm = hasil['prediksi_svm']
                st.markdown(
                    f"<div style='text-align:center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>"
                    f"<h4>SVM (Support Vector Machine)</h4>"
                    f"<div style='font-size: 36px;'>{EMOJI_MAP[pred_svm]}</div>"
                    f"<div style='font-size: 22px; font-weight: bold;'>{pred_svm}</div>"
                    f"</div>", unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("### Detail Pemrosesan Teks")
            st.write("**1. Teks Asli:**", hasil['teks_asli'])
            st.write("**2. Teks Normalisasi (Input IndoBERT):**", hasil.get('teks_normalized', '-'))
            st.write("**3. Teks Final / Stemmed (Input TF-IDF -> NB & SVM):**", hasil['teks_final'])

            with st.expander("Lihat Referensi Label IndoBERT"):
                st.write(f"Label IndoBERT: **{hasil['label_indobert']}** (Confidence: {hasil['confidence_bert']:.4f})")