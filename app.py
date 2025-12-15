# ==========================================
# IMPORT LIBRARY
# ==========================================
import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import re
import streamlit.components.v1 as components
import os
import time

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Paper PDF",
    page_icon="📑",
    layout="wide"
)

# ==========================================
# CSS MODERN + ANIMASI
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.7s ease-in-out;
}

.card {
    background: white;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.title-text {
    font-size: 2.4rem;
    font-weight: 700;
}

.subtitle {
    color: #666;
    font-size: 1rem;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #ddd, transparent);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# DOWNLOAD NLTK RESOURCE
# ==========================================
@st.cache_resource
def download_nltk_data():
    for res in ['punkt', 'stopwords']:
        try:
            nltk.data.find(res)
        except:
            nltk.download(res, quiet=True)

# ==========================================
# BACA PDF
# ==========================================
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    os.remove(path)
    return text

# ==========================================
# PREPROCESSING TEKS
# ==========================================
def process_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    try:
        words = word_tokenize(text)
    except:
        words = text.split()

    try:
        stop_words = set(stopwords.words('indonesian'))
    except:
        stop_words = set()

    custom_stopwords = [
        'dan','yang','di','ke','dari','ini','itu','pada','untuk','dengan','adalah',
        'penelitian','data','hasil','analisis','kesimpulan','sistem','metode','aplikasi',
        'gambar','tabel','bab','jurnal','paper','skripsi','tesis','abstrak','abstract'
    ]
    stop_words.update(custom_stopwords)

    return [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]

# ==========================================
# GRAPH CO-OCCURRENCE
# ==========================================
def build_graph(words, window_size=2):
    G = nx.Graph()
    for i in range(len(words) - window_size):
        for j in range(1, window_size + 1):
            if words[i] != words[i + j]:
                G.add_edge(words[i], words[i + j])
    return G

# ==========================================
# MAIN APP
# ==========================================
def main():
    download_nltk_data()

    if 'paper_data' not in st.session_state:
        st.session_state.paper_data = {}
    if 'active_file' not in st.session_state:
        st.session_state.active_file = None

    # ---------- HEADER ----------
    st.markdown("""
    <div class="fade-in">
        <div class="title-text">📑 Analisis Paper PDF</div>
        <p class="subtitle">Visualisasi relasi kata & ranking PageRank secara interaktif</p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("""
        <div class="card fade-in">
            <h4>⚙️ Kontrol Analisis</h4>
            <p class="subtitle">Upload PDF & atur parameter graph</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload File PDF",
            type="pdf",
            accept_multiple_files=True
        )

        window_size = st.slider("Jarak Hubungan Kata", 1, 5, 2)

    # ---------- PROSES FILE ----------
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.paper_data:
                with st.spinner("🔍 Menganalisis dokumen..."):
                    progress = st.progress(0)
                    time.sleep(0.2)

                    raw_text = extract_text_from_pdf(file)
                    progress.progress(40)

                    words = process_text(raw_text)
                    progress.progress(70)

                    if len(words) > 5:
                        G = build_graph(words, window_size)
                        pr = nx.pagerank(G)

                        df = pd.DataFrame(pr.items(), columns=["Kata", "PageRank"])
                        df = df.sort_values("PageRank", ascending=False).reset_index(drop=True)

                        st.session_state.paper_data[file.name] = {
                            "graph": G,
                            "pagerank": pr,
                            "df": df,
                            "count": len(words)
                        }
                        st.session_state.active_file = file.name

                    progress.progress(100)
                    time.sleep(0.3)

    # ---------- TAMPILAN ----------
    if st.session_state.paper_data:
        files = list(st.session_state.paper_data.keys())
        selected = st.sidebar.selectbox(
            "Pilih Paper",
            files,
            index=files.index(st.session_state.active_file)
        )

        data = st.session_state.paper_data[selected]
        G = data["graph"]
        df = data["df"]

        st.markdown(f"""
        <div class="card fade-in">
            <h3>📄 {selected}</h3>
            <p>Jumlah Kata Bersih: <b>{data['count']}</b> | Node: <b>{len(G.nodes())}</b></p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3,2])

        # GRAPH
        with col1:
            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
            st.subheader("🕸️ Network Relasi Kata")

            net = Network(height="600px", width="100%", bgcolor="#ffffff")
            net.from_nx(G)
            net.toggle_physics(False)
            net.save_graph("graph.html")

            with open("graph.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=620)

            st.markdown('</div>', unsafe_allow_html=True)

        # STATISTIK
        with col2:
            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
            st.subheader("📊 Top 20 Kata Dominan")
            st.bar_chart(df.head(20).set_index("Kata")["PageRank"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
            st.subheader("📋 Tabel Lengkap")
            st.dataframe(df, height=350, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("📥 Silakan upload file PDF di sidebar")

if __name__ == "__main__":
    main()
