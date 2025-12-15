import streamlit as st
import fitz
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

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Paper PDF",
    page_icon="📄",
    layout="wide"
)

# ==========================================
# CSS ELEGAN + ANIMASI (AMAN)
# ==========================================
st.markdown("""
<style>
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

.fade-in {
    animation: fadeIn 0.5s ease-in-out;
}

.card {
    background: var(--secondary-background-color);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# DOWNLOAD NLTK
# ==========================================
@st.cache_resource
def download_nltk_data():
    for res in ['punkt', 'stopwords']:
        try:
            nltk.data.find(res)
        except:
            nltk.download(res, quiet=True)

# ==========================================
# PDF READER
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
# TEXT PROCESSING
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
        'dan','yang','di','ke','dari','ini','itu','pada','untuk','dengan',
        'penelitian','data','hasil','analisis','kesimpulan','sistem',
        'gambar','tabel','bab','jurnal','paper','skripsi','tesis',
        'abstract','abstrak'
    ]
    stop_words.update(custom_stopwords)

    return [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]

# ==========================================
# GRAPH BUILDER
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
    if 'active_file_key' not in st.session_state:
        st.session_state.active_file_key = None

    # ---------- HEADER ----------
    st.markdown("""
    <div class="fade-in">
        <h1>📄 Analisis Paper PDF</h1>
        <p>Visualisasi relasi kata dan PageRank</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.subheader("📂 Upload & Pengaturan")
        uploaded_files = st.file_uploader(
            "Upload PDF (Bisa Banyak)",
            type="pdf",
            accept_multiple_files=True
        )
        window_size = st.slider("Jarak Hubungan Kata", 1, 5, 2)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- PROCESS ----------
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.paper_data:
                with st.spinner(f"Memproses {uploaded_file.name}..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                    words = process_text(raw_text)

                    if len(words) > 5:
                        G = build_graph(words, window_size)
                        pr = nx.pagerank(G)

                        df = pd.DataFrame(pr.items(), columns=['Kata', 'PageRank'])
                        df = df.sort_values(by='PageRank', ascending=False).reset_index(drop=True)
                        df.insert(0, 'ID', range(1, len(df) + 1))

                        st.session_state.paper_data[uploaded_file.name] = {
                            'graph': G,
                            'pagerank': pr,
                            'df': df,
                            'count': len(words)
                        }
                        st.session_state.active_file_key = uploaded_file.name

    # ---------- DISPLAY ----------
    if st.session_state.paper_data:
        files = list(st.session_state.paper_data.keys())
        selected = st.sidebar.selectbox(
            "Pilih Paper",
            files,
            index=files.index(st.session_state.active_file_key)
        )

        data = st.session_state.paper_data[selected]
        G, pr, df = data['graph'], data['pagerank'], data['df']

        st.markdown(f"""
        <div class="card fade-in">
            <b>{selected}</b><br>
            Kata Bersih: {data['count']} | Node: {len(G.nodes())}
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])

        # GRAPH
        with col1:
            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
            st.subheader("🕸️ Relasi Kata")
            pos = nx.spring_layout(G, seed=42)

            net = Network(height="600px", width="100%")
            net.from_nx(G)

            for node in net.nodes:
                score = pr.get(node['id'], 0.01)
                node['size'] = score * 1000
                node['physics'] = False

            net.save_graph("graph.html")
            components.html(open("graph.html", "r", encoding="utf-8").read(), height=620)
            st.markdown('</div>', unsafe_allow_html=True)

        # STATS
        with col2:
            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
            st.subheader("📊 Top 20 Kata")
            st.bar_chart(df.head(20).set_index("Kata")["PageRank"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
            st.subheader("📋 Tabel Lengkap")
            st.dataframe(df, height=320, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Silakan upload file PDF di sidebar.")

if __name__ == "__main__":
    main()
