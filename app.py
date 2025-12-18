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
# CSS SEDERHANA & AMAN
# ==========================================
st.markdown("""
<style>
.card {
    background: var(--secondary-background-color);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
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

    return [
        w for w in words
        if w.isalpha() and w not in stop_words and len(w) > 2
    ]

# ==========================================
# GRAPH BUILDER
# ==========================================
def build_graph(words, window_size):
    G = nx.Graph()
    for i in range(len(words) - window_size):
        for j in range(1, window_size + 1):
            if words[i] != words[i + j]:
                G.add_edge(words[i], words[i + j])
    return G

# ==========================================
# CACHE GRAPH & PAGERANK
# ==========================================
@st.cache_data
def compute_graph(words, window_size):
    G = build_graph(words, window_size)
    pr = nx.pagerank(G)
    return G, pr

# ==========================================
# MAIN APP
# ==========================================
def main():
    download_nltk_data()

    if 'paper_data' not in st.session_state:
        st.session_state.paper_data = {}
    if 'active_file_key' not in st.session_state:
        st.session_state.active_file_key = None

    st.title("📄 Analisis Paper PDF")
    st.caption("Visualisasi relasi kata & PageRank (dynamic window size)")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📂 Upload & Pengaturan")

        uploaded_files = st.file_uploader(
            "Upload PDF (Bisa Banyak)",
            type="pdf",
            accept_multiple_files=True
        )

        window_size = st.slider(
            "Jarak Hubungan Kata",
            min_value=1,
            max_value=5,
            value=2
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- PROCESS PDF ----------
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.paper_data:
                with st.spinner(f"Memproses {uploaded_file.name}..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                    words = process_text(raw_text)

                    st.session_state.paper_data[uploaded_file.name] = {
                        'words': words,
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
        words = data['words']

        # REBUILD GRAPH SETIAP SLIDER BERUBAH
        G, pr = compute_graph(words, window_size)

        df = pd.DataFrame(pr.items(), columns=['Kata', 'PageRank'])
        df = df.sort_values(by='PageRank', ascending=False).reset_index(drop=True)
        df.insert(0, 'ID', range(1, len(df) + 1))

        st.markdown(f"""
        <div class="card">
            <b>{selected}</b><br>
            Kata Bersih: {data['count']} |
            Node: {len(G.nodes())} |
            Edge: {len(G.edges())}
        </div>
        """, unsafe_allow_html=True)

        # ---------- FILTER ----------
        st.sidebar.divider()
        st.sidebar.subheader("🔎 Filter Graph")

        max_words = st.sidebar.number_input(
            "Jumlah kata ditampilkan",
            min_value=1,
            max_value=len(df),
            value=min(40, len(df))
        )

        top_words = df.head(max_words)['Kata'].tolist()

        selected_words = st.sidebar.multiselect(
            "Pilih kata",
            options=top_words,
            default=top_words
        )

        search_input = st.sidebar.text_input(
            "Cari & highlight kata",
            placeholder="pisahkan dengan koma"
        )

        search_terms = [
            w.strip().lower()
            for w in search_input.split(",")
            if w.strip()
        ]

        G_vis = G.subgraph(selected_words) if selected_words else G

        col1, col2 = st.columns([3, 2])

        # ---------- GRAPH ----------
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🕸️ Relasi Kata")

            net = Network(
                height="600px",
                width="100%",
                bgcolor="#ffffff",
                font_color="black"
            )

            net.from_nx(G_vis)

            for node in net.nodes:
                word = node['id']
                score = pr.get(word, 0.001)

                node['size'] = score * 1200
                node['physics'] = False
                node['color'] = "#97C2FC"

                if word in search_terms:
                    node['color'] = "#FF4B4B"
                    node['size'] = score * 1800
                    node['borderWidth'] = 3

                node['title'] = f"{word}<br>PageRank: {score:.4f}"

            net.toggle_physics(False)
            net.save_graph("graph.html")

            with open("graph.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=620)

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------- STATS ----------
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(f"📊 Top {max_words} Kata")
            st.bar_chart(
                df.head(max_words)
                    .set_index("Kata")["PageRank"]
            )

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📋 Tabel Lengkap")
            st.dataframe(df, height=320, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Silakan upload file PDF di sidebar.")

if __name__ == "__main__":
    main()
