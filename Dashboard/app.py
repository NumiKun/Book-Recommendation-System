import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BookMind — Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "Model")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "Dataset")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0d1117; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}
.hero-banner {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 40%, #1565c0 70%, #0288d1 100%);
    border-radius: 16px; padding: 40px 48px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 240px; height: 240px; background: rgba(255,255,255,0.05); border-radius: 50%;
}
.hero-banner h1 {
    color: #ffffff !important; font-size: 2.4rem !important; font-weight: 700 !important;
    margin: 0 0 8px !important; letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255,255,255,0.85) !important; font-size: 1.05rem !important;
    margin: 0 !important; font-weight: 400;
}
.metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 24px; text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
.metric-card .metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; display: block; }
.metric-card .metric-label {
    font-size: 0.82rem; color: #8b949e; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px;
}
.book-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 16px; text-align: center; height: 100%;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.book-card:hover {
    border-color: #58a6ff; box-shadow: 0 0 20px rgba(88,166,255,0.15); transform: translateY(-3px);
}
.book-card img {
    border-radius: 6px; width: 90px; height: 130px; object-fit: cover;
    margin-bottom: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.book-card .book-title {
    font-size: 0.82rem; font-weight: 600; color: #e6edf3; margin-bottom: 4px;
    line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
}
.book-card .book-author { font-size: 0.74rem; color: #8b949e; }
.book-card .book-score {
    display: inline-block;
    background: linear-gradient(90deg, #1565c0, #0288d1);
    color: #fff; font-size: 0.72rem; font-weight: 600;
    padding: 2px 10px; border-radius: 20px; margin-top: 8px;
}
.book-card .rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #58a6ff, #3fb950);
    color: #0d1117; font-size: 0.7rem; font-weight: 700;
    padding: 2px 8px; border-radius: 20px; margin-bottom: 6px;
}
.section-header {
    font-size: 1.3rem; font-weight: 600; color: #e6edf3;
    margin-bottom: 4px; padding-bottom: 8px; border-bottom: 2px solid #21262d;
}
.section-sub { font-size: 0.88rem; color: #8b949e; margin-bottom: 20px; }
.info-box {
    background: rgba(88,166,255,0.08); border: 1px solid rgba(88,166,255,0.3);
    border-radius: 8px; padding: 12px 16px; color: #8b949e; font-size: 0.84rem; margin: 10px 0;
}
.nav-title {
    font-size: 0.7rem; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 1px;
    margin-top: 20px; margin-bottom: 6px; padding-left: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff", "#ff7b72"]
CHART_LAYOUT = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Inter", color="#e6edf3", size=12),
    margin=dict(t=40, b=40, l=40, r=20),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    with open(os.path.join(MODEL_DIR, "als_model.pkl"), "rb") as f:
        als = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "svdpp_model.pkl"), "rb") as f:
        svd = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    tfidf_mat = sp.load_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"))
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
        enc = pickle.load(f)
    return als, svd, tfidf, tfidf_mat, enc


@st.cache_data(show_spinner="Loading book catalog…")
def load_books():
    books = pd.read_parquet(os.path.join(MODEL_DIR, "books_filtered.parquet"))
    books["Year"] = pd.to_numeric(books["Year"], errors="coerce")
    books = books[books["Year"].between(1900, 2025) | books["Year"].isna()]
    return books.reset_index(drop=True)


@st.cache_data(show_spinner="Loading ratings…")
def load_ratings():
    r = pd.read_csv(os.path.join(DATASET_DIR, "Ratings.csv"), encoding="latin-1", on_bad_lines="skip")
    r.columns = ["UserID", "ISBN", "Rating"]
    r["UserID"] = pd.to_numeric(r["UserID"], errors="coerce")
    r["Rating"] = pd.to_numeric(r["Rating"], errors="coerce")
    return r.dropna()


@st.cache_data(show_spinner="Building user-item matrix…")
def build_user_item_matrix(_ratings, _user_enc, _item_enc):
    explicit = _ratings[_ratings["Rating"] > 0].copy()
    valid_users = set(_user_enc.classes_.tolist())
    valid_isbns = set(_item_enc.classes_.tolist())
    explicit = explicit[explicit["UserID"].isin(valid_users)]
    explicit = explicit[explicit["ISBN"].isin(valid_isbns)]
    uid_idx = _user_enc.transform(explicit["UserID"])
    iid_idx = _item_enc.transform(explicit["ISBN"])
    conf = 1 + 40 * explicit["Rating"].values
    mat = sp.csr_matrix(
        (conf.astype(np.float32), (uid_idx, iid_idx)),
        shape=(len(_user_enc.classes_), len(_item_enc.classes_))
    )
    return mat, explicit


def content_recs(isbn, tfidf_mat, isbn_to_idx, books, top_n=10):
    if isbn not in isbn_to_idx:
        return pd.DataFrame()
    idx = isbn_to_idx[isbn]
    scores = cosine_similarity(tfidf_mat[idx], tfidf_mat).flatten()
    top_idx = np.argsort(scores)[::-1][1: top_n + 1]
    result = books.iloc[top_idx][["ISBN", "Title", "Author", "Year", "Image-L"]].copy()
    result["Score"] = scores[top_idx].round(4)
    return result.reset_index(drop=True)


def als_recs(user_id, als_model, user_enc, item_enc, books, uim, top_n=10):
    if user_id not in user_enc.classes_:
        return pd.DataFrame()
    uid = int(np.where(user_enc.classes_ == user_id)[0][0])
    ids, scores = als_model.recommend(uid, uim[uid], N=top_n, filter_already_liked_items=True)
    isbns = item_enc.inverse_transform(ids)
    result = books[books["ISBN"].isin(isbns)][["ISBN", "Title", "Author", "Year", "Image-L"]].copy()
    score_map = dict(zip(isbns, scores))
    result["Score"] = result["ISBN"].map(score_map)
    return result.sort_values("Score", ascending=False).reset_index(drop=True)


def render_book_card(row, rank=None, score_label="Score", show_score=True):
    rank_html = f'<div class="rank-badge">#{rank}</div><br>' if rank else ""
    score_html = ""
    if show_score and "Score" in row.index and pd.notna(row["Score"]):
        sv = row["Score"]
        if score_label == "Rating":
            score_html = f'<div class="book-score">⭐ {sv:.1f} / 10</div>'
        else:
            score_html = f'<div class="book-score">🎯 {sv:.3f}</div>'
    img_url = row.get("Image-L", "") or ""
    img_html = (
        f'<img src="{img_url}" onerror="this.style.display=\'none\'">'
        if img_url
        else '<div style="height:130px;display:flex;align-items:center;justify-content:center;color:#30363d;font-size:2rem;">📖</div>'
    )
    year = f" · {int(row['Year'])}" if pd.notna(row.get("Year")) and row.get("Year") else ""
    return f"""
    <div class="book-card">
        {rank_html}{img_html}
        <div class="book-title">{row['Title']}</div>
        <div class="book-author">{row['Author']}{year}</div>
        {score_html}
    </div>"""


def display_book_grid(df, score_label="Score", show_score=True, cols=5, show_rank=True):
    for start in range(0, len(df), cols):
        chunk = df.iloc[start: start + cols]
        columns = st.columns(cols)
        for i, (_, row) in enumerate(chunk.iterrows()):
            rank = start + i + 1 if show_rank else None
            with columns[i]:
                st.markdown(render_book_card(row, rank=rank, score_label=score_label, show_score=show_score),
                            unsafe_allow_html=True)


# ── Load resources ─────────────────────────────────────────────────────────────
als_model, svd_model, tfidf, tfidf_matrix, enc = load_models()
books = load_books()
ratings = load_ratings()
user_enc = enc["user_enc"]
item_enc = enc["item_enc"]
isbn_to_idx = enc["isbn_to_idx"]
user_item_matrix, explicit_df = build_user_item_matrix(ratings, user_enc, item_enc)

valid_users = sorted(
    explicit_df.groupby("UserID").size()[
        explicit_df.groupby("UserID").size() >= 5
    ].index.tolist()
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px;">
        <div style="font-size:2.5rem;">📚</div>
        <div style="font-size:1.2rem;font-weight:700;color:#e6edf3;">BookMind</div>
        <div style="font-size:0.78rem;color:#8b949e;">Book Recommendation System</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="nav-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("page",
                    ["🏠 Overview", "🔍 Content-Based", "👤 Collaborative", "🔀 Hybrid", "📊 Analytics"],
                    label_visibility="collapsed")

    st.markdown('<div class="nav-title">Settings</div>', unsafe_allow_html=True)
    top_n = st.slider("Results to show", 5, 20, 10, 1)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#8b949e;text-align:center;line-height:1.8;">
        <b style="color:#e6edf3;">Models Used</b><br>SVD++ · ALS · TF-IDF<br><br>
        <b style="color:#e6edf3;">Dataset</b><br>Book-Crossing<br>1.1M ratings · 13K books
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-banner">
        <h1>📚 BookMind</h1>
        <p>An intelligent hybrid book recommendation system combining collaborative filtering,
        matrix factorization, and content-based analysis — powered by the Book-Crossing dataset.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{len(books):,}", "Books in Catalog"),
        (f"{len(user_enc.classes_):,}", "Unique Users"),
        (f"{len(explicit_df):,}", "Explicit Ratings"),
        (f"{explicit_df['Rating'].mean():.2f}/10", "Avg. Rating"),
        ("3", "Models Deployed"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(f"""
        <div class="metric-card">
            <span class="metric-value">{val}</span>
            <span class="metric-label">{label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">🧠 System Architecture</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Three complementary models fused into a hybrid engine</div>', unsafe_allow_html=True)
        for icon, name, desc, color in [
            ("🎯", "Content-Based (TF-IDF)", "Finds books with similar metadata — title, author, and year — using cosine similarity on TF-IDF vectors.", "#58a6ff"),
            ("🤝", "Collaborative (ALS)", "Implicit Matrix Factorization trained on rating patterns, capturing latent user-item interactions.", "#3fb950"),
            ("⭐", "Model-Based (SVD++)", "Latent factor model with implicit feedback, achieving low RMSE via Surprise library.", "#d2a8ff"),
        ]:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-left:3px solid {color};
                border-radius:10px;padding:14px 18px;margin-bottom:10px;">
                <div style="font-size:0.9rem;font-weight:600;color:{color};margin-bottom:4px;">{icon} {name}</div>
                <div style="font-size:0.82rem;color:#8b949e;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">📈 Rating Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Distribution of explicit ratings (1–10)</div>', unsafe_allow_html=True)
        rc = explicit_df["Rating"].value_counts().sort_index()
        fig_r = go.Figure(go.Bar(
            x=rc.index.tolist(), y=rc.values.tolist(),
            marker=dict(color=rc.values.tolist(), colorscale=[[0, "#21262d"], [1, "#58a6ff"]], showscale=False),
            text=rc.values.tolist(), textposition="outside", textfont=dict(color="#8b949e", size=10),
        ))
        fig_r.update_layout(**CHART_LAYOUT, xaxis_title="Rating Score", yaxis_title="Number of Ratings",
                             bargap=0.2, height=300)
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="section-header">✍️ Most Reviewed Authors</div>', unsafe_allow_html=True)
        ac = (explicit_df.merge(books[["ISBN", "Author"]], on="ISBN", how="left")["Author"]
              .value_counts().head(10))
        fig_auth = go.Figure(go.Bar(
            x=ac.values[::-1], y=ac.index[::-1], orientation="h", marker_color=COLORS[0]))
        fig_auth.update_layout(**CHART_LAYOUT, height=340, xaxis_title="Ratings Count")
        st.plotly_chart(fig_auth, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">📅 Books Published by Decade</div>', unsafe_allow_html=True)
        dec_df = books[books["Year"].between(1900, 2024)].copy()
        dec_df["Decade"] = (dec_df["Year"] // 10 * 10).astype(int)
        dec_counts = dec_df.groupby("Decade").size().reset_index(name="Count")
        fig_dec = go.Figure(go.Bar(
            x=dec_counts["Decade"].tolist(), y=dec_counts["Count"].tolist(), marker=dict(color=COLORS[1])))
        fig_dec.update_layout(**CHART_LAYOUT, height=340, xaxis_title="Decade", yaxis_title="Number of Books")
        st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown('<div class="section-header">🔥 Most Rated Books</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Top books by number of explicit ratings received</div>', unsafe_allow_html=True)
    pop = (explicit_df.groupby("ISBN")
           .agg(NumRatings=("Rating", "count"), AvgRating=("Rating", "mean"))
           .reset_index()
           .merge(books[["ISBN", "Title", "Author", "Year", "Image-L"]], on="ISBN", how="left")
           .sort_values("NumRatings", ascending=False).head(10).reset_index(drop=True))
    pop["Score"] = pop["AvgRating"]
    display_book_grid(pop, score_label="Rating", show_score=True, cols=5, show_rank=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CONTENT-BASED
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Content-Based":
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,#1a3a1a 0%,#1b5e20 50%,#2e7d32 100%);">
        <h1>🔍 Content-Based Recommendations</h1>
        <p>Find books similar to any title using TF-IDF cosine similarity on book metadata
        (title, author, publication year).</p>
    </div>""", unsafe_allow_html=True)

    col_search, col_info = st.columns([2, 1], gap="large")
    with col_search:
        st.markdown('<div class="section-header">Select a Book</div>', unsafe_allow_html=True)
        query = st.text_input("🔎 Search by title or author", placeholder="e.g. Harry Potter, Stephen King…")
        if query:
            mask = (books["Title"].str.contains(query, case=False, na=False) |
                    books["Author"].str.contains(query, case=False, na=False))
            search_results = books[mask].reset_index(drop=True)
        else:
            pop_isbns = explicit_df.groupby("ISBN").size().sort_values(ascending=False).head(200).index
            search_results = books[books["ISBN"].isin(pop_isbns)].reset_index(drop=True)

        if len(search_results) == 0:
            st.warning("No books found. Try a different search term.")
            st.stop()

        opts = [f"{r['Title']} — {r['Author']}" for _, r in search_results.iterrows()]
        sel_idx = st.selectbox("Choose a book", range(len(opts)), format_func=lambda i: opts[i])
        sel_book = search_results.iloc[sel_idx]
        sel_isbn = sel_book["ISBN"]

    with col_info:
        st.markdown('<div class="section-header">Selected Book</div>', unsafe_allow_html=True)
        img = sel_book.get("Image-L", "") or ""
        year_str = str(int(sel_book["Year"])) if pd.notna(sel_book.get("Year")) else "N/A"
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;text-align:center;">
            <img src="{img}" style="border-radius:8px;max-height:180px;max-width:100%;
                box-shadow:0 4px 16px rgba(0,0,0,0.5);" onerror="this.style.display='none'">
            <div style="font-size:1rem;font-weight:600;color:#e6edf3;margin-top:12px;">{sel_book['Title']}</div>
            <div style="font-size:0.84rem;color:#58a6ff;margin-top:4px;">{sel_book['Author']}</div>
            <div style="font-size:0.78rem;color:#8b949e;margin-top:4px;">{year_str}</div>
            <div style="font-size:0.72rem;color:#30363d;margin-top:6px;font-family:monospace;">ISBN: {sel_isbn}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">📖 Similar Books</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Top {top_n} content-based recommendations</div>', unsafe_allow_html=True)

    with st.spinner("Computing similarities…"):
        cb_recs = content_recs(sel_isbn, tfidf_matrix, isbn_to_idx, books, top_n=top_n)

    if cb_recs.empty:
        st.info("No content-based recommendations available for this book.")
    else:
        display_book_grid(cb_recs, score_label="Score", show_score=True, cols=5)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Similarity Scores</div>', unsafe_allow_html=True)
        labels_cb = [
            f"{r['Title'][:30]}…" if len(r["Title"]) > 30 else r["Title"]
            for _, r in cb_recs.iterrows()
        ]
        fig_sim = go.Figure(go.Bar(
            x=cb_recs["Score"].tolist(), y=labels_cb,
            orientation="h", marker=dict(color=COLORS[1], opacity=0.85)))
        fig_sim.update_layout(**CHART_LAYOUT, height=350, xaxis_title="Cosine Similarity",
                              yaxis=dict(autorange="reversed", **CHART_LAYOUT["yaxis"]))
        st.plotly_chart(fig_sim, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COLLABORATIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Collaborative":
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,#4a148c 0%,#6a1b9a 50%,#8e24aa 100%);">
        <h1>👤 Collaborative Filtering</h1>
        <p>Personalized recommendations from ALS Implicit Matrix Factorization —
        learning from rating patterns across all users.</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 2], gap="large")
    with col_l:
        st.markdown('<div class="section-header">Select a User</div>', unsafe_allow_html=True)
        sel_user = st.selectbox("User ID", valid_users, format_func=lambda u: f"User #{u}")
        u_hist = explicit_df[explicit_df["UserID"] == sel_user]
        u_hist_books = u_hist.merge(books[["ISBN", "Title", "Author", "Year", "Image-L"]], on="ISBN", how="left")

        st.markdown(f"""
        <div class="info-box">
            👤 <b>User #{sel_user}</b><br>
            📚 {len(u_hist)} books rated<br>
            ⭐ Avg rating: {u_hist['Rating'].mean():.1f} / 10
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="font-size:1rem;margin-top:16px;">Reading History</div>',
                    unsafe_allow_html=True)
        for _, row in u_hist_books.sort_values("Rating", ascending=False).head(6).iterrows():
            title_trunc = str(row["Title"])[:40] + ("…" if len(str(row["Title"])) > 40 else "")
            stars = "⭐" * int(row["Rating"] // 2)
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:10px 12px;margin-bottom:6px;">
                <div style="font-size:0.82rem;color:#e6edf3;font-weight:500;">{stars} {title_trunc}</div>
                <div style="font-size:0.74rem;color:#8b949e;">{row['Author']} · {row['Rating']:.0f}/10</div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown(f'<div class="section-header">🤝 ALS Recommendations for User #{sel_user}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Top {top_n} personalized picks from collaborative filtering</div>',
                    unsafe_allow_html=True)
        with st.spinner("Generating ALS recommendations…"):
            als_result = als_recs(sel_user, als_model, user_enc, item_enc, books, user_item_matrix, top_n=top_n)

        if als_result.empty:
            st.info("No ALS recommendations available for this user.")
        else:
            display_book_grid(als_result, score_label="Score", show_score=True, cols=5)

    st.markdown("---")
    st.markdown('<div class="section-header">🎭 User Taste Profile</div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2, gap="large")

    with col_c1:
        fav_auth = (u_hist_books.groupby("Author")["Rating"].mean()
                    .sort_values(ascending=False).head(8))
        fig_fa = go.Figure(go.Bar(
            x=fav_auth.values[::-1], y=fav_auth.index[::-1],
            orientation="h", marker_color=COLORS[3]))
        fig_fa.update_layout(**CHART_LAYOUT, height=300,
                              title=dict(text="Avg. Rating by Author", font=dict(size=13, color="#e6edf3")),
                              xaxis=dict(range=[0, 10], **CHART_LAYOUT["xaxis"]))
        st.plotly_chart(fig_fa, use_container_width=True)

    with col_c2:
        rd = u_hist["Rating"].value_counts().sort_index()
        fig_rd = go.Figure(go.Bar(
            x=rd.index.tolist(), y=rd.values.tolist(), marker_color=COLORS[0]))
        fig_rd.update_layout(**CHART_LAYOUT, height=300,
                              title=dict(text="Rating Distribution", font=dict(size=13, color="#e6edf3")),
                              xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig_rd, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HYBRID
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔀 Hybrid":
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,#7f1d1d 0%,#991b1b 40%,#b45309 100%);">
        <h1>🔀 Hybrid Recommendation Engine</h1>
        <p>Combines SVD++ predicted ratings, ALS collaborative scores, and TF-IDF content similarity —
        fused with configurable weights for robust personalization.</p>
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_out = st.columns([1, 2], gap="large")
    with col_ctrl:
        st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        sel_user_h = st.selectbox("User ID", valid_users, format_func=lambda u: f"User #{u}", key="hybrid_user")

        u_hist_h = explicit_df[explicit_df["UserID"] == sel_user_h]
        u_hist_books_h = u_hist_h.merge(books[["ISBN", "Title", "Author"]], on="ISBN", how="left")
        top_rated_h = u_hist_books_h.sort_values("Rating", ascending=False).head(20)

        seed_opts = [f"{r['Title'][:40]} — {r['Author']}" for _, r in top_rated_h.iterrows()]
        seed_idx = st.selectbox("Seed book (content arm)", range(len(seed_opts)),
                                format_func=lambda i: seed_opts[i], key="hybrid_seed")
        seed_isbn = top_rated_h.iloc[seed_idx]["ISBN"] if len(top_rated_h) > 0 else None

        st.markdown("**Fusion Weights**")
        w_als = st.slider("ALS weight", 0.0, 1.0, 0.40, 0.05)
        w_svd = st.slider("SVD++ weight", 0.0, 1.0, 0.40, 0.05)
        w_cb = st.slider("Content weight", 0.0, 1.0, 0.20, 0.05)
        total_w = w_als + w_svd + w_cb
        if abs(total_w - 1.0) > 0.05:
            st.warning(f"⚠️ Weights sum to {total_w:.2f} — will be auto-normalised.")

        run_btn = st.button("🚀 Generate Hybrid Recommendations", type="primary", use_container_width=True)

    with col_out:
        st.markdown(f'<div class="section-header">🔀 Hybrid Results</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Top {top_n} recommendations fused from all three models</div>',
                    unsafe_allow_html=True)

        if run_btn:
            with st.spinner("Running all three models and fusing scores…"):
                tw = w_als + w_svd + w_cb
                wa, ws, wc = w_als / tw, w_svd / tw, w_cb / tw

                als_scores = {}
                if sel_user_h in user_enc.classes_:
                    uid_v = int(np.where(user_enc.classes_ == sel_user_h)[0][0])
                    ids, sc = als_model.recommend(uid_v, user_item_matrix[uid_v],
                                                  N=50, filter_already_liked_items=True)
                    isbns_als = item_enc.inverse_transform(ids)
                    mx = max(sc) if len(sc) > 0 else 1
                    als_scores = {isbn: float(s / mx) for isbn, s in zip(isbns_als, sc)}

                cb_scores = {}
                if seed_isbn and seed_isbn in isbn_to_idx:
                    cb_df = content_recs(seed_isbn, tfidf_matrix, isbn_to_idx, books, top_n=50)
                    for _, r in cb_df.iterrows():
                        cb_scores[r["ISBN"]] = float(r["Score"])

                candidate_list = list(set(als_scores.keys()) | set(cb_scores.keys()))
                svd_scores = {}
                for isbn in candidate_list[:300]:
                    try:
                        p = svd_model.predict(sel_user_h, isbn)
                        svd_scores[isbn] = float(p.est / 10.0)
                    except Exception:
                        pass

                all_isbns = set(als_scores.keys()) | set(cb_scores.keys()) | set(svd_scores.keys())
                fused = sorted(
                    [(isbn, wa * als_scores.get(isbn, 0) + ws * svd_scores.get(isbn, 0) + wc * cb_scores.get(isbn, 0))
                     for isbn in all_isbns],
                    key=lambda x: x[1], reverse=True
                )[:top_n]

                top_isbns_h = [x[0] for x in fused]
                score_map_h = {x[0]: x[1] for x in fused}
                hybrid_df = books[books["ISBN"].isin(top_isbns_h)][["ISBN", "Title", "Author", "Year", "Image-L"]].copy()
                hybrid_df["Score"] = hybrid_df["ISBN"].map(score_map_h)
                hybrid_df = hybrid_df.sort_values("Score", ascending=False).reset_index(drop=True)

            display_book_grid(hybrid_df, score_label="Score", show_score=True, cols=5)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚖️ Score Contribution Breakdown</div>', unsafe_allow_html=True)
            breakdown_data = []
            for _, r in hybrid_df.iterrows():
                isbn = r["ISBN"]
                title = r["Title"][:25] + "…" if len(r["Title"]) > 25 else r["Title"]
                breakdown_data.append({
                    "Book": title,
                    "ALS": wa * als_scores.get(isbn, 0),
                    "SVD++": ws * svd_scores.get(isbn, 0),
                    "Content": wc * cb_scores.get(isbn, 0),
                })
            bd = pd.DataFrame(breakdown_data)
            fig_stack = go.Figure()
            for arm, color in zip(["ALS", "SVD++", "Content"], [COLORS[0], COLORS[3], COLORS[1]]):
                fig_stack.add_trace(go.Bar(name=arm, x=bd["Book"], y=bd[arm], marker_color=color))
            fig_stack.update_layout(**CHART_LAYOUT, barmode="stack", height=320,
                                    legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:40px;">
                ⚙️ Configure the weights and click <b>Generate</b> to see hybrid recommendations.
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,#0f3460 0%,#16213e 50%,#1a1a2e 100%);">
        <h1>📊 Dataset Analytics</h1>
        <p>Exploratory analysis of the Book-Crossing dataset — user behaviour, rating patterns, and catalog insights.</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📚 Catalog", "👥 Users", "⭐ Ratings"])

    with tab1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="section-header">🏆 Top Publishers</div>', unsafe_allow_html=True)
            pc = books["Publisher"].value_counts().head(12)
            fig_pub = go.Figure(go.Bar(
                x=pc.values[::-1], y=pc.index[::-1], orientation="h", marker=dict(color=COLORS[4])))
            fig_pub.update_layout(**CHART_LAYOUT, height=380, xaxis_title="Book Count")
            st.plotly_chart(fig_pub, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">✍️ Top 15 Authors by Book Count</div>', unsafe_allow_html=True)
            abc = books["Author"].value_counts().head(15)
            fig_apie = px.pie(values=abc.values, names=abc.index, hole=0.5,
                              color_discrete_sequence=COLORS * 3)
            fig_apie.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                   font=dict(color="#e6edf3", family="Inter"),
                                   margin=dict(t=20, b=20, l=20, r=20), height=380,
                                   legend=dict(bgcolor="#161b22", font=dict(size=10)))
            st.plotly_chart(fig_apie, use_container_width=True)

        st.markdown('<div class="section-header">📅 Publications Timeline (1940–2024)</div>', unsafe_allow_html=True)
        tl = books[books["Year"].between(1940, 2024)].groupby("Year").size().reset_index(name="Count")
        fig_tl = go.Figure(go.Scatter(
            x=tl["Year"], y=tl["Count"], mode="lines+markers",
            line=dict(color=COLORS[0], width=2), marker=dict(size=4, color=COLORS[0]),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.1)"))
        fig_tl.update_layout(**CHART_LAYOUT, height=280, xaxis_title="Year", yaxis_title="Books Published")
        st.plotly_chart(fig_tl, use_container_width=True)

    with tab2:
        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.markdown('<div class="section-header">📊 Ratings per User Distribution</div>', unsafe_allow_html=True)
            urc = explicit_df.groupby("UserID").size()
            bins_u = [1, 5, 10, 20, 50, 100, 200, 500, 10000]
            lbls = ["1–4", "5–9", "10–19", "20–49", "50–99", "100–199", "200–499", "500+"]
            ub = pd.cut(urc, bins=bins_u, labels=lbls, right=False).value_counts().reindex(lbls)
            fig_ub = go.Figure(go.Bar(x=lbls, y=ub.values.tolist(), marker_color=COLORS[2]))
            fig_ub.update_layout(**CHART_LAYOUT, height=300,
                                 xaxis_title="Ratings Count", yaxis_title="Number of Users")
            st.plotly_chart(fig_ub, use_container_width=True)

        with c4:
            st.markdown('<div class="section-header">🌍 Top User Locations</div>', unsafe_allow_html=True)
            try:
                udf = pd.read_csv(os.path.join(DATASET_DIR, "Users.csv"), encoding="latin-1", on_bad_lines="skip")
                udf.columns = ["UserID", "Location", "Age"]
                udf["Country"] = udf["Location"].str.split(",").str[-1].str.strip().str.title()
                tc = udf["Country"].value_counts().head(12)
                fig_tc = go.Figure(go.Bar(
                    x=tc.values[::-1], y=tc.index[::-1], orientation="h", marker_color=COLORS[5]))
                fig_tc.update_layout(**CHART_LAYOUT, height=300, xaxis_title="Number of Users")
                st.plotly_chart(fig_tc, use_container_width=True)
            except Exception:
                st.info("User location data unavailable.")

        st.markdown('<div class="section-header">🏆 Most Active Reviewers</div>', unsafe_allow_html=True)
        top_rev = (explicit_df.groupby("UserID")
                   .agg(Total_Ratings=("Rating", "count"), Avg_Rating=("Rating", "mean"))
                   .sort_values("Total_Ratings", ascending=False).head(15).reset_index())
        top_rev["UserID"] = top_rev["UserID"].astype(str)
        fig_tr = go.Figure(go.Bar(
            x=top_rev["UserID"], y=top_rev["Total_Ratings"],
            marker=dict(
                color=top_rev["Avg_Rating"].tolist(),
                colorscale=[[0, "#21262d"], [0.5, "#58a6ff"], [1, "#3fb950"]],
                colorbar=dict(title="Avg Rating", tickfont=dict(color="#8b949e")),
                showscale=True)))
        fig_tr.update_layout(**CHART_LAYOUT, height=300,
                              xaxis_title="User ID", yaxis_title="Number of Ratings")
        st.plotly_chart(fig_tr, use_container_width=True)

    with tab3:
        c5, c6 = st.columns(2, gap="large")
        with c5:
            st.markdown('<div class="section-header">📈 Rating Score Distribution</div>', unsafe_allow_html=True)
            ar = ratings[ratings["Rating"] > 0]["Rating"].value_counts().sort_index()
            fig_ar = go.Figure(go.Bar(
                x=ar.index.tolist(), y=ar.values.tolist(),
                marker=dict(color=ar.values.tolist(),
                            colorscale=[[0, "#21262d"], [1, "#58a6ff"]], showscale=False)))
            fig_ar.update_layout(**CHART_LAYOUT, height=300,
                                 xaxis_title="Rating Score", yaxis_title="Count", bargap=0.15)
            st.plotly_chart(fig_ar, use_container_width=True)

        with c6:
            st.markdown('<div class="section-header">📊 Implicit vs. Explicit Ratings</div>', unsafe_allow_html=True)
            impl = (ratings["Rating"] == 0).sum()
            expl = (ratings["Rating"] > 0).sum()
            fig_pie2 = go.Figure(go.Pie(
                labels=["Implicit (0)", "Explicit (1–10)"],
                values=[impl, expl],
                marker=dict(colors=[COLORS[2], COLORS[0]]),
                hole=0.5, textinfo="label+percent",
                textfont=dict(color="#e6edf3")))
            fig_pie2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                   font=dict(color="#e6edf3", family="Inter"),
                                   margin=dict(t=20, b=20, l=20, r=20), height=300,
                                   legend=dict(bgcolor="#161b22"))
            st.plotly_chart(fig_pie2, use_container_width=True)

        st.markdown('<div class="section-header">📉 Average Rating per Year (1990–2024)</div>', unsafe_allow_html=True)
        yr = (explicit_df.merge(books[["ISBN", "Year"]], on="ISBN", how="left").dropna(subset=["Year"]))
        yr = yr[yr["Year"].between(1990, 2024)]
        yr_agg = yr.groupby("Year")["Rating"].agg(["mean", "count"]).reset_index()
        fig_yr = make_subplots(specs=[[{"secondary_y": True}]])
        fig_yr.add_trace(go.Scatter(
            x=yr_agg["Year"], y=yr_agg["mean"], name="Avg Rating",
            line=dict(color=COLORS[0], width=2), mode="lines+markers"), secondary_y=False)
        fig_yr.add_trace(go.Bar(
            x=yr_agg["Year"], y=yr_agg["count"], name="Rating Count",
            marker=dict(color=COLORS[1], opacity=0.4)), secondary_y=True)
        fig_yr.update_layout(**CHART_LAYOUT, height=300,
                              legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
        fig_yr.update_yaxes(title_text="Avg Rating", secondary_y=False,
                             gridcolor="#21262d", linecolor="#30363d", color="#e6edf3")
        fig_yr.update_yaxes(title_text="Rating Count", secondary_y=True,
                             gridcolor="#21262d", color="#e6edf3")
        st.plotly_chart(fig_yr, use_container_width=True)
