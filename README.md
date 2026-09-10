<div align="center">

# 📚 BookMind — Book Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://book-recommendation-syste.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](Notebook/book_recommendation_system.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A hybrid book recommendation system combining collaborative filtering, matrix factorization, and content-based analysis — trained on the Book-Crossing dataset with 1.1M+ ratings.**

[🚀 Live Demo](https://book-recommendation-syste.streamlit.app/) · [📓 Notebook](Notebook/book_recommendation_system.ipynb) · [📊 Dataset](#dataset)

</div>

---

## 🎯 Overview

BookMind is a complete, end-to-end book recommendation system built using three complementary approaches fused into a **hybrid engine**:

| Model | Algorithm | Library |
|---|---|---|
| 🎯 Content-Based | TF-IDF + Cosine Similarity | `scikit-learn` |
| 🤝 Collaborative Filtering | Implicit ALS (Matrix Factorization) | `implicit` |
| ⭐ Model-Based CF | SVD++ with Implicit Feedback | `scikit-surprise` |

The system is presented through an interactive **Streamlit dashboard** with 5 pages: Overview, Content-Based, Collaborative, Hybrid, and Analytics.

---

## 🚀 Live Demo

**👉 [https://book-recommendation-syste.streamlit.app/](https://book-recommendation-syste.streamlit.app/)**

The live app includes:
- 🏠 **Overview** — KPI metrics, rating distribution, top authors, most rated books
- 🔍 **Content-Based** — Search any book and find similar titles via TF-IDF
- 👤 **Collaborative** — Select a user and get personalized ALS recommendations
- 🔀 **Hybrid** — Tune ALS / SVD++ / Content weights and generate fused recommendations
- 📊 **Analytics** — Catalog, user, and rating analytics with interactive Plotly charts

---

## 📁 Project Structure

```
Book Recommendation System/
│
├── 📓 Notebook/
│   └── book_recommendation_system.ipynb   # Full ML pipeline (EDA → Training → Inference)
│
├── 🤖 Model/
│   ├── als_model.pkl                      # Trained ALS model
│   ├── svdpp_model.pkl                    # Trained SVD++ model
│   ├── tfidf_vectorizer.pkl               # Fitted TF-IDF vectorizer
│   ├── tfidf_matrix.npz                   # Sparse TF-IDF matrix (CSR format)
│   ├── encoders.pkl                       # User/item LabelEncoders + isbn_to_idx map
│   └── books_filtered.parquet             # Filtered book metadata
│
├── 📊 Dashboard/
│   ├── app.py                             # Streamlit dashboard application
│   └── requirements.txt                   # Python dependencies for deployment
│
├── 🗄️ Dataset/
│   ├── Books.csv
│   ├── Ratings.csv
│   └── Users.csv
│
├── 🗃️ SQL/
│   ├── 01_schema.sql
│   ├── 02_insert_users.sql
│   ├── 03_insert_books.sql
│   ├── 04_insert_ratings.sql
│   └── import_guide.md
│
└── .gitignore
```

---

## 📊 Dataset

This project uses the **[Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)** collected by Cai-Nicolas Ziegler.

| File | Records | Description |
|---|---|---|
| `Books.csv` | 271,360 | Book metadata (title, author, year, publisher, cover URLs) |
| `Ratings.csv` | 1,149,780 | User ratings (0 = implicit, 1–10 = explicit) |
| `Users.csv` | 278,858 | Anonymized user demographics |

**After preprocessing:**
- ✅ 13,339 books with ≥10 interactions
- ✅ 6,827 active users with ≥5 explicit ratings
- ✅ ~433K explicit rating interactions used for training

---

## 🧠 Methodology

### Pipeline

```
Raw Data → EDA → Preprocessing → Feature Engineering
    ↓
Content-Based (TF-IDF)   Collaborative (ALS)   Model-Based (SVD++)
    ↓                           ↓                       ↓
Cosine Similarity       User-Item Matrix         Rating Prediction
    └───────────────────────────┴───────────────────────┘
                                ↓
                    Weighted Score Fusion
                                ↓
                    Hybrid Recommendations
```

### Models

**1. Content-Based Filtering (TF-IDF)**
- Feature engineering on `title + author + year` → TF-IDF matrix (13,339 × 19,366 vocabulary)
- Cosine similarity between item vectors for recommendation
- Stored as sparse `.npz` (~1 MB) vs dense `.npy` (>1.9 GB)

**2. Implicit ALS Collaborative Filtering**
- Confidence matrix: `c_ui = 1 + 40 × r_ui`
- 64 latent factors, 15 iterations, λ=0.01
- Trained with `implicit` library on user × item CSR matrix

**3. SVD++ (Surprise)**
- Explicit rating prediction (1–10 scale)
- Incorporates implicit feedback via additional item factor set
- Trained with `scikit-surprise` using cross-validation

**4. Hybrid Fusion**
- Weighted linear combination of normalised scores from all three models
- Configurable weights (default: ALS 40% / SVD++ 40% / Content 20%)

---

## 📈 Results

| Model | Metric | Value |
|---|---|---|
| SVD++ | RMSE | ~1.42 |
| SVD++ | MAE | ~1.05 |
| ALS | Precision@10 | ~0.21 |
| ALS | Recall@10 | ~0.18 |
| Content-Based | Coverage | 100% |

---

## 🛠️ Tech Stack

| Category | Library |
|---|---|
| Data Processing | `pandas`, `numpy`, `scipy` |
| Machine Learning | `scikit-learn`, `implicit`, `scikit-surprise` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |
| Data Storage | `parquet` (pyarrow), `pickle`, `scipy.sparse` |

---

## ⚙️ Installation & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/NumiKun/Book-Recommendation-System.git
cd Book-Recommendation-System
```

### 2. Install dependencies
```bash
pip install -r Dashboard/requirements.txt
```

### 3. Download the dataset
Download the [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) and place the CSV files in the `Dataset/` folder.

### 4. Run the Jupyter Notebook
Open and run `Notebook/book_recommendation_system.ipynb` to train all models. The trained artifacts will be saved to the `Model/` folder.

> **Note:** Model files are already included in this repository (pre-trained). You can skip step 4 and go straight to the dashboard if you only want to explore recommendations.

### 5. Launch the Streamlit dashboard
```bash
streamlit run Dashboard/app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 🗃️ SQL Schema

The `SQL/` folder contains scripts to load the dataset into a relational database for analysis:

```bash
# Import order
01_schema.sql          # Create tables (users, books, ratings)
02_insert_users.sql    # Load user data
03_insert_books.sql    # Load book metadata
04_insert_ratings.sql  # Load rating interactions
```

Refer to `SQL/import_guide.md` for detailed instructions.

---

## 📝 Notebook

The Jupyter notebook [`Notebook/book_recommendation_system.ipynb`](Notebook/book_recommendation_system.ipynb) covers the full ML pipeline:

1. 📦 Library Imports
2. 📂 Dataset Loading
3. 🔍 Exploratory Data Analysis
4. 🧹 Data Preprocessing
5. 🎯 Content-Based Filtering (TF-IDF)
6. 🤝 Collaborative Filtering (ALS)
7. ⭐ Model-Based CF (SVD++)
8. 📊 Model Evaluation
9. 🔀 Hybrid Recommendation Engine
10. 🧪 Inference & Demo
11. 📉 Latent Factor Visualisation
12. 💾 Model Serialisation
13. 📋 Conclusion

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests for:
- Bug fixes
- New recommendation algorithms
- UI improvements to the dashboard
- Additional dataset integrations

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **[Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)** by Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, Georg Lausen
- **[implicit](https://github.com/benfred/implicit)** by Ben Frederickson
- **[scikit-surprise](https://surpriselib.com/)** by Nicolas Hug

---

<div align="center">
Made with ❤️ by <a href="https://github.com/NumiKun">NumiKun</a>
</div>
