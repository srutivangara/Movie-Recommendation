
import streamlit as st
import pandas as pd
import re
import ast
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page configuration
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.write("Find movies with similar casts and crew!")

# 2. Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_ds.csv")
    df["title"] = df["title"].fillna("")
    df["cast"] = df["cast"].fillna("[]")
    df["crew"] = df["crew"].fillna("[]")
    return df

df = load_data()

# 3. Extract names from cast and crew
def extract_names(value):
    try:
        people = ast.literal_eval(value)
        if isinstance(people, list):
            return " ".join(
                str(person.get("name", ""))
                for person in people
                if isinstance(person, dict)
            )
    except (ValueError, SyntaxError, TypeError):
        pass
    return ""

df["cast_names"] = df["cast"].apply(extract_names)
df["crew_names"] = df["crew"].apply(extract_names)

# 4. Combine and clean text
df["movie_features"] = (
    df["title"] + " " + df["cast_names"] + " " + df["crew_names"]
)

stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["movie_features"].apply(clean_text)

# 5. Create TF-IDF matrix
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["clean_text"])

# 6. Define recommendation function BEFORE calling it
def recommend(item_name, top_n=5):
    matches = df.index[
        df["title"].str.strip().str.lower()
        == item_name.strip().lower()
    ].tolist()

    if not matches:
        return pd.DataFrame(
            columns=["title", "similarity_score"]
        )

    movie_index = matches[0]

    scores = cosine_similarity(
        tfidf_matrix[movie_index], tfidf_matrix
    ).flatten()

    similar_indices = scores.argsort()[::-1]
    similar_indices = [
        i for i in similar_indices if i != movie_index
    ][:top_n]

    return pd.DataFrame({
        "title": df.iloc[similar_indices]["title"].values,
        "similarity_score": scores[similar_indices]
    }).reset_index(drop=True)

# 7. Movie dropdown
movie_list = sorted(df["title"].dropna().unique().tolist())

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

# 8. Recommendation button
if st.button("Get Recommendations"):
    results = recommend(selected_movie, top_n=5)

    if results.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader(f"Movies similar to {selected_movie}")

        for _, row in results.iterrows():
            st.write(f"🎥 **{row['title']}**")
            st.caption(
                f"Similarity score: {row['similarity_score']:.3f}"
            )
