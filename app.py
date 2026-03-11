import streamlit as st
import pandas as pd
import difflib
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# Background image function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .movie-box {{
            background: rgba(0,0,0,0.65);
            padding: 12px;
            border-radius: 10px;
            margin: 6px;
            color: white;
            font-size: 20px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
add_bg_from_local("download.jpg")

# BIG Title
st.markdown(
    "<h1 style='text-align:center; font-size:90px; color:white;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

# Subtitle
st.markdown(
    "<h3 style='text-align:center; color:white;'>Discover movies similar to your favourite ones</h3>",
    unsafe_allow_html=True
)

st.write("")

# Load dataset
movies_data = pd.read_csv("movies.csv")

# Selected features
selected_features = ['genres','keywords','tagline','cast','director']

# Fill missing values
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features
combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)

# Vectorization
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Similarity matrix
similarity = cosine_similarity(feature_vectors)

# Movie dropdown
movie_name = st.selectbox(
    "🎥 Select your favourite movie",
    movies_data['title'].values
)

# Recommendation button
if st.button("Recommend Movies"):

    list_of_titles = movies_data['title'].tolist()

    find_close_match = difflib.get_close_matches(movie_name, list_of_titles)

    if len(find_close_match) == 0:
        st.write("Movie not found")

    else:

        close_match = find_close_match[0]

        index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]

        similarity_score = list(enumerate(similarity[index_of_movie]))

        sorted_movies = sorted(similarity_score, key=lambda x:x[1], reverse=True)

        st.markdown("<h2 style='color:white;'>🎬 Recommended Movies</h2>", unsafe_allow_html=True)

        i = 1

        for movie in sorted_movies[1:11]:

            index = movie[0]

            title = movies_data[movies_data.index == index]['title'].values[0]

            st.markdown(
                f'<div class="movie-box">{i}. {title}</div>',
                unsafe_allow_html=True
            )

            i += 1