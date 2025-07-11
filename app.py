import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
anime_df = pd.read_csv("anime.csv")
anime_df['genre'] = anime_df['genre'].fillna('')
anime_df['rating'] = anime_df['rating'].fillna(0)
anime_df['episodes'] = anime_df['episodes'].replace('Unknown', 0).fillna(0)
anime_df['episodes'] = anime_df['episodes'].astype(str).str.extract('(\d+)')
anime_df['episodes'] = pd.to_numeric(anime_df['episodes'], errors='coerce').fillna(0).astype(int)
anime_df.reset_index(drop=True, inplace=True)

# Vectorize genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime_df['genre'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Recommendation function
def get_recommendations(anime_name, num=5):
    anime_name = anime_name.lower()
    matches = anime_df[anime_df['name'].str.lower() == anime_name]
    if matches.empty:
        return []

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num + 1]
    anime_indices = [i[0] for i in sim_scores]
    return anime_df.iloc[anime_indices][['name', 'rating', 'episodes']]


# Fetch poster from Anilist
def fetch_poster_url(anime_name):
    query = '''
    query ($search: String) {
      Media (search: $search, type: ANIME) {
        coverImage {
          large
        }
      }
    }
    '''
    variables = {'search': anime_name}
    url = 'https://graphql.anilist.co'
    response = requests.post(url, json={'query': query, 'variables': variables})
    if response.status_code == 200:
        try:
            return response.json()['data']['Media']['coverImage']['large']
        except:
            return None
    return None


# Streamlit UI
st.title("🎴 Anime Recommendation System")
anime_input = st.text_input("Enter an anime name (e.g. Naruto, Death Note)", "Naruto")

if st.button("Recommend"):
    recommendations = get_recommendations(anime_input)

    if len(recommendations) == 0:
        st.error("Anime not found in dataset.")
    else:
        st.success(f"Top recommendations based on '{anime_input.title()}':")
        for index, row in recommendations.iterrows():
            st.subheader(row['name'])
            st.write(f"⭐ Rating: {row['rating']}  🎬 Episodes: {row['episodes']}")
            poster_url = fetch_poster_url(row['name'])
            if poster_url:
                st.image(poster_url, width=200)
            else:
                st.info("Poster not found")
