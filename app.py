import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Anime Recommender", layout="wide")

# Load anime data
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df['genre'] = df['genre'].fillna('')
    df['rating'] = df['rating'].fillna(0)
    df['episodes'] = df['episodes'].replace('Unknown', 0).fillna(0)
    df['episodes'] = df['episodes'].astype(str).str.extract('(\d+)')
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce').fillna(0).astype(int)
    return df.reset_index(drop=True)

anime_df = load_data()

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime_df['genre'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation logic
def get_recommendations(anime_name, num=5):
    anime_name = anime_name.strip().lower()
    matches = anime_df[anime_df['name'].str.lower() == anime_name]

    if matches.empty:
        return pd.DataFrame()  # Always return DataFrame

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    anime_indices = [i[0] for i in sim_scores]

    return anime_df.iloc[anime_indices][['name', 'rating', 'episodes']].reset_index(drop=True)

# Poster fetch
def fetch_poster_url(anime_name):
    try:
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
            return response.json()['data']['Media']['coverImage']['large']
    except:
        pass
    return None

# Streamlit UI
st.title("🎴 Anime Recommendation System")
anime_input = st.text_input("🔍 Enter an anime name", value="Naruto")

if st.button("Recommend"):
    st.markdown("### 🔍 Searching...")

    try:
        recommendations = get_recommendations(anime_input)

        if recommendations.empty:
            st.error("❌ Anime not found. Try checking spelling or try another title.")
        else:
            st.success(f"🎯 Top recommendations based on: **{anime_input.title()}**")
            for _, row in recommendations.iterrows():
                st.subheader(row['name'])
                st.write(f"⭐ **Rating:** {row['rating']} &nbsp;&nbsp;&nbsp; 🎬 **Episodes:** {row['episodes']}")
                poster_url = fetch_poster_url(row['name'])
                if poster_url:
                    st.image(poster_url, width=200)
                else:
                    st.warning("🖼 Poster not found")
                st.markdown("---")

    except Exception as e:
        st.exception(f"🔥 Unexpected error:\n\n{str(e)}")
