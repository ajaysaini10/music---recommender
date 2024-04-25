import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset preparation
songs = pd.read_csv('content based recommedation system/songdata.csv')
songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
songs['text'] = songs['text'].str.replace(r'\n', '')

# TF-IDF vectorization
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
lyrics_matrix = tfidf.fit_transform(songs['text'])

# Calculate cosine similarities
cosine_similarities = cosine_similarity(lyrics_matrix)

# Prepare similarities dictionary
similarities = {}
for i in range(len(cosine_similarities)):
    similar_indices = cosine_similarities[i].argsort()[:-50:-1]
    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]

# Content-based recommender class
class ContentBasedRecommender:
    def __init__(self, similarities):
        self.similarities = similarities

    def recommend(self, song_name, number_songs=5):
        if song_name not in self.similarities:
            return f"Error: The song '{song_name}' is not in the dataset."

        recom_song = self.similarities[song_name][:number_songs]
        return recom_song
