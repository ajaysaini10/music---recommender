from flask import Flask, render_template, request
import pandas as pd
from recommender import ContentBasedRecommender
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask application
app = Flask(__name__)

# Load dataset and prepare the similarities
# Update the path as needed to ensure the dataset is correct
songs = pd.read_csv('content based recommedation system/songdata.csv')  
songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
songs['text'] = songs['text'].str.replace(r'\n', '', regex=True)

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
lyrics_matrix = tfidf.fit_transform(songs['text'])

# Compute cosine similarities
cosine_similarities = cosine_similarity(lyrics_matrix)

# Create similarities dictionary
similarities = {}
for i in range(len(cosine_similarities)):
    similar_indices = cosine_similarities[i].argsort()[:-50:-1]
    song_name = songs['song'].iloc[i]  # Get the song name from the dataset
    similarities[song_name] = [
        (cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices
    ][1:]  # Exclude the first item because it's the same song

# Initialize the recommender
recommender = ContentBasedRecommender(similarities)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    song_name = request.form['song_name'].strip().lower()  # Normalize case and remove extra spaces
    num_recommendations = int(request.form.get('num_recommendations', 5))

    # Convert keys to lowercase to ensure case-insensitive matching
    similarities_keys = [key.lower() for key in recommender.similarities.keys()]

    # Check if the song name is in the similarities dictionary
    if song_name not in similarities_keys:
        return f"Error: The song '{song_name}' is not in the dataset.", 400  # Return 400 Bad Request

    # Find the exact key in the similarities dictionary to fetch recommendations
    original_song_name = next(key for key in recommender.similarities.keys() if key.lower() == song_name)

    recommendations = recommender.recommend(original_song_name, num_recommendations)

    return render_template(
        'recommendations.html', 
        song=original_song_name, 
        recommendations=recommendations
    )

if __name__ == '__main__':
    app.run(debug=True)
