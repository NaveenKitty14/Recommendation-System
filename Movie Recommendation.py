import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval

# Load the datasets
path = r"C:\Users\navee\OneDrive\Documents\Recommendation--master\dataset"
credits_df = pd.read_csv(path + r"\tmdb_5000_credits.csv")
movies_df = pd.read_csv(path + r"\tmdb_5000_movies.csv")

# Debugging: Check columns before merge
print("Movies Columns before merge:", movies_df.columns)
print("Credits Columns before merge:", credits_df.columns)

# Fix column names in credits_df
credits_df.columns = ['id', 'title', 'cast', 'crew']  # Ensure 'title' is correctly named

# Merge datasets on 'id'
movies_df = movies_df.merge(credits_df, on="id", how="left")

# Debugging: Check columns after merge
print("Movies Columns after merge:", movies_df.columns)

# Debugging: Check first few rows of merged dataset
print(movies_df.head())

# Verify 'title' exists in movies_df
if 'title' not in movies_df.columns:
    raise ValueError("The 'title' column is missing. Check your dataset or merge logic.")
else:
    print("Merge successful, 'title' column is present.")

# Demographic Filtering
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (v + m) * C)

new_movies_df = movies_df[movies_df["vote_count"] >= m].copy()
new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
new_movies_df = new_movies_df.sort_values("score", ascending=False)

# Plot Top 10 Movies
def plot_top_movies():
    popularity = movies_df.sort_values("popularity", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Movies by Popularity")
    plt.xlabel("Popularity")
    plt.show()

plot_top_movies()

# Content-Based Filtering (Overview-Based)
tfidf = TfidfVectorizer(stop_words="english")
movies_df["overview"] = movies_df["overview"].fillna("")
tfidf_matrix = tfidf.fit_transform(movies_df["overview"])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [ind[0] for ind in sim_scores]
    return movies_df["title"].iloc[movie_indices]

print("Recommendations for 'The Dark Knight Rises':")
print(get_recommendations("The Dark Knight Rises"))

print("\nRecommendations for 'The Avengers':")
print(get_recommendations("The Avengers"))

# Enhanced Content-Based Filtering (Metadata-Based)
features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i.get("job") == "Director":
            return i["name"]
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        return names[:3] if len(names) > 3 else names
    return []

movies_df["director"] = movies_df["crew"].apply(get_director)

for feature in ["cast", "keywords", "genres"]:
    movies_df[feature] = movies_df[feature].apply(get_list)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    return ""

for feature in ["cast", "keywords", "director", "genres"]:
    movies_df[feature] = movies_df[feature].apply(clean_data)

movies_df["soup"] = movies_df.apply(
    lambda x: " ".join(x["keywords"]) + " " + " ".join(x["cast"]) + " " + x["director"] + " " + " ".join(x["genres"]),
    axis=1
)

count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

print("\nEnhanced Recommendations for 'The Dark Knight Rises':")
print(get_recommendations("The Dark Knight Rises", cosine_sim2))
