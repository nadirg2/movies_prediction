import pandas as pd
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split

from app.scripts.logger import logging
from app.scripts.pickler import save_object


def preprocess_data():
    logging.info("Importing data for preprocessing")
    rating = pd.read_csv('data/raw/rating.csv')
    movie = pd.read_csv('data/raw/movie.csv')

    logging.info("Starting data preprocessing")
    
    rating['timestamp'] = pd.to_datetime(rating['timestamp'])
    rating['day_of_week'] = rating['timestamp'].dt.weekday
    rating['hour_of_day'] = rating['timestamp'].dt.hour
    rating['month'] = rating['timestamp'].dt.month
    

    movie['genres'] = movie['genres'].str.split('|')

    genres = set(g for sublist in movie['genres'] for g in sublist)
    for genre in genres:
        movie[genre] = movie['genres'].apply(lambda x: int(genre in x))


    logging.info("Getting item features dataframe")
    item_features_df = movie.drop(['title', 'genres'], axis=1)

    dataset = Dataset()
    dataset.fit(rating['userId'].unique(), movie['movieId'].unique(),
                item_features=item_features_df.columns.tolist())

    item_id_mapping = dataset.mapping()[2]
    index_to_movie_id = {index: movie_id for movie_id, index in item_id_mapping.items()}

    logging.info("Generating interactions coo matrix")
    (interactions, weights) = dataset.build_interactions(rating[['userId', 'movieId', 'rating']].values)

    logging.info("Generating item features csr matrix")
    item_features = dataset.build_item_features([
        (row.movieId, dict(zip(item_features_df.columns[1:], row[1:].values))) # Changed this line
        for _, row in item_features_df.iterrows()
    ])

    logging.info("Splitting interactions coo matrix into train and test")
    train_interactions, test_interactions = random_train_test_split(interactions, test_percentage=0.2, random_state=42)

    rating.to_csv('data/processed/rating.csv', index=False)
    movie.to_csv('data/processed/movie.csv', index=False)


    logging.info("Saving train interactions")
    save_object('data/processed/train_interactions.pkl', train_interactions)
    
    logging.info("Saving test interactions")
    save_object('data/processed/test_interactions.pkl', test_interactions)
    
    logging.info("Saving index to movie_id")
    save_object('data/processed/index_to_movie_id.pkl', index_to_movie_id)

    logging.info("Saving movies features")
    save_object('data/processed/item_features.pkl', item_features)

    logging.info("Data preprocessing completed")


