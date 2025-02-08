from lightfm import LightFM

from app.scripts.logger import logging
from app.scripts.pickler import save_object, load_object


def train_model():
    logging.info("Importing data for training")
    train_interactions = load_object('data/processed/train_interactions.pkl')
    item_features = load_object('data/processed/item_features.pkl')

    logging.info("Init model")
    model = LightFM(loss='warp', learning_rate=0.15, no_components=30)
    logging.info("Fitting model")
    model.fit(train_interactions, item_features=item_features, epochs=30, num_threads=4)


    logging.info("Saving model")
    save_object('models/lightfm_model.pkl', model)

    print("Model training completed!")
