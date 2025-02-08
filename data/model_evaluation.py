from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

from app.scripts.logger import logging
from app.scripts.pickler import save_object, load_object


def evaluate_model():
    logging.info("Importing data for evaluation")

    train_interactions = load_object('data/processed/train_interactions.pkl')
    test_interactions = load_object('data/processed/test_interactions.pkl')
    item_features = load_object('data/processed/item_features.pkl')
    model = load_object('models/lightfm_model.pkl')

    logging.info("Scoring precision")
    train_precision = precision_at_k(model, train_interactions, item_features=item_features, k=10).mean()
    test_precision = precision_at_k(model, test_interactions, k=10, item_features=item_features).mean()

    logging.info("Scoring AUC")
    train_auc = auc_score(model, train_interactions, item_features=item_features).mean()
    test_auc = auc_score(model, test_interactions, item_features=item_features).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

    print("Model evaluating completed!")
