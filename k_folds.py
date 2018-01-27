import numpy as np
import pandas as pd

def train_folds(X, y, fold_count, batch_size, get_model_func):
    fold_size = len(X) // fold_count
    models = []
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]

        model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y)
        models.append(model)

    return models

csv_files = ['model_e4v0.0409t0.0492.csv',
             'model_e7v0.0403t0.0456.csv',
             'model_e9v0.0409t0.0433.csv',
             'model_e8v0.0406t0.0445.csv',
             'model_e6v0.0402t0.0467.csv',
             'model_e5v0.0405t0.0479.csv',
]

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

test_predicts_list = []
for csv_file in csv_files:
    orig_submission = pd.read_csv('submissions/pavel20/' + csv_file)
    predictions = orig_submission[list_classes]
    test_predicts_list.append(predictions)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

#test_predicts = np.multiply(*test_predicts_list)
test_predicts **= (1. / len(test_predicts_list))

new_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
new_submission[list_classes] = test_predicts
new_submission.to_csv("submissions/pavel20/fold_e4-e9.csv", index=False)