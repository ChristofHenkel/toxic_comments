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



test_predicts = np.multiply(*test_predicts_list)
test_predicts **= (1. / len(test_predicts_list))
test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT