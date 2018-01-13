from sklearn.metrics import log_loss
import numpy as np

def multi_log_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(log_loss(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()