# custom_loss.py
import numpy as np

def exp_weight_rmse_loss(preds, train_data, alpha=None):
    if alpha == None:
        alpha = 0.2
    y = train_data.get_label()
    p = preds
    w = np.exp(alpha * p)
    d = p - y
    grad = w*(2*d + alpha*d**2)
    hess = w*(2 + 2*alpha*d + alpha**2*d**2)
    return grad, hess

def weighted_rmse_eval(preds, dataset, alpha = None):
    """
    LightGBM custom metric: weighted RMSE using exponential weights w = exp(alpha * preds).
    Returns (name, value, is_higher_better=False).
    """
    y = dataset.get_label()
    p = preds
    alpha = 0.2 if alpha is None else alpha

    # compute weights
    w = np.exp(alpha * p)
    # normalize weights so sum to 1
    w = w / np.sum(w)

    # compute weighted MSE
    mse = np.dot(w, (p - y)**2)
    # RMSE
    rmse = np.sqrt(mse)
    return 'w_rmse', float(rmse), False

def compute_wpcc(preds, y):
    """
    Standalone computation of weighted Pearson correlation (WPCC) between preds and y.
    Weights per sample = 0.5 ** (rank/(n-1)), where rank sorted by preds descending.
    Returns the correlation value.
    """
    p = np.asarray(preds)
    y = np.asarray(y)
    n = p.shape[0]
    if n < 2:
        return 0.0
    order = np.argsort(-y)
    decay = 0.5 ** (np.arange(n) / (n - 1))
    w = np.empty(n, dtype=float)
    w[order] = decay
    w /= w.sum()
    p_mean = np.dot(w, p)
    y_mean = np.dot(w, y)
    cov = np.dot(w, (p - p_mean) * (y - y_mean))
    var_p = np.dot(w, (p - p_mean)**2)
    var_y = np.dot(w, (y - y_mean)**2)
    return cov / np.sqrt(var_p * var_y + 1e-16)