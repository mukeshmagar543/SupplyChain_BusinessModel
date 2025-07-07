from sklearn.metrics import r2_score

def evaluate_model(y_true, y_pred):
    return r2_score(y_true, y_pred)