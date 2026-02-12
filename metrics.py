from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

def calculate_metrics(y_true, y_pred, y_scores=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_scores is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['auc'] = 0.0
            
    return metrics
