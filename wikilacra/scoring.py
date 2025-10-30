from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    confusion_matrix,
    f1_score,
)

tpr_score = recall_score  # TPR and recall are the same metric


def fpr_score(y, y_pred, neg_label, pos_label):
    """False positive rate

    Args:
        y: Trues
        y_pred: Predicteds
        neg_label: Negative defn
        pos_label: Pos defn

    Returns:
        float: False positive rate
    """
    cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    tn, fp, _, _ = cm.ravel()
    tnr = tn / (tn + fp)
    return 1 - tnr


pos_label = 1
neg_label = 0
scoring = {
    "precision": make_scorer(precision_score, pos_label=pos_label),
    "recall": make_scorer(recall_score, pos_label=pos_label),
    "fpr": make_scorer(fpr_score, neg_label=neg_label, pos_label=pos_label),
    "tpr": make_scorer(tpr_score, pos_label=pos_label),
    "f1": make_scorer(f1_score, pos_label=pos_label),
}
