import numpy as np
from keras import backend as K
from sklearn.metrics import roc_auc_score


def pairwise_precision_recall_f1(preds, truths, sparse=False):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples, = labels_true.shape

    c = contingency_matrix(labels_true, labels_pred, sparse=True)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    precision = tk / pk
    recall = tk / qk
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1

def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency

def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred

def cal_f1(prec, rec):
    return 2*prec*rec/(prec+rec)

def get_hidden_output(model, inp):
    get_activations = K.function(model.inputs[:1] + [K.learning_phase()], [model.layers[5].get_output_at(0), ])
    activations = get_activations([inp, 0])
    return activations[0]


def predict(anchor_emb, test_embs):
    score1 = np.linalg.norm(anchor_emb-test_embs[0])
    score2 = np.linalg.norm(anchor_emb-test_embs[1])
    return [score1, score2]


def full_auc(model, test_triplets):
    """
    Measure AUC for model and ground truth on all items.
    Returns:
    - float AUC
    """

    grnds = []
    preds = []
    preds_before = []
    embs_anchor, embs_pos, embs_neg = test_triplets

    inter_embs_anchor = get_hidden_output(model, embs_anchor)
    inter_embs_pos = get_hidden_output(model, embs_pos)
    inter_embs_neg = get_hidden_output(model, embs_neg)
    # print(inter_embs_pos.shape)

    accs = []
    accs_before = []

    for i, e in enumerate(inter_embs_anchor):
        if i % 10000 == 0:
            print('test', i)

        emb_anchor = e
        emb_pos = inter_embs_pos[i]
        emb_neg = inter_embs_neg[i]
        test_embs = np.array([emb_pos, emb_neg])

        emb_anchor_before = embs_anchor[i]
        emb_pos_before = embs_pos[i]
        emb_neg_before = embs_neg[i]
        test_embs_before = np.array([emb_pos_before, emb_neg_before])

        predictions = predict(emb_anchor, test_embs)
        predictions_before = predict(emb_anchor_before, test_embs_before)

        acc_before = 1 if predictions_before[0] < predictions_before[1] else 0
        acc = 1 if predictions[0] < predictions[1] else 0
        accs_before.append(acc_before)
        accs.append(acc)

        grnd = [0, 1]
        grnds += grnd
        preds += predictions
        preds_before += predictions_before

    auc_before = roc_auc_score(grnds, preds_before)
    auc = roc_auc_score(grnds, preds)
    print('test accuracy before', np.mean(accs_before))
    print('test accuracy after', np.mean(accs))

    print('test AUC before', auc_before)
    print('test AUC after', auc)
    return auc