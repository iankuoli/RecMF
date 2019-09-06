
from scipy.sparse import *
import numpy as np
import heapq
import operator


def precision_n_recall(mat_test, mat_train, model, uid, top_ks):
    vec_test_uid = mat_test[uid, :]
    vec_train_uid = mat_train[uid, :]
    if vec_train_uid.nnz == 0:
        return 0, 0

    # Process predicted result
    _, item_ids, values = find(vec_train_uid)
    vec_pred_items = model.infer_user_consumption(uid)[:, 0].numpy()
    vec_pred_items[item_ids] = -1e10
    vec_pred_items = list(vec_pred_items)
    topk_pred_index = list(zip(*heapq.nlargest(max(top_ks), enumerate(vec_pred_items), key=operator.itemgetter(1))))[0]
    topk_pred_index = list(topk_pred_index)

    # Process ground truth result
    _, item_ids, values = find(vec_test_uid)
    topk_grnd_index = list(zip(*heapq.nlargest(max(top_ks), enumerate(values), key=operator.itemgetter(1))))[0]
    topk_grnd_index = set(item_ids[list(topk_grnd_index)])
    grnd_size = len(topk_grnd_index)

    precision = np.zeros(len(top_ks))
    recall = np.zeros(len(top_ks))
    index = 0
    for topk in top_ks:
        hits = set(topk_pred_index[:topk]) & topk_grnd_index
        prec_at_topk = len(hits) / topk
        recl_at_topk = len(hits) / grnd_size
        precision[index] = prec_at_topk
        recall[index] = recl_at_topk
        index += 1

    return precision, recall


def evaluate(mat_test, mat_train, model, user_ids, top_ks, use_prec_n_recl=True):

    ret = {}
    if use_prec_n_recl:
        mat_prec = np.zeros([len(user_ids), len(top_ks)])
        mat_recl = np.zeros([len(user_ids), len(top_ks)])
        u_index = 0
        for uid in user_ids:
            mat_prec[u_index, :], mat_recl[u_index, :] = precision_n_recall(mat_test, mat_train, model, uid, top_ks)
            u_index += 1
        avg_prec = np.mean(mat_prec, axis=0)
        avg_recl = np.mean(mat_recl, axis=0)
        ret['precision'] = avg_prec
        ret['recall'] = avg_recl

    return ret

